#!/usr/bin/env python3
"""
eval_eot_cpt_logistic.py

Evaluate a trained EventEOTSummaryCPTModel backbone with a logistic-regression
probe on EHRSHOT downstream tasks.

Pipeline:
  1. Load the fine-tuned AutoModelForCausalLM backbone (HF format).
  2. For each sample, run a forward pass with the event-EOT summary attention mask
     (same mask used during CPT training) and extract hidden states.
  3. Pool hidden states at EOT positions (pad token positions inside valid span)
     → per-patient fixed-dim embedding.
  4. Per task: train sklearn LogisticRegression on the "train" split embeddings,
     evaluate on val / test, report AUROC and AUPRC.
  5. Write a JSON results file.

Usage example:
  python eval_eot_cpt_logistic.py \
      --checkpoint_dir exps/hx1/ckpts/ehr_event_eot_cpt/ehr-event-eot-cpt/554535/best \
      --eval_data_dir hx1/eval_data_tokenized_eot_2048 \
      --output_dir results/eot_cpt_probe \
      --bf16 \
      --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

TASKS = [
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EvalParquetDataset(Dataset):
    def __init__(self, parquet_path: str):
        table = pq.read_table(
            parquet_path,
            columns=["patient_id", "label", "input_ids", "attention_mask", "event_ids"],
            memory_map=True,
        )
        d = table.to_pydict()
        self.patient_ids  = d["patient_id"]
        self.labels       = d["label"]
        self.input_ids    = d["input_ids"]
        self.attention_mask = d["attention_mask"]
        self.event_ids    = d["event_ids"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "patient_id":    self.patient_ids[idx],
            "label":         int(self.labels[idx]),
            "input_ids":     self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "event_ids":     self.event_ids[idx],
        }


def collate_fn(samples: List[dict]) -> dict:
    """Pad variable-length sequences to the max length in the batch."""
    max_len = max(len(s["input_ids"]) for s in samples)
    batch_input_ids    = []
    batch_attn_mask    = []
    batch_event_ids    = []
    batch_labels       = []
    batch_patient_ids  = []

    for s in samples:
        n = len(s["input_ids"])
        pad = max_len - n
        batch_input_ids.append(s["input_ids"] + [0] * pad)
        batch_attn_mask.append(s["attention_mask"] + [0] * pad)
        batch_event_ids.append(s["event_ids"] + [-1] * pad)
        batch_labels.append(s["label"])
        batch_patient_ids.append(s["patient_id"])

    return {
        "patient_id":    batch_patient_ids,
        "label":         torch.tensor(batch_labels, dtype=torch.long),
        "input_ids":     torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attn_mask, dtype=torch.long),
        "event_ids":     torch.tensor(batch_event_ids, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Attention mask (same as EventEOTSummaryCPTModel)
# ---------------------------------------------------------------------------

def build_event_summary_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    event_ids: torch.Tensor,
    eos_token_id: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Replicate the CPT training attention mask for inference."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    valid      = attention_mask.bool()
    pos        = torch.arange(seq_len, device=device)
    causal     = pos.view(1, 1, seq_len) <= pos.view(1, seq_len, 1)
    same_event = event_ids[:, :, None] == event_ids[:, None, :]
    eos_keys   = ((input_ids == eos_token_id) & valid)[:, None, :]
    q_valid    = valid[:, :, None]
    k_valid    = valid[:, None, :]

    allowed = ((same_event & causal) | (eos_keys & causal)) & q_valid & k_valid

    # padding positions attend to themselves to keep attention numerically stable
    eye     = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
    allowed = allowed | ((~valid)[:, :, None] & eye)

    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=model_dtype, device=device)
    mask = mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(model_dtype).min)
    return mask


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _pool_embeddings(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    pad_token_id: int,
    pool: str,
) -> torch.Tensor:
    """
    Pool hidden states (B, T, D) → (B, D).

    pool='eot'      — mean over all valid EOT (pad token) positions.
    pool='last_eot' — hidden state at the last valid EOT position;
                      falls back to the last valid token if no EOT exists.
    pool='mean'     — mean over all valid token positions.
    """
    B, T, D = hidden.shape
    device  = hidden.device

    eot_pos = ((input_ids == pad_token_id) & attn_mask.bool())  # (B, T)

    if pool == "eot":
        eot_sum = (hidden * eot_pos.unsqueeze(-1).float()).sum(1)        # (B, D)
        eot_cnt = eot_pos.float().sum(1, keepdim=True).clamp(min=1)     # (B, 1)
        return eot_sum / eot_cnt

    if pool == "last_eot":
        pos = torch.arange(T, device=device).unsqueeze(0)               # (1, T)
        # Among EOT positions keep the index; elsewhere use -1 so argmax picks EOT.
        eot_indices = pos.masked_fill(~eot_pos, -1)                     # (B, T)
        last_eot    = eot_indices.argmax(dim=1)                         # (B,)
        # If a sample has no EOT at all, fall back to the last valid token.
        has_eot     = eot_pos.any(dim=1)                                # (B,)
        valid_indices = pos.masked_fill(~attn_mask.bool(), -1)
        last_valid    = valid_indices.argmax(dim=1)                     # (B,)
        idx = torch.where(has_eot, last_eot, last_valid)                # (B,)
        return hidden[torch.arange(B, device=device), idx]             # (B, D)

    # pool == "mean"
    valid_f = attn_mask.float().unsqueeze(-1)                           # (B, T, 1)
    return (hidden * valid_f).sum(1) / valid_f.sum(1).clamp(min=1)


@torch.no_grad()
def extract_embeddings(
    model: AutoModelForCausalLM,
    loader: DataLoader,
    pad_token_id: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    pool: str,
    attn_mask_type: str = "event_eot",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (embeddings, labels), shapes (N, D) and (N,).

    pool='eot'      — mean over hidden states at valid EOT (pad token) positions.
    pool='last_eot' — hidden state at the last event's EOT position.
    pool='mean'     — mean over all valid token positions.

    attn_mask_type='event_eot' — custom CPT mask (used during CPT training).
    attn_mask_type='causal'    — standard causal mask (for baseline models).
    """
    model_dtype = next(model.parameters()).dtype

    all_embs   = []
    all_labels = []

    for batch in tqdm(loader, desc="  embed", dynamic_ncols=True, leave=False):
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        event_ids    = batch["event_ids"].to(device)
        labels       = batch["label"]

        if attn_mask_type == "causal":
            effective_mask = attn_mask
        else:
            effective_mask = build_event_summary_attention_mask(
                input_ids, attn_mask, event_ids, pad_token_id, model_dtype
            )

        autocast_enabled = (device.type == "cuda") and (autocast_dtype is not None)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=effective_mask,
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state.float()  # (B, T, D)

        emb = _pool_embeddings(hidden, input_ids, attn_mask, pad_token_id, pool)

        all_embs.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate EventEOTSummaryCPTModel with a logistic-regression probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B",
                   help="Base model name (for tokenizer).")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Path to the HF checkpoint saved by train_ehr_event_eot_cpt.py.")
    p.add_argument("--eval_data_dir", required=True,
                   help="Root dir from build_eval_tokenized_eot_data.py.")
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_splits", nargs="+", default=["val", "test"])
    p.add_argument("--output_dir", default="results/eot_cpt_probe")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pool", default="eot", choices=["eot", "last_eot", "mean"],
                   help="Pooling: 'eot' = mean over EOT positions, "
                        "'last_eot' = last event EOT, 'mean' = all valid positions.")
    p.add_argument("--lr_c", type=float, default=1.0,
                   help="LogisticRegression C (inverse regularisation strength).")
    p.add_argument("--lr_max_iter", type=int, default=1000)
    p.add_argument("--no_scale", action="store_true",
                   help="Skip StandardScaler before logistic regression.")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--attn_implementation", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--attn_mask_type", default="event_eot", choices=["event_eot", "causal"],
                   help="'event_eot' = custom CPT mask; 'causal' = standard causal mask for baseline models.")
    return p.parse_args()


def load_split(
    eval_data_dir: Path,
    task: str,
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader | None:
    path = eval_data_dir / task / f"{split}.parquet"
    if not path.exists():
        return None
    ds = EvalParquetDataset(str(path))
    if len(ds) == 0:
        return None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    torch_dtype   = torch.float32
    autocast_dtype = None
    if args.bf16:
        torch_dtype    = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype    = torch.float16
        autocast_dtype = torch.float16

    logger.info("Loading tokenizer from %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id.")
    logger.info("EOT / pad token id: %d (%r)", pad_token_id, tokenizer.pad_token)

    logger.info("Loading backbone from %s", args.checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Backbone params: %.1fM", n_params)

    eval_data_dir = Path(args.eval_data_dir)
    all_results: Dict[str, dict] = {}

    for task in args.tasks:
        logger.info("=== Task: %s ===", task)
        task_results: Dict[str, dict] = {}

        # --- Build train embeddings ---
        train_loader = load_split(eval_data_dir, task, args.train_split, args.batch_size, args.num_workers)
        if train_loader is None:
            logger.warning("  No %s split for task %s — skipping", args.train_split, task)
            continue
        logger.info("  Extracting %s embeddings (%d samples)...", args.train_split, len(train_loader.dataset))
        train_embs, train_labels = extract_embeddings(
            model, train_loader, pad_token_id, device, autocast_dtype, args.pool,
            attn_mask_type=args.attn_mask_type,
        )
        logger.info("  train embs: %s  positives: %d / %d",
                    train_embs.shape, train_labels.sum(), len(train_labels))

        # --- Fit logistic regression ---
        scaler = None
        X_train = train_embs
        if not args.no_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_embs)

        clf = LogisticRegression(
            C=args.lr_c,
            max_iter=args.lr_max_iter,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(X_train, train_labels)
        logger.info("  LogReg trained (C=%.3g, balanced)", args.lr_c)

        train_probs = clf.predict_proba(X_train)[:, 1]
        train_auroc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else float("nan")
        train_auprc = average_precision_score(train_labels, train_probs) if len(set(train_labels)) > 1 else float("nan")
        logger.info("  [train] AUROC=%.4f  AUPRC=%.4f", train_auroc, train_auprc)
        task_results["train"] = {
            "auroc": train_auroc, "auprc": train_auprc,
            "n": int(len(train_labels)), "n_pos": int(train_labels.sum()),
        }

        # --- Evaluate on each requested split ---
        for split in args.eval_splits:
            loader = load_split(eval_data_dir, task, split, args.batch_size, args.num_workers)
            if loader is None:
                logger.warning("  No %s split for task %s — skipping", split, task)
                continue
            logger.info("  Extracting %s embeddings (%d samples)...", split, len(loader.dataset))
            embs, labels = extract_embeddings(
                model, loader, pad_token_id, device, autocast_dtype, args.pool,
                attn_mask_type=args.attn_mask_type,
            )

            X_eval = scaler.transform(embs) if scaler is not None else embs
            probs  = clf.predict_proba(X_eval)[:, 1]

            if len(set(labels)) > 1:
                auroc = roc_auc_score(labels, probs)
                auprc = average_precision_score(labels, probs)
            else:
                auroc = auprc = float("nan")
                logger.warning("  Only one class present in %s split — metrics are nan", split)

            logger.info("  [%s]   AUROC=%.4f  AUPRC=%.4f  (n=%d, pos=%d)",
                        split, auroc, auprc, len(labels), int(labels.sum()))
            task_results[split] = {
                "auroc": auroc, "auprc": auprc,
                "n": int(len(labels)), "n_pos": int(labels.sum()),
            }

        all_results[task] = task_results

    # --- Summary ---
    logger.info("\n===== Summary =====")
    for task, splits in all_results.items():
        for split, metrics in splits.items():
            logger.info(
                "  %-25s  %-5s  AUROC=%.4f  AUPRC=%.4f  (n=%d, pos=%d)",
                task, split,
                metrics["auroc"], metrics["auprc"],
                metrics["n"], metrics["n_pos"],
            )

    # --- Save ---
    result_cfg = {
        "checkpoint_dir":  str(args.checkpoint_dir),
        "eval_data_dir":   str(args.eval_data_dir),
        "attn_mask_type":  args.attn_mask_type,
        "pool":            args.pool,
        "train_split":     args.train_split,
        "lr_c":            args.lr_c,
        "results":         all_results,
    }
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(result_cfg, indent=2, ensure_ascii=False) + "\n")
    logger.info("Results saved -> %s", out_path)


if __name__ == "__main__":
    main()
