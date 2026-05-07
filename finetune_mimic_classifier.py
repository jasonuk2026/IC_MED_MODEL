#!/usr/bin/env python3
"""
Fine-tune a pretrained Event-EOT CPT model for MIMIC binary classification.

Architecture:
  CPT backbone (frozen or trainable) + Linear(hidden_size → 2) head
  Event-EOT attention mask applied during both fine-tuning and inference.

Matches the friend's TextCancEHR2 setup:
  - CrossEntropyLoss, no class weighting (AUROC as early-stopping metric handles imbalance)
  - Best checkpoint selected by AUROC on validation set
  - Early stopping with configurable patience
  - Saves per-split predictions to JSON for downstream analysis

Usage:
  python finetune_mimic_classifier.py \
      --pretrained_dir /path/to/cpt/final \
      --eval_parquet_dir /gpfs/home/zduan/codes/ehr/ordered_data/mimic_eval \
      --task icu_mortality \
      --output_dir /path/to/classifier_output/icu_mortality \
      --freeze_backbone \
      --epochs 10 --batch_size 8 --lr 1e-4 \
      --early_stopping_patience 5 \
      --bf16
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
torch.set_float32_matmul_precision("high")
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_warning()

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def maybe_init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank       = int(os.environ.get("RANK",       "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, local_rank, world_size

def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIMICEvalDataset(Dataset):
    """
    Reads a per-task evaluation parquet (from build_mimic_eval_parquet.py).
    Each sample is variable-length; padding is done in the collator.
    """
    def __init__(self, parquet_path: str):
        logger.info("Loading eval parquet: %s", parquet_path)
        table = pq.read_table(
            parquet_path,
            columns=["subject_id", "label", "input_ids", "attention_mask", "event_ids"],
            memory_map=True,
        )
        self.subject_ids   = table["subject_id"].to_pylist()
        self.labels        = table["label"].to_pylist()
        self.input_ids     = table["input_ids"].to_pylist()
        self.attention_masks = table["attention_mask"].to_pylist()
        self.event_ids     = table["event_ids"].to_pylist()
        pos = sum(self.labels)
        logger.info("  %d samples  pos=%d (%.1f%%)  neg=%d",
                    len(self.labels), pos, 100 * pos / len(self.labels), len(self.labels) - pos)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "subject_id":    self.subject_ids[idx],
            "label":         self.labels[idx],
            "input_ids":     self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "event_ids":     self.event_ids[idx],
        }


def collate_fn(batch: List[dict], pad_token_id: int) -> dict:
    """
    Dynamic right-padding to the longest sequence in the batch.
    event_ids are padded with -1.
    """
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids_padded    = []
    attention_mask_padded = []
    event_ids_padded    = []
    labels              = []
    subject_ids         = []

    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        input_ids_padded.append(   x["input_ids"]     + [pad_token_id] * pad)
        attention_mask_padded.append(x["attention_mask"] + [0]           * pad)
        event_ids_padded.append(   x["event_ids"]     + [-1]           * pad)
        labels.append(x["label"])
        subject_ids.append(x["subject_id"])

    return {
        "input_ids":      torch.tensor(input_ids_padded,     dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
        "event_ids":      torch.tensor(event_ids_padded,     dtype=torch.long),
        "labels":         torch.tensor(labels,               dtype=torch.long),
        "subject_ids":    subject_ids,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MIMICClassifier(nn.Module):
    """
    CPT backbone + linear classification head.

    The event-EOT 4D attention mask is re-constructed on the fly from event_ids,
    exactly as during pretraining. This ensures the hidden representations at
    inference time are consistent with those learned during CPT.

    Embedding from: last non-padding token's hidden state (same as TextCancEHR2).
    Loss: CrossEntropyLoss, no class weighting (AUROC handles imbalance at eval time).
    """
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int = 2,
        freeze_backbone: bool = True,
        pooling: str = "last_token",
    ):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.pooling = pooling

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone frozen. Only classification head is trainable.")
        else:
            logger.info("Backbone trainable (full fine-tuning).")

        self.classifier = nn.Linear(hidden_size, num_labels)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info("Trainable parameters: %.2fM", trainable)

    def _build_event_eot_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_ids: torch.Tensor,
        eos_token_id: int,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        valid = attention_mask.bool()

        # Cache static masks that only depend on seq_len.
        if not hasattr(self, "_causal_cache") or self._causal_cache.shape[-1] != seq_len or self._causal_cache.device != device:
            pos = torch.arange(seq_len, device=device)
            self._causal_cache = pos.view(1, 1, seq_len) <= pos.view(1, seq_len, 1)
            self._eye_cache    = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)

        causal = self._causal_cache
        eye    = self._eye_cache

        same_event = event_ids[:, :, None] == event_ids[:, None, :]
        eos_keys   = ((input_ids == eos_token_id) & valid)[:, None, :]
        q_valid    = valid[:, :, None]
        k_valid    = valid[:, None, :]

        allowed = ((same_event & causal) | (eos_keys & causal)) & q_valid & k_valid
        allowed = allowed | ((~valid)[:, :, None] & eye)

        dtype = next(self.backbone.parameters()).dtype
        mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)
        mask = mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
        return mask

    def get_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_ids: torch.Tensor,
        eos_token_id: int,
        attn_mask_type: str = "event_eot",
    ) -> torch.Tensor:
        if attn_mask_type == "causal":
            eff_mask = attention_mask
        else:
            eff_mask = self._build_event_eot_mask(input_ids, attention_mask, event_ids, eos_token_id)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=eff_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]                      # (B, L, H)

        if self.pooling == "mean_eot":
            # Average hidden states of all valid EOT tokens.
            eot_mask = (input_ids == eos_token_id) & attention_mask.bool()  # (B, L)
            eot_count = eot_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            rep = (hidden * eot_mask.unsqueeze(-1)).sum(dim=1) / eot_count  # (B, H)
            # print(f"[DEBUG] eot_mask hits per sample: {eot_mask.sum(dim=1).tolist()[:4]}")
            # print(f"[DEBUG] rep std: {rep.float().std().item():.6f}")
        else:
            # last_token: hidden state of the last non-padding token.
            seq_lens = attention_mask.sum(dim=1) - 1
            rep = hidden[range(hidden.size(0)), seq_lens]       # (B, H)
        return rep

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        eos_token_id: int,
        attn_mask_type: str = "event_eot",
    ) -> dict:
        rep = self.get_representation(input_ids, attention_mask, event_ids,
                                      eos_token_id, attn_mask_type)
        logits = self.classifier(rep)                           # (B, 2)

        loss = None
        if labels is not None:
            # No class weights — matches TextCancEHR2 setup
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    auroc  = roc_auc_score(labels, probs)
    auprc  = average_precision_score(labels, probs)
    preds  = (probs >= 0.5).astype(int)
    f1     = f1_score(labels, preds, zero_division=0)
    pos_rate = labels.mean()
    return {"auroc": auroc, "auprc": auprc, "f1": f1,
            "prevalence": pos_rate, "n": len(labels),
            "n_pos": int(labels.sum()), "n_neg": int((1 - labels).sum())}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: MIMICClassifier, loader: DataLoader, device: torch.device,
             eos_token_id: int, attn_mask_type: str,
             autocast_dtype: Optional[torch.dtype]) -> Tuple[float, dict, list, list]:
    model.eval()
    all_labels, all_probs, all_sids = [], [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        labels   = batch["labels"].to(device, non_blocking=True)
        sids     = batch["subject_ids"]
        inp      = {k: batch[k].to(device, non_blocking=True) for k in ("input_ids", "attention_mask", "event_ids")}
        autocast_on = torch.cuda.is_available() and autocast_dtype is not None
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_on):
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                event_ids=inp["event_ids"],
                labels=labels,
                eos_token_id=eos_token_id,
                attn_mask_type=attn_mask_type,
            )
        total_loss += float(out["loss"].item())
        n_batches  += 1
        probs = torch.softmax(out["logits"].float(), dim=-1)[:, 1].cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())
        all_sids.extend(sids)

    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    metrics  = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics, all_labels, all_probs


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    p.add_argument("--pretrained_dir",   required=True,
                   help="CPT checkpoint directory saved by train_ehr_event_eot_cpt.py")
    p.add_argument("--eval_parquet_dir", required=True,
                   help="Root dir from build_mimic_eval_parquet.py (contains {task}/{split}.parquet)")
    p.add_argument("--task",             required=True,
                   help="Task name, e.g. icu_mortality or hospital_readmission_30d")
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--tokenizer",       default="Qwen/Qwen3-0.6B")
    p.add_argument("--train_split",      default="train")
    p.add_argument("--val_split",        default="test",
                   help="Used for early stopping (we only have train/test from extract_task_labels)")
    p.add_argument("--test_split",       default=None,
                   help="Optional held-out split; if None, val_split is used for final eval")

    # Model
    p.add_argument("--freeze_backbone",  action="store_true", default=True,
                   help="Freeze backbone, train only the linear head")
    p.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--attn_mask_type",       default="event_eot", choices=["event_eot", "causal"])
    p.add_argument("--pooling",              default="last_token", choices=["last_token", "mean_eot"],
                   help="last_token: last valid token hidden state; "
                        "mean_eot: average of all EOT token hidden states (recommended with event_eot mask).")
    p.add_argument("--attn_implementation",  default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
                   help="eager/sdpa work with event_eot mask; flash_attention_2 only for causal.")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--compile",          action="store_true",
                   help="torch.compile the model for faster iteration (applied after DDP).")
    p.add_argument("--compile_mode",     default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])

    # Training
    p.add_argument("--epochs",           type=int,   default=10)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--eval_batch_size",  type=int,   default=16)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--warmup_ratio",     type=float, default=0.1)
    p.add_argument("--grad_accum",       type=int,   default=1)
    p.add_argument("--max_grad_norm",    type=float, default=1.0)
    p.add_argument("--bf16",             action="store_true")
    p.add_argument("--fp16",             action="store_true")
    p.add_argument("--num_workers",      type=int,   default=4)

    # Early stopping (same as TextCancEHR2; monitored metric: AUROC)
    p.add_argument("--early_stopping_patience", type=int, default=5,
                   help="Stop if val AUROC does not improve for this many epochs. 0 = disabled.")
    p.add_argument("--eval_every_steps", type=int,   default=0,
                   help="Evaluate every N steps within an epoch (0 = only at epoch end)")

    # Logging
    p.add_argument("--log_steps",        type=int,   default=20)
    p.add_argument("--wandb_project",    default=None)
    p.add_argument("--wandb_run_name",   default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── distributed setup ──
    rank, local_rank, world_size = maybe_init_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # Route SDPA away from flash kernel when using event_eot 4D mask.
    if args.attn_implementation == "sdpa" and args.attn_mask_type == "event_eot":
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # ── tokenizer ──
    if is_main_process():
        logger.info("Loading tokenizer from %s", args.pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=args.local_files_only)
    eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id.")
    if is_main_process():
        logger.info("EOT token: %r (id=%d)", tokenizer.pad_token, eos_token_id)

    # ── datasets ──
    eval_dir = Path(args.eval_parquet_dir) / args.task
    train_ds = MIMICEvalDataset(str(eval_dir / f"{args.train_split}.parquet"))
    val_ds   = MIMICEvalDataset(str(eval_dir / f"{args.val_split}.parquet"))
    test_ds  = MIMICEvalDataset(str(eval_dir / f"{args.test_split}.parquet")) \
               if args.test_split else None

    _collate = lambda b: collate_fn(b, eos_token_id)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=True) if world_size > 1 else None
    _pin = torch.cuda.is_available()
    _pw  = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=_collate,
        num_workers=args.num_workers,
        pin_memory=_pin,
        persistent_workers=_pw,
        drop_last=True,
    )
    # Val/test: all ranks evaluate on the full set independently (no sampler).
    val_loader  = DataLoader(val_ds,  batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=_collate, num_workers=args.num_workers,
                             pin_memory=_pin, persistent_workers=_pw)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=_collate, num_workers=args.num_workers,
                             pin_memory=_pin, persistent_workers=_pw) \
                  if test_ds else None

    # ── backbone ──
    if is_main_process():
        logger.info("Loading backbone from %s  (attn_impl=%s)", args.pretrained_dir, args.attn_implementation)
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.pretrained_dir,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    hidden_size = backbone.config.hidden_size

    model = MIMICClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        num_labels=2,
        freeze_backbone=args.freeze_backbone,
        pooling=args.pooling,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.freeze_backbone)

    if args.compile:
        if is_main_process():
            logger.info("torch.compile(mode=%s) …", args.compile_mode)
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model, mode=args.compile_mode)

    # ── optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // max(args.grad_accum, 1))
    total_steps  = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if is_main_process():
        logger.info("steps_per_epoch=%d  total_steps=%d  warmup=%d", steps_per_epoch, total_steps, warmup_steps)

    autocast_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    # ── wandb ──
    wandb_run = None
    if args.wandb_project and is_main_process():
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.task}-{args.attn_mask_type}",
            config=vars(args),
        )

    # ── training loop ──
    best_auroc  = -1.0
    best_epoch  = -1
    patience_ct = 0
    global_step = 0
    results_log = []

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        run_loss, run_n = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    dynamic_ncols=True, disable=not is_main_process())
        for batch_idx, batch in enumerate(pbar):
            inp    = {k: batch[k].to(device, non_blocking=True) for k in ("input_ids", "attention_mask", "event_ids")}
            labels = batch["labels"].to(device, non_blocking=True)

            autocast_on  = torch.cuda.is_available() and autocast_dtype is not None
            is_last_accum = (batch_idx + 1) % args.grad_accum == 0
            sync_ctx = contextlib.nullcontext() if is_last_accum else \
                       model.no_sync() if isinstance(model, DDP) else contextlib.nullcontext()

            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_on):
                    out  = model(labels=labels, eos_token_id=eos_token_id,
                                 attn_mask_type=args.attn_mask_type, **inp)
                    loss = out["loss"] / args.grad_accum
                loss.backward()

            run_loss += float(out["loss"].item())
            run_n    += 1

            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0 and is_main_process():
                    avg = run_loss / max(run_n, 1)
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    if wandb_run:
                        wandb_run.log({"train/loss": avg,
                                       "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                    run_loss, run_n = 0.0, 0

                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    val_loss, val_metrics, _, _ = evaluate(
                        model, val_loader, device, eos_token_id, args.attn_mask_type, autocast_dtype)
                    if is_main_process():
                        logger.info("  Step %d | val_loss=%.4f auroc=%.4f auprc=%.4f f1=%.4f",
                                    global_step, val_loss,
                                    val_metrics["auroc"], val_metrics["auprc"], val_metrics["f1"])
                    model.train()

        # ── epoch-end evaluation (all ranks, same full val set → same metrics) ──
        val_loss, val_metrics, val_labels, val_probs = evaluate(
            model, val_loader, device, eos_token_id, args.attn_mask_type, autocast_dtype)

        if is_main_process():
            logger.info(
                "Epoch %d/%d | val_loss=%.4f  auroc=%.4f  auprc=%.4f  f1=%.4f",
                epoch + 1, args.epochs, val_loss,
                val_metrics["auroc"], val_metrics["auprc"], val_metrics["f1"],
            )
            if wandb_run:
                wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
                wandb_run.log({"val/loss": val_loss}, step=global_step)

            results_log.append({"epoch": epoch + 1, "val_loss": val_loss,
                                 **{f"val_{k}": v for k, v in val_metrics.items()}})

            if val_metrics["auroc"] > best_auroc:
                best_auroc  = val_metrics["auroc"]
                best_epoch  = epoch + 1
                patience_ct = 0
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save({
                    "epoch":       epoch + 1,
                    "model_state": unwrap_model(model).state_dict(),
                    "val_metrics": val_metrics,
                }, best_dir / "classifier.pt")
                logger.info("  >> New best AUROC=%.4f at epoch %d — saved to %s",
                            best_auroc, best_epoch, best_dir)
            else:
                patience_ct += 1
                logger.info("  No improvement. Patience: %d/%d",
                            patience_ct, args.early_stopping_patience)

        # Broadcast early-stopping decision from rank 0 to all ranks.
        should_stop = torch.tensor(
            int(args.early_stopping_patience > 0 and patience_ct >= args.early_stopping_patience),
            device=device,
        )
        if dist.is_initialized():
            dist.broadcast(should_stop, src=0)
        if should_stop.item():
            if is_main_process():
                logger.info("Early stopping triggered at epoch %d (best=%d, AUROC=%.4f)",
                            epoch + 1, best_epoch, best_auroc)
            break

    # ── load best model for final evaluation (rank 0 only) ──
    if dist.is_initialized():
        dist.barrier()
    if is_main_process():
        logger.info("Loading best checkpoint (epoch=%d, AUROC=%.4f)…", best_epoch, best_auroc)
        ckpt = torch.load(output_dir / "best" / "classifier.pt", map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model_state"])

        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            if loader is None:
                continue
            _, metrics, labels_list, probs_list = evaluate(
                model, loader, device, eos_token_id, args.attn_mask_type, autocast_dtype)
            logger.info("FINAL [%s]  auroc=%.4f  auprc=%.4f  f1=%.4f  n=%d  pos=%.1f%%",
                        split_name, metrics["auroc"], metrics["auprc"], metrics["f1"],
                        metrics["n"], 100 * metrics["prevalence"])
            if wandb_run:
                wandb_run.log({f"final_{split_name}/{k}": v for k, v in metrics.items()})
            result_payload = {
                "task": args.task, "split": split_name, "best_epoch": best_epoch,
                "metrics": metrics, "labels": labels_list, "probs": probs_list,
            }
            (output_dir / f"{split_name}_results.json").write_text(
                json.dumps(result_payload, indent=2))
            logger.info("Predictions saved -> %s", output_dir / f"{split_name}_results.json")

        (output_dir / "training_log.json").write_text(json.dumps(results_log, indent=2))
        logger.info("Training log -> %s", output_dir / "training_log.json")
        if wandb_run:
            wandb_run.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
