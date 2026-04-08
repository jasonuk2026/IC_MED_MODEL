#!/usr/bin/env python3
"""
train_embedding_disease_cond_v2.py

Disease-conditioned EHR embedding model.

Train data: output of prepare_task_data.py
    schema: task_idx (int16), label (int8), event_ids (list<int32>), source_row (int32)
    Rows are pre-shuffled with guaranteed pos/neg ratio — no runtime sampling needed.
    Sequential reads: each batch = batch_size consecutive rows from the shard.

Eval data: output of dataset/build_eval_task_data.py
    schema: patient_id (int64), task_idx (int16), label (int8), event_ids (list<int32>)
    One row per labeled patient.  Triplet evaluation with patient-level constraints:
      anchor and positive come from different patients,
      anchor and negative come from different patients.

Two-stage training
──────────────────
  Stage 1 (--stage1_steps N):
    • Only bert_proj_1, bert_proj_2, input_norm are trained.
    • LoRA params are frozen (requires_grad=False).
    • Forward still runs through the frozen Qwen backbone, matching stage 2.
    • This keeps the representation path consistent while warming up the
      projection layers before LoRA is unfrozen.

  Stage 2 (remaining epochs):
    • LoRA params are unfrozen.
    • Full forward: Qwen processes events; mean-pool over event hidden states.
    • Separate LRs: --lr_proj (higher) for projection layers,
                    --lr_lora (lower)  for LoRA.

Loss: BatchHardTripletLoss (cosine distance) throughout both stages.

Usage (single GPU)
──────────────────
  python train_embedding_disease_cond_v2.py \\
      --train_data_dir data/llm_data_v7 \\
      --eval_data_paths EHRSHOT_ASSETS/llm_eval_data/new_*/val.parquet \\
      --bert_embeddings data/biolinkbert_embeddings/embeddings.npy \\
      --bf16 --flash_attn \\
      --stage1_steps 1000 --lr_proj 2e-4 --lr_lora 5e-5

NOTE: QLoRA (--qlora) is incompatible with multi-GPU DDP.
"""

import json
import os
import math
import random
import logging
import argparse
from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, logging as hf_logging
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sentence_transformers.losses import (
    BatchHardSoftMarginTripletLoss,
    BatchHardTripletLoss,
    BatchHardTripletLossDistanceFunction,
)
from model import DiseaseAwareEHREncoder

hf_logging.set_verbosity_warning()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── Constants ──────────────────────────────────────────────────────────────────

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}

# Stable integer index for each task — used as input to the model instead of
# disease embeddings.  Sorted alphabetically so the mapping is reproducible.
TASK_2_IDX: dict[str, int] = {t: i for i, t in enumerate(sorted(TASK_2_DISEASE_NAME))}

BERT_DIM    = 768     # BioLinkBERT-base hidden size
EOS_TOKEN_ID = 151643  # Qwen3 <|endoftext|> — pooling anchor token

# Prompt template split around the disease virtual token and event virtual tokens:
#   "Please predict disease <DISEASE> based on the following events.\nStart of medical events: <EVENTS>"
PROMPT_PREFIX = "Please predict disease "
PROMPT_MIDDLE = " based on the following events.\nStart of medical events:"


# ── BioLinkBERT embeddings (read-only mmap) ───────────────────────────────────

class EmbeddingStore:
    """Memory-mapped BioLinkBERT event embeddings array."""

    def __init__(self, embeddings_path: str):
        logger.info(f"Loading embeddings (mmap) from {embeddings_path} …")
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        logger.info(f"  Embeddings shape: {self.embeddings.shape}  dtype: {self.embeddings.dtype}")


# ── Shared collation helper ───────────────────────────────────────────────────

def _collate_event_embs(
    eids_list:         list[np.ndarray],
    task_idxs:         list[int],
    labels:            list[int] | None,
    embeddings:        np.ndarray,
    pad_to_num_events: int | None,
) -> dict[str, torch.Tensor]:
    """Collate a list of event-id arrays into padded tensors ready for the model."""
    bert_dim   = embeddings.shape[1]
    embs_list: list[np.ndarray] = []
    for eids in eids_list:
        if pad_to_num_events is not None:
            eids = eids[:pad_to_num_events]
        if len(eids) == 0:
            embs_list.append(np.zeros((1, bert_dim), dtype=np.float32))
        else:
            embs_list.append(embeddings[eids].astype(np.float32))

    B     = len(embs_list)
    max_e = pad_to_num_events if pad_to_num_events is not None \
            else max(e.shape[0] for e in embs_list)
    padded = np.zeros((B, max_e, bert_dim), dtype=np.float32)
    mask   = np.zeros((B, max_e),           dtype=np.int64)
    for i, embs in enumerate(embs_list):
        n = embs.shape[0]
        padded[i, :n] = embs
        mask[i, :n]   = 1

    out: dict[str, torch.Tensor] = {
        "event_embs": torch.from_numpy(padded),
        "event_mask": torch.from_numpy(mask),
        "task_idxs":  torch.tensor(task_idxs, dtype=torch.long),
    }
    if labels is not None:
        out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out


# ── Per-epoch lazy-loading Dataset ────────────────────────────────────────────

import bisect

# Worker-local state: ParquetFile handles + cumulative row-group offsets.
_worker_pq_files:    dict[int, pq.ParquetFile] = {}
_worker_pq_cum_rows: dict[int, list[int]]       = {}   # task_idx → [0, rg0, rg0+rg1, ...]


def _build_cum_rows(pf: pq.ParquetFile) -> list[int]:
    meta = pf.metadata
    cum  = [0]
    for i in range(meta.num_row_groups):
        cum.append(cum[-1] + meta.row_group(i).num_rows)
    return cum


def _epoch_worker_init(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    ds: EpochBatchDataset = worker_info.dataset
    global _worker_pq_files, _worker_pq_cum_rows
    _worker_pq_files    = {}
    _worker_pq_cum_rows = {}
    for task_idx, path in ds.task_parquet_paths.items():
        pf = pq.ParquetFile(str(path))
        _worker_pq_files[task_idx]    = pf
        _worker_pq_cum_rows[task_idx] = _build_cum_rows(pf)


def _read_rows(
    pf:       pq.ParquetFile,
    cum_rows: list[int],
    start:    int,
    length:   int,
    columns:  list[str],
) -> "pd.DataFrame":
    end      = start + length - 1
    rg_first = bisect.bisect_right(cum_rows, start) - 1
    rg_last  = bisect.bisect_right(cum_rows, end)   - 1

    if rg_first == rg_last:
        df          = pf.read_row_group(rg_first, columns=columns).to_pandas()
        local_start = start - cum_rows[rg_first]
        return df.iloc[local_start : local_start + length]

    parts = [pf.read_row_group(rg, columns=columns).to_pandas()
             for rg in range(rg_first, rg_last + 1)]
    df          = pd.concat(parts, ignore_index=True)
    local_start = start - cum_rows[rg_first]
    return df.iloc[local_start : local_start + length]


class EpochBatchDataset(Dataset):
    """Schedule-based lazy-loading dataset for one training epoch."""

    def __init__(
        self,
        data_dir:          str,
        data_epoch_idx:    int,
        tasks:             list[str],
        store:             EmbeddingStore,
        batch_size:        int,
        training_epoch:    int,
        seed:              int,
        world_size:        int,
        rank:              int,
        pad_to_num_events: int | None = None,
    ):
        self.store               = store
        self.batch_size          = batch_size
        self.pad_to_num_events   = pad_to_num_events
        self.task_parquet_paths: dict[int, Path] = {}
        self._main_pq_files:    dict[int, pq.ParquetFile] = {}
        self._main_pq_cum_rows: dict[int, list[int]]       = {}

        all_batches: list[tuple[int, int]] = []
        max_batch_size: int | None = None

        for task in sorted(tasks):
            task_idx  = TASK_2_IDX[task]
            p_parquet = Path(data_dir) / task / f"train_prepared_{data_epoch_idx:03d}.parquet"
            p_json    = Path(data_dir) / task / f"train_prepared_{data_epoch_idx:03d}.json"

            if not p_parquet.exists():
                logger.warning(f"  [{task}] {p_parquet} not found — skipping")
                continue

            with open(p_json) as f:
                meta = json.load(f)

            mbs = meta["max_batch_size"]
            if max_batch_size is None:
                max_batch_size = mbs
            elif max_batch_size != mbs:
                raise ValueError(
                    f"Inconsistent max_batch_size: {max_batch_size} vs {mbs} for {task}"
                )

            self.task_parquet_paths[task_idx] = p_parquet

            sub_per_max = mbs // batch_size
            n_max       = meta["num_batches"]
            for max_i in range(n_max):
                for sub_i in range(sub_per_max):
                    all_batches.append((task_idx, max_i * mbs + sub_i * batch_size))

            logger.info(f"  [{task}] epoch={data_epoch_idx}  "
                        f"{n_max} max-batches × {sub_per_max} sub = "
                        f"{n_max * sub_per_max} training batches")

        if not all_batches:
            raise ValueError("No valid prepared data found for any task.")

        rng = random.Random(seed + training_epoch * 1337)
        rng.shuffle(all_batches)
        n_total     = (len(all_batches) // world_size) * world_size
        all_batches = all_batches[:n_total]
        self.schedule: list[tuple[int, int]] = all_batches[rank::world_size]

    def __len__(self) -> int:
        return len(self.schedule)

    def _get_pq_state(self, task_idx: int) -> tuple[pq.ParquetFile, list[int]]:
        global _worker_pq_files, _worker_pq_cum_rows
        if _worker_pq_files:
            return _worker_pq_files[task_idx], _worker_pq_cum_rows[task_idx]
        if task_idx not in self._main_pq_files:
            pf = pq.ParquetFile(str(self.task_parquet_paths[task_idx]))
            self._main_pq_files[task_idx]    = pf
            self._main_pq_cum_rows[task_idx] = _build_cum_rows(pf)
        return self._main_pq_files[task_idx], self._main_pq_cum_rows[task_idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        task_idx, start_row = self.schedule[idx]
        pf, cum_rows = self._get_pq_state(task_idx)

        sub = _read_rows(pf, cum_rows, start_row, self.batch_size,
                         columns=["label", "event_ids"])

        eids_list = [np.array(e, dtype=np.int32) for e in sub["event_ids"]]
        labels    = sub["label"].tolist()

        return _collate_event_embs(
            eids_list, [task_idx] * len(eids_list), labels,
            self.store.embeddings, self.pad_to_num_events,
        )


# ── Eval triplet pre-computation ──────────────────────────────────────────────

EVAL_TRIPLET_SCHEMA = pa.schema([
    pa.field("task_idx",      pa.int16()),
    pa.field("anchor_eids",   pa.list_(pa.int32())),
    pa.field("positive_eids", pa.list_(pa.int32())),
    pa.field("negative_eids", pa.list_(pa.int32())),
])


def precompute_eval_triplets(
    eval_data_paths:      list[str],
    n_triplets_per_task:  int,
    seed:                 int,
    output_path:          Path,
) -> int:
    pos: dict[int, list[list[int]]] = defaultdict(list)
    neg: dict[int, list[list[int]]] = defaultdict(list)

    for path in eval_data_paths:
        df = pd.read_parquet(path, columns=["task_idx", "label", "event_ids"])
        for row in df.itertuples(index=False):
            task_idx = int(row.task_idx)
            eids     = list(row.event_ids)
            (pos if int(row.label) == 1 else neg)[task_idx].append(eids)
        del df

    rng = random.Random(seed)
    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    records: list[dict] = []

    for task_idx in sorted(set(list(pos.keys()) + list(neg.keys()))):
        pos_list = pos.get(task_idx, [])
        neg_list = neg.get(task_idx, [])

        if not pos_list or not neg_list:
            logger.warning(f"  [{idx_to_name.get(task_idx, task_idx)}] "
                           f"insufficient eval data for triplets — skipping")
            continue

        task_count = 0
        for _ in range(n_triplets_per_task):
            anchor_eids   = rng.choice(pos_list)
            positive_eids = rng.choice(pos_list)
            negative_eids = rng.choice(neg_list)
            records.append({
                "task_idx":      task_idx,
                "anchor_eids":   anchor_eids,
                "positive_eids": positive_eids,
                "negative_eids": negative_eids,
            })
            task_count += 1

        logger.info(f"  [{idx_to_name.get(task_idx, task_idx)}] {task_count} eval triplets")

    rng.shuffle(records)

    table = pa.table(
        {
            "task_idx":      pa.array([r["task_idx"]      for r in records], type=pa.int16()),
            "anchor_eids":   pa.array([r["anchor_eids"]   for r in records], type=pa.list_(pa.int32())),
            "positive_eids": pa.array([r["positive_eids"] for r in records], type=pa.list_(pa.int32())),
            "negative_eids": pa.array([r["negative_eids"] for r in records], type=pa.list_(pa.int32())),
        },
        schema=EVAL_TRIPLET_SCHEMA,
    )
    pq.write_table(table, str(output_path))
    logger.info(f"  Saved {len(records)} eval triplets → {output_path}")
    return len(records)


# ── Optimisation helpers ───────────────────────────────────────────────────────

def _set_lora_trainable(model: torch.nn.Module, trainable: bool) -> int:
    """Set requires_grad on all LoRA parameters. Returns the count of affected params."""
    n = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(trainable)
            n += 1
    return n


def _make_param_groups(
    encoder:  torch.nn.Module,
    lr_proj:  float,
    lr_lora:  float,
) -> list[dict]:
    """Return AdamW param-group list with separate LRs for projection vs LoRA layers."""
    raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
    proj_ids = {
        id(p)
        for p in [
            *raw_enc.input_norm.parameters(),
            *raw_enc.bert_proj_1.parameters(),
            *raw_enc.bert_proj_2.parameters(),
        ]
    }
    proj_params, lora_params = [], []
    for p in encoder.parameters():
        if not p.requires_grad:
            continue
        if id(p) in proj_ids:
            proj_params.append(p)
        else:
            lora_params.append(p)
    groups: list[dict] = [{"params": proj_params, "lr": lr_proj}]
    if lora_params:
        groups.append({"params": lora_params, "lr": lr_lora})
    return groups


# ── Per-batch diagnostic metrics ──────────────────────────────────────────────

def _compute_batch_metrics(
    emb:    torch.Tensor,   # (B, D) L2-normalised float32
    labels: torch.Tensor,   # (B,)   long, same device
) -> dict[str, float]:
    """Compute embedding diagnostics consistent with cosine-distance triplet loss."""
    device = emb.device
    e      = emb.float()
    lbl    = labels
    e_pos  = e[lbl == 1]
    e_neg  = e[lbl == 0]
    B      = e.size(0)

    # Mean pairwise cosine distance over all (i<j) pairs
    sim_mat  = e @ e.T                                          # (B, B)
    tri_mask = torch.triu(
        torch.ones(B, B, dtype=torch.bool, device=device), diagonal=1
    )
    cdist = (1.0 - sim_mat[tri_mask]).mean().item()

    # Positive-positive cosine distance
    d_pp = float("nan")
    if e_pos.size(0) >= 2:
        sim_pp = e_pos @ e_pos.T
        np_    = e_pos.size(0)
        tri_pp = torch.triu(torch.ones(np_, np_, dtype=torch.bool, device=device), diagonal=1)
        d_pp   = (1.0 - sim_pp[tri_pp]).mean().item()

    # Positive-negative cosine distance
    d_pn = float("nan")
    if e_pos.size(0) >= 1 and e_neg.size(0) >= 1:
        d_pn = (1.0 - (e_pos @ e_neg.T)).mean().item()

    # Per-dimension variance (mean over dims = trace(Cov) / D)
    var_all = e.var(dim=0).mean().item()
    var_pos = e_pos.var(dim=0).mean().item() if e_pos.size(0) >= 2 else float("nan")
    var_neg = e_neg.var(dim=0).mean().item() if e_neg.size(0) >= 2 else float("nan")

    return {
        "cdist":   cdist,
        "d_pp":    d_pp,
        "d_pn":    d_pn,
        "var_all": var_all,
        "var_pos": var_pos,
        "var_neg": var_neg,
    }


def _supervised_contrastive_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss over a batch with binary labels.

    The loss uses all same-label examples as positives for each anchor and all
    opposite-label examples as negatives. Features are normalised inside the
    loss to focus learning on direction while keeping stage-1 diagnostics on the
    raw pre-normalised representation available separately.
    """
    feat = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
    labels = labels.view(-1)

    logits = (feat @ feat.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~eye
    valid = pos_mask.any(dim=1)
    if not valid.any():
        return feat.new_zeros(())

    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    pos_counts = pos_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_counts
    return -mean_log_prob_pos[valid].mean()


# ── Model setup ───────────────────────────────────────────────────────────────

def load_qwen(args):
    kwargs: dict = {}
    if args.flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if args.qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        args.bf16 = True
    elif args.bf16:
        kwargs["dtype"] = torch.bfloat16
    elif args.fp16:
        kwargs["dtype"] = torch.float16

    model     = AutoModel.from_pretrained(args.model_name, local_files_only=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Pad token undetected, set to eos token")

    return model, tokenizer


def setup_lora(model, args):
    model.config.use_cache = False

    if args.qlora:
        prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    ))

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    q_total     = sum(p.numel() for p in model.parameters())
    q_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Qwen trainable params: {q_trainable:,} / {q_total:,} "
                f"({100 * q_trainable / q_total:.2f}%)")
    return model


def build_task_prefix_ids(tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    tasks_sorted  = sorted(TASK_2_DISEASE_NAME)
    pad_id        = tokenizer.pad_token_id

    all_ids: list[list[int]] = []
    for task in tasks_sorted:
        prefix_str = PROMPT_PREFIX + TASK_2_DISEASE_NAME[task]
        ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
        all_ids.append(ids)
        logger.info(f"  Task prefix [{task}]: {repr(prefix_str)}  ({len(ids)} tokens)")

    max_len = max(len(ids) for ids in all_ids)
    task_prefix_ids  = torch.full((len(tasks_sorted), max_len), pad_id, dtype=torch.long)
    task_prefix_mask = torch.zeros((len(tasks_sorted), max_len), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        n = len(ids)
        task_prefix_ids[i,  max_len - n:] = torch.tensor(ids, dtype=torch.long)
        task_prefix_mask[i, max_len - n:] = 1

    logger.info(f"  Padded prefix length: {max_len} tokens (left-padded with pad_token_id={pad_id})")
    return task_prefix_ids, task_prefix_mask


def build_encoder(qwen_model, tokenizer, args) -> DiseaseAwareEHREncoder:
    qwen_dim = qwen_model.config.hidden_size

    task_prefix_ids, task_prefix_mask = build_task_prefix_ids(tokenizer)
    middle_ids = tokenizer(
        PROMPT_MIDDLE, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    dtype = (
        torch.bfloat16 if args.bf16 else
        torch.float16  if args.fp16 else
        torch.float32
    )
    encoder = DiseaseAwareEHREncoder(
        qwen_model       = qwen_model,
        bert_dim         = BERT_DIM,
        qwen_dim         = qwen_dim,
        task_prefix_ids  = task_prefix_ids,
        task_prefix_mask = task_prefix_mask,
        middle_ids       = middle_ids,
        dtype            = dtype,
    )
    logger.info(
        f"  Prompt template: [{repr(PROMPT_PREFIX)}<disease_name>]{repr(PROMPT_MIDDLE)}<events>"
    )
    logger.info(f"  Projection MLP: Linear({BERT_DIM}→{BERT_DIM}) → GELU → Linear({BERT_DIM}→{qwen_dim})  dtype={dtype}")

    total     = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"  Total encoder trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)")
    return encoder


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvalBatchDataset(Dataset):
    def __init__(
        self,
        entries:           list[tuple[np.ndarray, int]],
        store:             EmbeddingStore,
        batch_size:        int,
        pad_to_num_events: int | None = None,
    ):
        self.store             = store
        self.pad_to_num_events = pad_to_num_events
        self._batches = [
            entries[i : i + batch_size]
            for i in range(0, len(entries), batch_size)
        ]

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, j: int) -> dict[str, torch.Tensor]:
        batch      = self._batches[j]
        eids_list  = [e[0] for e in batch]
        task_idxs  = [e[1] for e in batch]
        return _collate_event_embs(
            eids_list, task_idxs, None,
            self.store.embeddings, self.pad_to_num_events,
        )


@torch.inference_mode()
def evaluate_rank0(
    encoder:           DiseaseAwareEHREncoder,
    eval_triplet_path: Path,
    store:             EmbeddingStore,
    device:            torch.device,
    args,
) -> tuple[float, dict[str, float]]:
    raw_encoder = encoder.module if isinstance(encoder, DDP) else encoder
    raw_encoder.eval()

    df = pd.read_parquet(
        str(eval_triplet_path),
        columns=["task_idx", "anchor_eids", "positive_eids", "negative_eids"],
    )
    n_triplets = len(df)

    if n_triplets == 0:
        logger.warning("eval_triplets.parquet is empty — skipping eval.")
        raw_encoder.train()
        return 0.0, {}

    entries: list[tuple[np.ndarray, int]] = []
    for row in df.itertuples(index=False):
        t = int(row.task_idx)
        entries.append((np.array(row.anchor_eids,   dtype=np.int32), t))
        entries.append((np.array(row.positive_eids, dtype=np.int32), t))
        entries.append((np.array(row.negative_eids, dtype=np.int32), t))

    eval_ds = EvalBatchDataset(entries, store, args.eval_batch_size, args.pad_to_num_events)
    eval_dl = DataLoader(
        eval_ds, batch_size=1, shuffle=False,
        collate_fn=lambda b: b[0], num_workers=0,
    )

    all_embs: list[torch.Tensor] = []
    for batch in tqdm(eval_dl, desc="Evaluating", dynamic_ncols=True):
        all_embs.append(
            raw_encoder(
                batch["event_embs"].to(device),
                batch["event_mask"].to(device),
                batch["task_idxs"].to(device),
            ).cpu()
        )

    embs      = torch.cat(all_embs, dim=0)
    anchors   = embs[0::3]
    positives = embs[1::3]
    negatives = embs[2::3]

    d_ap    = (anchors - positives).norm(dim=1)
    d_an    = (anchors - negatives).norm(dim=1)
    correct = d_ap < d_an

    task_idxs_arr = df["task_idx"].values
    idx_to_name   = {v: k for k, v in TASK_2_IDX.items()}
    task_acc: dict[str, float] = {}

    for t_idx in sorted(set(task_idxs_arr.tolist())):
        mask = task_idxs_arr == t_idx
        n_t  = int(mask.sum())
        if n_t > 0:
            acc = correct[torch.from_numpy(mask)].float().mean().item()
            task_acc[idx_to_name.get(t_idx, str(t_idx))] = acc

    overall_acc = correct.float().mean().item()
    raw_encoder.train()
    return overall_acc, task_acc


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Disease-aware EHR embedding (v2, mean-pool over event hidden states).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--eval_only",  action="store_true")
    p.add_argument("--debug_batches", type=int, default=None)
    p.add_argument("--checkpoint", default=None)

    # Data
    p.add_argument("--train_data_dir", default=None)
    p.add_argument("--tasks", nargs="+", default=list(sorted(TASK_2_DISEASE_NAME.keys())))
    p.add_argument("--eval_data_paths",  nargs="+", default=None)

    # BioLinkBERT embeddings
    p.add_argument("--bert_embeddings", required=True)

    # Qwen model
    p.add_argument("--model_name",   default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--flash_attn",   action="store_true")
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--fp16",         action="store_true")
    p.add_argument("--qlora",        action="store_true")

    # LoRA
    p.add_argument("--lora_r",              type=int,   default=4)
    p.add_argument("--lora_alpha",          type=int,   default=8)
    p.add_argument("--lora_dropout",        type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Training
    p.add_argument("--output_dir",   default="output/medical-embedding-disease-cond-v2")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--log_steps",    type=int,   default=10)

    # Two-stage training
    p.add_argument("--stage1_steps", type=int, default=0,
                   help="Number of optimizer steps for stage 1 (LoRA frozen, full frozen-Qwen forward, "
                        "only proj params trainable). "
                        "0 = skip stage 1.")
    p.add_argument("--lr_proj", type=float, default=2e-4,
                   help="LR for bert_proj_1, bert_proj_2, input_norm (higher).")
    p.add_argument("--lr_lora", type=float, default=5e-5,
                   help="LR for LoRA parameters (lower).")

    # Loss
    p.add_argument("--triplet_margin", type=float, default=0.3,
                   help="Margin for BatchHardTripletLoss. Set to 0 for soft-margin variant.")
    p.add_argument("--var_reg_weight", type=float, default=0.1,
                   help="Weight for variance regularisation in Stage 1 to prevent embedding "
                        "collapse. Loss += var_reg_weight * relu(1 - std(pre_emb, dim=0)).mean() "
                        "where pre_emb is the mean-pooled projection before L2 normalisation. "
                        "Set to 0 to disable.")
    p.add_argument("--stage1_temperature", type=float, default=0.1,
                   help="Temperature for supervised contrastive loss in Stage 1.")

    # Eval
    p.add_argument("--n_eval_triplets_per_task", type=int,   default=256)
    p.add_argument("--eval_batch_size",          type=int,   default=32)
    p.add_argument("--pad_to_num_events",        type=int,   default=None)

    p.add_argument("--num_workers",      type=int, default=4)
    p.add_argument("--prefetch_factor",  type=int, default=4)

    p.add_argument("--compile", action="store_true")

    # wandb
    p.add_argument("--wandb_project",  default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags",     nargs="+", default=None)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp     = world_size > 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_ddp:
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    if args.qlora and is_ddp:
        raise RuntimeError("QLoRA is incompatible with multi-GPU DDP.")
    if not args.eval_only and args.train_data_dir is None:
        raise ValueError("--train_data_dir is required unless --eval_only is set.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(args.output_dir) / timestamp
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run dir: {run_dir}")
        logger.info(f"World size: {world_size}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = (args.wandb_project is not None) and (rank == 0)
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or timestamp,
            tags=args.wandb_tags,
            config=vars(args),
            dir=str(run_dir),
            settings=wandb.Settings(console="wrap"),
        )
        wandb.save(os.path.abspath(__file__),
                   base_path=os.path.dirname(os.path.abspath(__file__)))
        logger.info(f"wandb run: {wandb_run.url}")

    # ── BioLinkBERT embeddings ────────────────────────────────────────────────
    store = EmbeddingStore(args.bert_embeddings)

    # ── Build Qwen model + tokenizer ─────────────────────────────────────────
    if rank == 0:
        logger.info(f"Loading Qwen: {args.model_name}")
    qwen_model, tokenizer = load_qwen(args)

    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        qwen_model.config.use_cache = False
        qwen_lora = PeftModel.from_pretrained(qwen_model, str(Path(args.checkpoint) / "lora"))
        encoder = build_encoder(qwen_lora, tokenizer, args)
        extra = torch.load(Path(args.checkpoint) / "extra_modules.pt", map_location="cpu")
        encoder.bert_proj_1.load_state_dict(extra["bert_proj_1"])
        encoder.bert_proj_2.load_state_dict(extra["bert_proj_2"])
        if "input_norm" in extra:
            encoder.input_norm.load_state_dict(extra["input_norm"])
    else:
        qwen_lora = setup_lora(qwen_model, args)
        encoder = build_encoder(qwen_lora, tokenizer, args)

    encoder = encoder.to(device)

    if args.compile:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        encoder = torch.compile(encoder)

    if is_ddp and not args.eval_only:
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=False, static_graph=True)

    # ── Pre-compute eval triplets (rank 0 only) ───────────────────────────────
    eval_triplet_path: Path | None = None
    if args.eval_data_paths and rank == 0:
        eval_triplet_path = run_dir / "eval_triplets.parquet"
        logger.info("Pre-computing eval triplets …")
        precompute_eval_triplets(
            args.eval_data_paths,
            args.n_eval_triplets_per_task,
            args.seed,
            eval_triplet_path,
        )

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        if eval_triplet_path is None:
            raise ValueError("--eval_only requires --eval_data_paths.")
        val_acc, val_task_acc = evaluate_rank0(encoder, eval_triplet_path, store, device, args)
        if rank == 0:
            logger.info(f"Eval triplet accuracy: {val_acc:.4f}")
            for task, acc in sorted(val_task_acc.items()):
                logger.info(f"  {task}: {acc:.4f}")
        if use_wandb and wandb_run:
            import wandb
            log_dict = {"eval/triplet_acc": val_acc}
            for task, acc in val_task_acc.items():
                log_dict[f"eval/{task}/triplet_acc"] = acc
            wandb.log(log_dict)
            wandb_run.finish()
        if is_ddp:
            dist.destroy_process_group()
        return

    # ── Loss function (BatchHardTripletLoss) ─────────────────────────────────
    if args.triplet_margin > 0:
        _loss_module = BatchHardTripletLoss(
            model=None,
            distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
            margin=args.triplet_margin,
        )
        _loss_fn  = lambda labels, emb: _loss_module.batch_hard_triplet_loss(labels, emb)
        loss_desc = f"BatchHardTripletLoss (margin={args.triplet_margin}, cosine_distance)"
    else:
        _loss_module = BatchHardSoftMarginTripletLoss(
            model=None,
            distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
        )
        _loss_fn  = lambda labels, emb: _loss_module.batch_hard_triplet_soft_margin_loss(labels, emb)
        loss_desc = "BatchHardSoftMarginTripletLoss (cosine_distance)"
    if rank == 0:
        logger.info(loss_desc)

    # ── Count available data epochs ───────────────────────────────────────────
    first_task = sorted(args.tasks)[0]
    task_dir   = Path(args.train_data_dir) / first_task
    data_epoch_files = sorted(task_dir.glob("train_prepared_*.parquet"))
    n_data_epochs    = len(data_epoch_files)
    if n_data_epochs == 0:
        raise ValueError(f"No train_prepared_*.parquet found in {task_dir}")
    if rank == 0:
        logger.info(f"Found {n_data_epochs} data epoch(s) (task dir: {task_dir})")

    # ── Helper: build DataLoader from a given data epoch ─────────────────────
    def _make_loader(data_epoch: int, training_epoch: int) -> tuple[EpochBatchDataset, DataLoader]:
        ds = EpochBatchDataset(
            args.train_data_dir, data_epoch, args.tasks, store,
            args.batch_size, training_epoch, args.seed, world_size, rank,
            args.pad_to_num_events,
        )
        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda b: b[0],
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_epoch_worker_init,
            persistent_workers=False,
        )
        return ds, dl

    # ── Stage 1: full frozen-Qwen forward, train only projection layers ─────
    s1_step = 0   # tracks steps completed; used as wandb offset for Stage 2
    if args.stage1_steps > 0:
        if rank == 0:
            logger.info(
                f"Stage 1: {args.stage1_steps} steps — "
                f"LoRA frozen, full frozen-Qwen forward, lr_proj={args.lr_proj}"
            )

        n_frozen = _set_lora_trainable(encoder, False)
        if rank == 0:
            logger.info(f"  Froze {n_frozen} LoRA parameter tensors")

        # Separate param groups so RMSNorm gamma is not weight-decayed.
        raw_enc_s1 = encoder.module if isinstance(encoder, DDP) else encoder
        s1_no_wd_ids = {id(p) for p in raw_enc_s1.input_norm.parameters()}
        s1_wd_params    = [p for p in encoder.parameters()
                           if p.requires_grad and id(p) not in s1_no_wd_ids]
        s1_no_wd_params = [p for p in encoder.parameters()
                           if p.requires_grad and id(p) in s1_no_wd_ids]
        s1_param_groups = [{"params": s1_wd_params, "weight_decay": args.weight_decay}]
        if s1_no_wd_params:
            s1_param_groups.append({"params": s1_no_wd_params, "weight_decay": 0.0})

        stage1_opt = torch.optim.AdamW(
            s1_param_groups,
            lr=args.lr_proj,
            fused=True,
        )

        # Linear warmup for Stage 1 (same warmup_ratio as Stage 2).
        s1_warmup = max(1, int(args.stage1_steps * args.warmup_ratio))
        s1_sched  = torch.optim.lr_scheduler.LinearLR(
            stage1_opt, start_factor=1e-3, end_factor=1.0, total_iters=s1_warmup
        )
        if rank == 0:
            logger.info(f"  Stage 1 warmup: {s1_warmup} steps (ratio={args.warmup_ratio})")

        stage1_ds, stage1_dl = _make_loader(data_epoch=0, training_epoch=0)
        encoder.train()
        stage1_opt.zero_grad()
        s1_step = 0

        pbar_s1 = tqdm(
            stage1_dl,
            desc="Stage 1",
            disable=(rank != 0),
            dynamic_ncols=True,
            total=min(args.stage1_steps, len(stage1_ds)),
        )

        for batch in pbar_s1:
            if s1_step >= args.stage1_steps:
                break

            labels_t = batch["labels"]

            enc_out = encoder(
                batch["event_embs"].to(device),
                batch["event_mask"].to(device),
                batch["task_idxs"].to(device),
                return_pre_emb=True,
            )
            emb, pre_emb = enc_out
            supcon_loss = _supervised_contrastive_loss(
                pre_emb,
                labels_t.to(device),
                args.stage1_temperature,
            )

            # Variance regularisation is applied to the pre-normalisation pooled
            # embedding from the same full-Qwen path used in stage 2.
            if args.var_reg_weight > 0.0:
                std      = torch.sqrt(pre_emb.var(dim=0) + 1e-4)   # (D,)
                var_loss = torch.relu(1.0 - std).mean()
                loss     = supcon_loss + args.var_reg_weight * var_loss
            else:
                loss = supcon_loss
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in encoder.parameters() if p.requires_grad],
                args.grad_clip,
            )
            stage1_opt.step()
            if s1_step < s1_warmup:
                s1_sched.step()
            stage1_opt.zero_grad()
            s1_step += 1

            if rank == 0:
                loss_now  = loss.item()
                gnorm_now = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                lr_now    = stage1_opt.param_groups[0]["lr"]

                with torch.no_grad():
                    metrics = _compute_batch_metrics(emb.detach(), labels_t.to(device))
                    pre_var_now = (
                        pre_emb.detach().float().var(dim=0).mean().item()
                    )
                    loss_supcon_now = supcon_loss.item()
                    var_loss_now = var_loss.item() if args.var_reg_weight > 0.0 else 0.0

                pbar_s1.set_postfix(
                    loss=f"{loss_now:.4f}",
                    lsup=f"{loss_supcon_now:.4f}",
                    cdist=f"{metrics['cdist']:.3f}",
                    dpp=f"{metrics['d_pp']:.3f}",
                    dpn=f"{metrics['d_pn']:.3f}",
                    var=f"{metrics['var_all']:.4f}",
                    prevar=f"{pre_var_now:.4f}",
                    gnorm=f"{gnorm_now:.3f}",
                    lr=f"{lr_now:.2e}",
                )

                if use_wandb and s1_step % args.log_steps == 0:
                    import wandb
                    wandb.log({
                        "stage1/loss":         loss_now,
                        "stage1/loss_supcon":  loss_supcon_now,
                        "stage1/loss_var":     var_loss_now,
                        "stage1/gnorm":        gnorm_now,
                        "stage1/lr":           lr_now,
                        "stage1/cdist":        metrics["cdist"],
                        "stage1/dist_pos_pos": metrics["d_pp"],
                        "stage1/dist_pos_neg": metrics["d_pn"],
                        "stage1/var_all":      metrics["var_all"],
                        "stage1/pre_var_all":  pre_var_now,
                        "stage1/var_pos":      metrics["var_pos"],
                        "stage1/var_neg":      metrics["var_neg"],
                    }, step=s1_step)

        pbar_s1.close()
        if rank == 0:
            logger.info(f"Stage 1 done ({s1_step} steps). Unfreezing LoRA for stage 2 …")

        n_unfrozen = _set_lora_trainable(encoder, True)
        if rank == 0:
            logger.info(f"  Unfroze {n_unfrozen} LoRA parameter tensors")

        # Eval after stage 1
        if eval_triplet_path is not None and rank == 0:
            val_acc, val_task_acc = evaluate_rank0(encoder, eval_triplet_path, store, device, args)
            logger.info(f"  [post-stage1] val triplet accuracy: {val_acc:.4f}")
            for task, acc in sorted(val_task_acc.items()):
                logger.info(f"    {task}: {acc:.4f}")
            if use_wandb:
                import wandb
                ld = {"stage1/val_triplet_acc": val_acc}
                for task, acc in val_task_acc.items():
                    ld[f"stage1/val/{task}/triplet_acc"] = acc
                wandb.log(ld, step=s1_step)

    # Wandb step offset so Stage 2 steps are monotonically after Stage 1 steps.
    wandb_step_offset = s1_step if args.stage1_steps > 0 else 0

    # ── Stage 2 optimizer + LR scheduler ─────────────────────────────────────
    # Estimate total steps from data epoch 0 (reused across all training epochs)
    est_ds = EpochBatchDataset(
        args.train_data_dir, 0, args.tasks, store,
        args.batch_size, 0, args.seed, world_size, rank,
        args.pad_to_num_events,
    )
    n_batches_per_rank_0 = len(est_ds)
    del est_ds

    n_opt_steps_per_epoch = math.ceil(n_batches_per_rank_0 / args.grad_accum)
    total_opt_steps       = n_opt_steps_per_epoch * args.epochs
    warmup_steps          = int(total_opt_steps * args.warmup_ratio)

    param_groups = _make_param_groups(encoder, args.lr_proj, args.lr_lora)
    if rank == 0:
        for i, g in enumerate(param_groups):
            logger.info(f"  Param group {i}: {len(g['params'])} tensors, lr={g['lr']}")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        fused=True,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if rank == 0:
        logger.info(f"Training: {args.epochs} epochs, "
                    f"~{n_batches_per_rank_0} batches/rank/epoch, "
                    f"{total_opt_steps} optimizer steps total")
        logger.info(f"  lr_proj={args.lr_proj}, lr_lora={args.lr_lora}, "
                    f"warmup={warmup_steps} steps")

    # ── Training loop (stage 2) ───────────────────────────────────────────────
    best_val_acc = 0.0
    opt_step     = 0

    for epoch in range(args.epochs):
        if is_ddp:
            dist.barrier()

        data_epoch = epoch % n_data_epochs
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: data epoch {data_epoch} …")

        epoch_ds, train_loader = _make_loader(data_epoch, epoch)
        n_batches_this_epoch   = len(epoch_ds)

        encoder.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=(rank != 0),
            dynamic_ncols=True,
            total=n_batches_this_epoch,
        )

        batch_idx = -1
        for batch_idx, batch in enumerate(pbar):
            if args.debug_batches is not None and batch_idx >= args.debug_batches:
                break

            labels_t = batch["labels"]

            is_update_step = (
                (batch_idx + 1) % args.grad_accum == 0
                or (batch_idx + 1) == n_batches_this_epoch
            )
            sync_ctx = nullcontext() if (is_update_step or not is_ddp) \
                       else encoder.no_sync()

            with sync_ctx:
                emb, pre_emb = encoder(
                    batch["event_embs"].to(device),
                    batch["event_mask"].to(device),
                    batch["task_idxs"].to(device),
                    return_pre_emb=True,
                )
                supcon_loss = _supervised_contrastive_loss(
                    pre_emb, labels_t.to(device), args.stage1_temperature
                )
                if args.var_reg_weight > 0.0:
                    std = torch.sqrt(pre_emb.var(dim=0) + 1e-4)
                    var_loss = torch.relu(1.0 - std).mean()
                else:
                    var_loss = supcon_loss.new_zeros(())

                loss = supcon_loss + args.var_reg_weight * var_loss
                loss = loss / args.grad_accum
                loss.backward()

            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in encoder.parameters() if p.requires_grad),
                    args.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if rank == 0:
                    loss_now  = loss.item() * args.grad_accum
                    gnorm_now = grad_norm.item() if isinstance(grad_norm, torch.Tensor) \
                                else grad_norm
                    # Use proj-group LR as the displayed LR (group 0)
                    lr_now = optimizer.param_groups[0]["lr"]

                    with torch.no_grad():
                        metrics = _compute_batch_metrics(emb.detach(), labels_t.to(device))
                        pre_var_now = pre_emb.detach().float().var(dim=0).mean().item()
                        supcon_now = supcon_loss.item()
                        var_loss_now = var_loss.item() if args.var_reg_weight > 0.0 else 0.0

                    pbar.set_postfix(
                        loss=f"{loss_now:.4f}",
                        lsup=f"{supcon_now:.4f}",
                        cdist=f"{metrics['cdist']:.3f}",
                        dpp=f"{metrics['d_pp']:.3f}",
                        dpn=f"{metrics['d_pn']:.3f}",
                        var=f"{metrics['var_all']:.4f}",
                        prevar=f"{pre_var_now:.4f}",
                        gnorm=f"{gnorm_now:.3f}",
                        lr=f"{lr_now:.2e}",
                    )

                    if use_wandb and opt_step % args.log_steps == 0:
                        import wandb
                        log_dict = {
                            "train/loss":         loss_now,
                            "train/loss_supcon":  supcon_now,
                            "train/loss_var":     var_loss_now,
                            "train/gnorm":        gnorm_now,
                            "train/lr_proj":      optimizer.param_groups[0]["lr"],
                            "train/cdist":        metrics["cdist"],
                            "train/dist_pos_pos": metrics["d_pp"],
                            "train/dist_pos_neg": metrics["d_pn"],
                            "train/var_all":      metrics["var_all"],
                            "train/pre_var_all":  pre_var_now,
                            "train/var_pos":      metrics["var_pos"],
                            "train/var_neg":      metrics["var_neg"],
                            "epoch": epoch + (batch_idx + 1) / max(n_batches_this_epoch, 1),
                        }
                        if len(optimizer.param_groups) > 1:
                            log_dict["train/lr_lora"] = optimizer.param_groups[1]["lr"]
                        wandb.log(log_dict, step=wandb_step_offset + opt_step)

            epoch_loss += loss.item() * args.grad_accum

        n_batches_ran = batch_idx + 1
        avg_loss      = epoch_loss / max(n_batches_ran, 1)
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")

        # ── Eval ──────────────────────────────────────────────────────────────
        if eval_triplet_path is not None and rank == 0:
            val_acc, val_task_acc = evaluate_rank0(encoder, eval_triplet_path, store, device, args)
            logger.info(f"  val triplet accuracy: {val_acc:.4f}")
            for task, acc in sorted(val_task_acc.items()):
                logger.info(f"    {task}: {acc:.4f}")
            if use_wandb:
                import wandb
                log_dict = {
                    "val/triplet_acc":  val_acc,
                    "train/epoch_loss": avg_loss,
                    "epoch":            epoch + 1,
                }
                for task, acc in val_task_acc.items():
                    log_dict[f"val/{task}/triplet_acc"] = acc
                wandb.log(log_dict, step=wandb_step_offset + opt_step)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
                raw_enc.save_checkpoint(run_dir / "best")
                logger.info(f"  New best: {best_val_acc:.4f}")

        if rank == 0:
            raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
            raw_enc.save_checkpoint(run_dir / f"epoch_{epoch+1}")

        if is_ddp:
            dist.barrier()

    # ── Final save ─────────────────────────────────────────────────────────────
    if rank == 0:
        raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
        raw_enc.save_checkpoint(run_dir / "final")
        tokenizer.save_pretrained(str(run_dir / "tokenizer"))
        logger.info(f"Done. Best val triplet accuracy: {best_val_acc:.4f}")

    if use_wandb and wandb_run:
        import wandb
        wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
