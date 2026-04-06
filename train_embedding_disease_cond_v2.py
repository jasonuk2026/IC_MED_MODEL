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

Architecture / training loop are unchanged from v1.

Usage (single node, 4 GPUs)
─────────────────────────────
  torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py \\
      --train_data_paths data/prepared/all/train/shard_*.parquet \\
      --eval_data_paths  EHRSHOT_ASSETS/llm_eval_data/new_*/val.parquet \\
      --bert_embeddings  data/biolinkbert_embeddings/embeddings.npy \\
      --bf16 --flash_attn

Single GPU
──────────
  python train_embedding_disease_cond_v2.py \\
      --train_data_paths data/prepared/all/train/shard_0.parquet \\
      --eval_data_paths  EHRSHOT_ASSETS/llm_eval_data/new_*/val.parquet \\
      --bert_embeddings  data/biolinkbert_embeddings/embeddings.npy

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
from sentence_transformers.losses import BatchHardSoftMarginTripletLoss, BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from model import DiseaseAwareEHREncoder


def supervised_infonce_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised InfoNCE (SupCon) loss.

    For each anchor i, positives = same-label samples (j != i),
    negatives = all other samples (denominator = all j != i).

    Loss = mean_i[ -log( sum_{j in pos(i)} exp(sim(i,j)/T)
                        / sum_{j != i}     exp(sim(i,j)/T) ) ]

    Embeddings must be L2-normalised (cosine similarity = dot product).
    Gradient is well-defined even when embeddings collapse: the softmax
    always spans multiple distinct vectors, preventing vanishing gradients.
    """
    B      = embeddings.size(0)
    device = embeddings.device

    # (B, B) cosine similarity matrix, scaled by temperature
    sim = (embeddings @ embeddings.T) / temperature

    # Masks
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # (B, B)

    # Exclude self from denominator via -inf before logsumexp
    sim_no_self = sim.masked_fill(self_mask, float("-inf"))

    # log denominator: logsumexp over all j != i  (handles -inf safely)
    log_denom = torch.logsumexp(sim_no_self, dim=1)   # (B,)

    # log p(j|i) = sim[i,j] - log_denom[i]  for j != i
    log_probs = sim_no_self - log_denom.unsqueeze(1)   # (B, B), diag = -inf

    # Sum log-probs over positives.
    # Use masked_fill(~pos_mask, 0) instead of multiplying to avoid 0 * (-inf) = NaN.
    n_pos = pos_mask.sum(dim=1).float()               # (B,)
    valid = n_pos > 0
    if not valid.any():
        return embeddings.sum() * 0.0                 # differentiable zero

    pos_log_probs   = log_probs.masked_fill(~pos_mask, 0.0)  # safe: only -inf at diag, which pos_mask already excludes
    loss_per_anchor = -pos_log_probs.sum(dim=1) / n_pos.clamp(min=1)
    return loss_per_anchor[valid].mean()

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
    """Memory-mapped BioLinkBERT event embeddings array.

    Unlike v1's EventLookup, no key→index mapping is needed: the embedding_idx
    is already stored in each event JSON inside the parquet timeline column.
    """

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


# ── Per-epoch in-memory data store ────────────────────────────────────────────

class PreparedEpochData:
    """Loads all tasks' prepare_task_data_v2.py output for one data epoch.

    Data layout (per task): rows are interleaved pos/neg blocks of max_batch_size.
    Each training batch of batch_size rows is a contiguous slice within one block.

    Schema: patient_id (int64), task_idx (int16), label (int8),
            event_ids (list<int32>), source_row (int32)
    """

    def __init__(
        self,
        data_dir:          str,
        data_epoch_idx:    int,
        tasks:             list[str],
        store:             EmbeddingStore,
        pad_to_num_events: int | None = None,
    ):
        self.store             = store
        self.pad_to_num_events = pad_to_num_events
        self.max_batch_size:   int | None = None
        self.num_max_batches:  dict[int, int] = {}   # task_idx → n_max_batches
        self.event_ids:        dict[int, list[np.ndarray]] = {}   # task_idx → per-row eids
        self.labels:           dict[int, np.ndarray] = {}         # task_idx → label array
        self.valid_task_idxs:  list[int] = []

        idx_to_name = {v: k for k, v in TASK_2_IDX.items()}

        for task in sorted(tasks):
            task_idx = TASK_2_IDX[task]
            p_parquet = Path(data_dir) / task / f"train_prepared_{data_epoch_idx:03d}.parquet"
            p_json    = Path(data_dir) / task / f"train_prepared_{data_epoch_idx:03d}.json"

            if not p_parquet.exists():
                logger.warning(f"  [{task}] {p_parquet} not found — skipping")
                continue

            with open(p_json) as f:
                meta = json.load(f)

            mbs = meta["max_batch_size"]
            if self.max_batch_size is None:
                self.max_batch_size = mbs
            elif self.max_batch_size != mbs:
                raise ValueError(
                    f"Inconsistent max_batch_size across tasks: "
                    f"{self.max_batch_size} vs {mbs} for {task}"
                )

            self.num_max_batches[task_idx] = meta["num_batches"]
            logger.info(f"  [{task}] Loading {p_parquet.name} "
                        f"({meta['num_batches']} max-batches, "
                        f"{meta['used_rows']:,} rows) …")

            df = pd.read_parquet(str(p_parquet), columns=["label", "event_ids"])
            self.event_ids[task_idx] = [np.array(e, dtype=np.int32) for e in df["event_ids"]]
            self.labels[task_idx]    = df["label"].values.astype(np.int8)
            self.valid_task_idxs.append(task_idx)
            del df

        if not self.valid_task_idxs:
            raise ValueError("No valid prepared data found for any task.")

    def get_batch(self, task_idx: int, start_row: int, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a collated batch dict for rows [start_row, start_row+batch_size)."""
        end_row   = start_row + batch_size
        eids_list = self.event_ids[task_idx][start_row:end_row]
        labels    = self.labels[task_idx][start_row:end_row].tolist()
        return _collate_event_embs(
            eids_list, [task_idx] * len(eids_list), labels,
            self.store.embeddings, self.pad_to_num_events,
        )


def build_epoch_schedule(
    epoch_data: PreparedEpochData,
    batch_size: int,
    training_epoch: int,
    seed: int,
    world_size: int,
) -> list[tuple[int, int]]:
    """Build a shuffled list of (task_idx, start_row) for one training epoch.

    Each max_batch of max_batch_size rows yields (max_batch_size // batch_size)
    training batches. The full list is shuffled, truncated to a world_size
    multiple, and returned (all ranks get the same full list and slice by rank).
    """
    sub_per_max = epoch_data.max_batch_size // batch_size
    batches: list[tuple[int, int]] = []

    for task_idx in epoch_data.valid_task_idxs:
        n_max = epoch_data.num_max_batches[task_idx]
        for max_i in range(n_max):
            for sub_i in range(sub_per_max):
                start_row = max_i * epoch_data.max_batch_size + sub_i * batch_size
                batches.append((task_idx, start_row))

    rng = random.Random(seed + training_epoch * 1337)
    rng.shuffle(batches)

    # Truncate to world_size multiple for balanced DDP
    n_total = (len(batches) // world_size) * world_size
    return batches[:n_total]


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
    """Sample anchor/positive/negative triplets from eval data and save to parquet.

    Triplet constraints (same as before):
      - anchor and positive from different patients
      - anchor and negative from different patient than anchor

    Returns the number of triplets saved.
    """
    pos: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    neg: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)

    for path in eval_data_paths:
        df = pd.read_parquet(path, columns=["patient_id", "task_idx", "label", "event_ids"])
        for row in df.itertuples(index=False):
            task_idx = int(row.task_idx)
            pid      = int(row.patient_id)
            eids     = list(row.event_ids)
            (pos if int(row.label) == 1 else neg)[task_idx].append((pid, eids))
        del df

    rng = random.Random(seed)
    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    records: list[dict] = []

    for task_idx in sorted(set(list(pos.keys()) + list(neg.keys()))):
        pos_list = pos.get(task_idx, [])
        neg_list = neg.get(task_idx, [])

        if len(pos_list) < 2 or not neg_list:
            logger.warning(f"  [{idx_to_name.get(task_idx, task_idx)}] "
                           f"insufficient eval data for triplets — skipping")
            continue

        pos_pool = list(range(len(pos_list)))
        neg_pool = list(range(len(neg_list)))
        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)

        n = min(n_triplets_per_task, len(pos_list))
        ni_cursor = 0
        task_count = 0

        for i in range(n):
            ai          = pos_pool[i]
            anchor_pid  = pos_list[ai][0]
            anchor_eids = pos_list[ai][1]

            # Positive: different patient
            pi = None
            for offset in range(1, len(pos_pool)):
                cand = pos_pool[(i + offset) % len(pos_pool)]
                if pos_list[cand][0] != anchor_pid:
                    pi = cand
                    break
            if pi is None:
                continue

            # Negative: different patient from anchor
            ni = None
            for offset in range(len(neg_pool)):
                cand = neg_pool[(ni_cursor + offset) % len(neg_pool)]
                if neg_list[cand][0] != anchor_pid:
                    ni = cand
                    break
            ni_cursor += 1
            if ni is None:
                continue

            records.append({
                "task_idx":      task_idx,
                "anchor_eids":   anchor_eids,
                "positive_eids": pos_list[pi][1],
                "negative_eids": neg_list[ni][1],
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
    """Build left-padded prefix token ids and attention masks for all tasks.

    Each task's prefix is  PROMPT_PREFIX + disease_name,
    e.g. "Please predict disease hypertension".
    All prefixes are left-padded to the same length (max across tasks) using
    the tokenizer's pad_token_id so that torch.compile sees static shapes.

    Returns:
        task_prefix_ids  : (num_tasks, max_P)  long  — left-padded token ids
        task_prefix_mask : (num_tasks, max_P)  long  — 0=pad, 1=real token
    """
    tasks_sorted  = sorted(TASK_2_DISEASE_NAME)  # stable order matching TASK_2_IDX
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
    """Batch-level Dataset for evaluation.

    Receives a flat list of (event_ids_array, task_idx) entries (already split
    into anchor / positive / negative segments by evaluate_ddp) and yields
    collated CPU tensors batch-by-batch.
    """

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
    """Rank-0-only triplet evaluation using pre-computed triplet parquet.

    Loads triplets from eval_triplet_path (written by precompute_eval_triplets),
    encodes anchor/positive/negative, computes accuracy per task and overall.
    """
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

    # Build interleaved entry list: [a0, p0, n0, a1, p1, n1, ...]
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

    embs      = torch.cat(all_embs, dim=0)   # (3 * n_triplets, D)
    anchors   = embs[0::3]                    # (n_triplets, D)
    positives = embs[1::3]
    negatives = embs[2::3]

    d_ap    = (anchors - positives).norm(dim=1)
    d_an    = (anchors - negatives).norm(dim=1)
    correct = d_ap < d_an                    # (n_triplets,) bool

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
        description="Disease-aware EHR embedding (v2, v6 parquet format).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--eval_only",  action="store_true")
    p.add_argument("--debug_batches", type=int, default=None,
                   help="If set, stop each epoch after this many batches (for DDP smoke-test).")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a saved checkpoint dir (contains lora/ + extra_modules.pt).")

    # Data
    p.add_argument("--train_data_dir", default=None,
                   help="Base directory for prepared training data from prepare_task_data_v2.py "
                        "(contains {task}/train_prepared_{epoch:03d}.parquet). "
                        "Required unless --eval_only.")
    p.add_argument("--tasks", nargs="+", default=list(sorted(TASK_2_DISEASE_NAME.keys())),
                   help="Tasks to train on (default: all tasks in TASK_2_IDX).")
    p.add_argument("--eval_data_paths",  nargs="+", default=None,
                   help="Eval parquets (output of build_eval_task_data.py).")

    # BioLinkBERT event embeddings
    p.add_argument("--bert_embeddings", required=True,
                   help="embeddings.npy produced by extract_biolinkbert_embeddings.py.")

    # Qwen model
    p.add_argument("--model_name",   default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--flash_attn",   action="store_true")
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--fp16",         action="store_true")
    p.add_argument("--qlora",        action="store_true",
                   help="4-bit NF4 QLoRA. Single GPU only.")

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
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--log_steps",    type=int,   default=10)

    # Loss / Eval
    p.add_argument("--loss",                     choices=["infonce", "triplet"], default="infonce",
                   help="Loss function. infonce=SupCon (recommended), triplet=BatchHardTripletLoss.")
    p.add_argument("--temperature",              type=float, default=0.07,
                   help="Temperature for InfoNCE loss.")
    p.add_argument("--triplet_margin",           type=float, default=0.3,
                   help="Margin for BatchHardTripletLoss (only used when --loss triplet).")
    p.add_argument("--n_eval_triplets_per_task", type=int,   default=256)
    p.add_argument("--eval_batch_size",          type=int,   default=32)
    p.add_argument("--pad_to_num_events",        type=int,   default=None)

    p.add_argument("--num_workers",      type=int, default=4)
    p.add_argument("--prefetch_factor",  type=int, default=4)

    p.add_argument("--compile", action="store_true",
                   help="torch.compile (requires --pad_to_num_events for static shapes).")

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
        raise RuntimeError("QLoRA is incompatible with multi-GPU DDP. Use single GPU.")
    if not args.eval_only and args.train_data_dir is None:
        raise ValueError("--train_data_dir is required unless --eval_only is set.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(args.output_dir) / timestamp
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run dir: {run_dir}")
        logger.info(f"World size: {world_size}")

    # ── wandb (rank 0 only) ───────────────────────────────────────────────────
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

    # ── BioLinkBERT event embeddings (shared read-only mmap) ─────────────────
    store = EmbeddingStore(args.bert_embeddings)

    # ── Build Qwen model + tokenizer ───────────────────────────────────────────
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

    # ── Pre-compute eval triplets (rank 0 only, once before training) ─────────
    # Stored in run_dir so rank 0 can reload per epoch. Other ranks do not need
    # this path because eval is rank-0-only.
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

    # ── Eval-only mode ─────────────────────────────────────────────────────────
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

    # ── Count available data epochs ────────────────────────────────────────────
    first_task = sorted(args.tasks)[0]
    task_dir   = Path(args.train_data_dir) / first_task
    data_epoch_files = sorted(task_dir.glob("train_prepared_*.parquet"))
    n_data_epochs    = len(data_epoch_files)
    if n_data_epochs == 0:
        raise ValueError(f"No train_prepared_*.parquet found in {task_dir}")
    if rank == 0:
        logger.info(f"Found {n_data_epochs} data epoch(s) (task dir: {task_dir})")

    # ── Estimate schedule size for LR scheduler (use data epoch 0) ────────────
    if rank == 0:
        logger.info("Estimating epoch size from data epoch 0 …")
    epoch_data_0 = PreparedEpochData(
        args.train_data_dir, 0, args.tasks, store, args.pad_to_num_events
    )
    schedule_0            = build_epoch_schedule(epoch_data_0, args.batch_size, 0, args.seed, world_size)
    n_batches_per_rank_0  = len(schedule_0) // world_size
    del epoch_data_0

    # ── Optimizer + LR scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        fused=True,
    )

    n_opt_steps_per_epoch = math.ceil(n_batches_per_rank_0 / args.grad_accum)
    total_opt_steps       = n_opt_steps_per_epoch * args.epochs
    warmup_steps          = int(total_opt_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.loss == "infonce":
        _loss_fn  = lambda labels, emb: supervised_infonce_loss(emb, labels, args.temperature)
        loss_desc = f"SupervisedInfoNCE (temperature={args.temperature})"
    elif args.triplet_margin > 0:
        loss_module = BatchHardTripletLoss(
            model=None,
            distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
            margin=args.triplet_margin,
        )
        _loss_fn  = lambda labels, emb: loss_module.batch_hard_triplet_loss(labels, emb)
        loss_desc = f"BatchHardTripletLoss (margin={args.triplet_margin})"
    else:
        loss_module = BatchHardSoftMarginTripletLoss(
            model=None,
            distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
        )
        _loss_fn  = lambda labels, emb: loss_module.batch_hard_triplet_soft_margin_loss(labels, emb)
        loss_desc = "BatchHardSoftMarginTripletLoss (soft margin)"

    if rank == 0:
        logger.info(f"Training: {args.epochs} epochs, "
                    f"~{n_batches_per_rank_0} batches/rank/epoch, "
                    f"{total_opt_steps} optimizer steps total")
        logger.info(loss_desc)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    opt_step     = 0

    for epoch in range(args.epochs):
        if is_ddp:
            dist.barrier()

        # Map training epoch → data epoch (wraps around if fewer data epochs)
        data_epoch = epoch % n_data_epochs
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: loading data epoch {data_epoch} …")

        epoch_data = PreparedEpochData(
            args.train_data_dir, data_epoch, args.tasks, store, args.pad_to_num_events
        )
        schedule             = build_epoch_schedule(epoch_data, args.batch_size, epoch, args.seed, world_size)
        my_schedule          = schedule[rank::world_size]   # this rank's batches
        n_batches_this_epoch = len(my_schedule)

        encoder.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(my_schedule),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=(rank != 0),
            dynamic_ncols=True,
            total=n_batches_this_epoch,
        )

        batch_idx = -1
        for batch_idx, (task_idx, start_row) in pbar:
            if args.debug_batches is not None and batch_idx >= args.debug_batches:
                break

            batch    = epoch_data.get_batch(task_idx, start_row, args.batch_size)
            labels_t = batch["labels"]

            is_update_step = (
                (batch_idx + 1) % args.grad_accum == 0
                or (batch_idx + 1) == n_batches_this_epoch
            )
            sync_ctx = nullcontext() if (is_update_step or not is_ddp) \
                       else encoder.no_sync()

            with sync_ctx:
                emb = encoder(
                    batch["event_embs"].to(device),
                    batch["event_mask"].to(device),
                    batch["task_idxs"].to(device),
                )
                loss = _loss_fn(labels_t.to(device), emb)
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
                    lr_now    = scheduler.get_last_lr()[0]
                    loss_now  = loss.item() * args.grad_accum
                    gnorm_now = grad_norm.item() if isinstance(grad_norm, torch.Tensor) \
                                else grad_norm

                    # Pairwise distance diagnostics
                    d_pp = d_pn = float("nan")
                    with torch.no_grad():
                        e     = emb.detach().float()
                        lbl   = labels_t.detach().to(device)
                        e_pos = e[lbl == 1]
                        e_neg = e[lbl == 0]
                        if e_pos.size(0) >= 2:
                            sim_pp = e_pos @ e_pos.T
                            n_pos_ = e_pos.size(0)
                            tri    = torch.triu(torch.ones(n_pos_, n_pos_, dtype=torch.bool, device=e.device), diagonal=1)
                            d_pp   = (1 - sim_pp[tri]).mean().item()
                        if e_pos.size(0) >= 1 and e_neg.size(0) >= 1:
                            d_pn   = (1 - (e_pos @ e_neg.T)).mean().item()

                    pbar.set_postfix(
                        loss=f"{loss_now:.4f}",
                        dpp=f"{d_pp:.3f}",
                        dpn=f"{d_pn:.3f}",
                        gnorm=f"{gnorm_now:.3f}",
                        lr=f"{lr_now:.2e}",
                    )
                    if use_wandb and opt_step % args.log_steps == 0:
                        import wandb
                        wandb.log({
                            "train/loss":         loss_now,
                            "train/gnorm":        gnorm_now,
                            "train/lr":           lr_now,
                            "train/dist_pos_pos": d_pp,
                            "train/dist_pos_neg": d_pn,
                            "epoch":              epoch + (batch_idx + 1) / max(n_batches_this_epoch, 1),
                        }, step=opt_step)

            epoch_loss += loss.item() * args.grad_accum

        del epoch_data   # free in-memory data before eval

        n_batches_ran = batch_idx + 1   # 0 if no batches ran
        avg_loss      = epoch_loss / max(n_batches_ran, 1)
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")

        # ── Eval (rank 0 only — no barrier needed) ─────────────────────────────
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
                wandb.log(log_dict, step=opt_step)
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
