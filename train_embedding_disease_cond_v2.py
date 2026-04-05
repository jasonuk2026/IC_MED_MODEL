#!/usr/bin/env python3
"""
train_embedding_disease_cond_v2.py

Disease-conditioned EHR embedding model — adapted for the v6 dataset format
produced by dataset/build_task_data.py.

Key differences from v1:
- Input parquet schema: patient_id, prediction_time, label (string), label_type,
  split, timeline (newline-delimited JSON), task_name, task_description,
  num_events, sample_event_indices, oversample_copy_idx.
- Each event JSON already contains an "embedding_idx" field pointing to the
  corresponding row in embeddings.npy — no separate event_index.parquet lookup
  needed.  --bert_index is therefore removed.
- Label is a string ("0"/"1" for binary tasks).  Positive = int(float(label)) != 0.
- Batch-level 1:1 pos/neg balance is enforced by construction in EpochShuffledDataset
  (half positives + half negatives per batch), consistent with the approximate 1:1
  ratio already produced by the data collection strategy.

Architecture / training loop are unchanged from v1.

Usage (single node, 4 GPUs)
─────────────────────────────
  torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py \\
      --data_paths     EHRSHOT_ASSETS/llm_data_v6/new_hypertension/train.parquet \\
      --val_data_paths EHRSHOT_ASSETS/llm_data_v6/new_hypertension/val.parquet \\
      --bert_embeddings data/biolinkbert_embeddings/embeddings.npy \\
      --bf16 --flash_attn

Single GPU
──────────
  python train_embedding_disease_cond_v2.py \\
      --data_paths EHRSHOT_ASSETS/llm_data_v6/new_hypertension/train.parquet \\
      --bert_embeddings data/biolinkbert_embeddings/embeddings.npy

NOTE: QLoRA (--qlora) is incompatible with multi-GPU DDP.
"""

import os
import math
import json
import random
import logging
import argparse
from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from utils.h2d import CudaPrefetcher
from utils.async_dataloader import AsyncDataLoader

import pandas as pd
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, logging as hf_logging
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sentence_transformers.losses import BatchHardSoftMarginTripletLoss, BatchHardTripletLossDistanceFunction
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_positive(label_str) -> bool:
    """Return True if label represents a positive sample.

    Handles numeric strings ("0"/"1"), boolean strings ("True"/"False"),
    and actual bool/int values.
    """
    if isinstance(label_str, bool):
        return label_str
    s = str(label_str).strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(float(s)) != 0
    except (ValueError, TypeError):
        return False


def _parse_timeline_event_ids(timeline: str) -> list[int]:
    """Parse a newline-delimited timeline string and extract embedding_idx values.

    Each line is a JSON object with an "embedding_idx" field (int).
    Lines without "embedding_idx" or with null are skipped.
    """
    eids: list[int] = []
    for line in timeline.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = ev.get("embedding_idx")
        if eid is not None:
            eids.append(int(eid))
    return eids


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


# ── Lazy data index ───────────────────────────────────────────────────────────

class LazyDataIndex:
    """Lightweight per-split index for the v6 parquet schema.

    Reads only split / label / task_name / timeline columns at init time.
    Each event's embedding_idx is extracted from the timeline JSON strings
    and stored as a compact int list — no parquet I/O during training.

    Attributes
    ----------
    pos : dict[task -> list[(file_idx, abs_row)]]  — positive samples per task
    neg : dict[task -> list[(file_idx, abs_row)]]  — negative samples per task
    """

    def __init__(self, paths: list[str], split: str | None, store: EmbeddingStore):
        self.file_paths = list(paths)
        self.store      = store

        pos: dict[str, list[tuple[int, int]]] = defaultdict(list)
        neg: dict[str, list[tuple[int, int]]] = defaultdict(list)
        task_counts: Counter = Counter()
        pos_counts:  Counter = Counter()

        # (file_idx, abs_row) → (task_name, np.ndarray[int32] of embedding_idx)
        # np.int32 costs 4 bytes/element vs ~28 bytes for a Python int in a list,
        # reducing peak memory by ~7x when storing thousands of event IDs per sample.
        self._preloaded: dict[tuple[int, int], tuple[str, np.ndarray]] = {}

        for file_idx, path in enumerate(self.file_paths):
            logger.info(f"Indexing + pre-loading {path} …")
            df = pd.read_parquet(
                path,
                columns=["split", "label", "task_name", "timeline"],
            )
            for abs_row, (split_val, label, task_name, timeline) in enumerate(
                zip(df["split"], df["label"], df["task_name"], df["timeline"])
            ):
                if split is not None and split_val != split:
                    continue
                # Only train tasks that have a known disease name mapping.
                if task_name not in TASK_2_DISEASE_NAME:
                    continue

                is_pos = _is_positive(label)
                lbl    = 1 if is_pos else 0
                (pos if lbl else neg)[task_name].append((file_idx, abs_row))
                task_counts[task_name] += 1
                if lbl:
                    pos_counts[task_name] += 1

                # Extract embedding indices and store as compact int32 array.
                eids = _parse_timeline_event_ids(timeline if isinstance(timeline, str) else "")
                self._preloaded[(file_idx, abs_row)] = (
                    task_name,
                    np.array(eids, dtype=np.int32),
                )
            del df

        self.pos: dict[str, list[tuple[int, int]]] = dict(pos)
        self.neg: dict[str, list[tuple[int, int]]] = dict(neg)

        logger.info(f"  {sum(task_counts.values())} samples "
                    f"(split={split!r}) across {len(task_counts)} tasks  "
                    f"[{len(self._preloaded):,} pre-loaded]")
        for task in sorted(task_counts):
            logger.info(f"    {task}: {pos_counts[task]} pos / "
                        f"{task_counts[task] - pos_counts[task]} neg")

    def read_rows(self, entries: list[tuple[int, int]]) -> list[dict]:
        """Return pre-loaded event IDs and task index for a list of (file_idx, abs_row) entries."""
        return [
            {
                "event_ids": self._preloaded[(fi, ar)][1],
                "task":      self._preloaded[(fi, ar)][0],
                "task_idx":  TASK_2_IDX[self._preloaded[(fi, ar)][0]],
            }
            for fi, ar in entries
        ]


# ── Training dataset ──────────────────────────────────────────────────────────

class EpochShuffledDataset(Dataset):
    """PyTorch Dataset where each item IS a full collated batch.

    Each batch contains exactly half positives and half negatives from the
    same task — enforcing strict 1:1 pos/neg balance regardless of the overall
    ratio in the data files.

    __len__ returns the number of batches this rank handles.
    __getitem__(j) reads all samples for batch j, collates and returns a dict
    of CPU tensors.

    Call set_epoch(epoch) before each epoch to regenerate the shuffled schedule.
    """

    def __init__(
        self,
        index:             LazyDataIndex,
        batch_size:          int,
        rank:                int = 0,
        world_size:          int = 1,
        seed:                int = 42,
        pad_to_num_events:   int | None = None,
    ):
        if batch_size < 4 or batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even and ≥4, got {batch_size}")
        self.index              = index
        self.batch_size         = batch_size
        self.half               = batch_size // 2
        self.rank               = rank
        self.world_size         = world_size
        self.seed               = seed
        self.pad_to_num_events  = pad_to_num_events

        half = self.half
        self.valid_tasks = sorted(
            t for t in index.pos
            if len(index.pos[t]) >= half and len(index.neg.get(t, [])) >= half
        )
        if not self.valid_tasks:
            raise ValueError(
                f"No task has ≥{half} positives and ≥{half} negatives. "
                f"Reduce --batch_size or provide more data."
            )
        skipped = set(index.pos) - set(self.valid_tasks)
        if skipped:
            logger.warning(f"  EpochShuffledDataset: skipped tasks with insufficient "
                           f"data (need ≥{half} per class): {sorted(skipped)}")

        # Number of batches = min(pos, neg) // half  across all valid tasks.
        self._n_batches_per_task = min(
            min(len(index.pos[t]), len(index.neg[t])) // half
            for t in self.valid_tasks
        )
        total_batches  = len(self.valid_tasks) * self._n_batches_per_task
        self.n_batches = len(range(rank, total_batches, world_size))

        # _batches[j] = [(file_idx, abs_row, label), ...] for batch j.
        self._batches: list[list[tuple[int, int, int]]] = []

    def set_epoch(self, epoch: int) -> None:
        """Regenerate the shuffled batch list for this epoch."""
        g    = torch.Generator()
        g.manual_seed(self.seed + epoch)
        half = self.half
        n    = self._n_batches_per_task

        task_pos_perm = {
            t: torch.randperm(len(self.index.pos[t]), generator=g).tolist()
            for t in self.valid_tasks
        }
        task_neg_perm = {
            t: torch.randperm(len(self.index.neg[t]), generator=g).tolist()
            for t in self.valid_tasks
        }

        flat    = [(t, b) for t in self.valid_tasks for b in range(n)]
        flat    = [flat[i] for i in torch.randperm(len(flat), generator=g).tolist()]
        my_flat = flat[self.rank :: self.world_size]

        batches: list[list[tuple[int, int, int]]] = []
        for task, b in my_flat:
            pos_list = self.index.pos[task]
            neg_list = self.index.neg[task]
            entries: list[tuple[int, int, int]] = []
            for i in range(half):
                fi, ar = pos_list[task_pos_perm[task][b * half + i]]
                entries.append((fi, ar, 1))
            for i in range(half):
                fi, ar = neg_list[task_neg_perm[task][b * half + i]]
                entries.append((fi, ar, 0))
            batches.append(entries)

        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, j: int) -> dict[str, torch.Tensor]:
        batch_entries = self._batches[j]
        entries = [(fi, ar) for fi, ar, _   in batch_entries]
        labels  = [lbl      for _,  _,  lbl in batch_entries]

        rows       = self.index.read_rows(entries)
        embeddings = self.index.store.embeddings

        embs_list:     list[np.ndarray] = []
        task_idx_list: list[int]        = []
        for row in rows:
            eids = row["event_ids"]
            if self.pad_to_num_events is not None:
                eids = eids[:self.pad_to_num_events]
            if not eids:
                bert_dim = embeddings.shape[1]
                embs_list.append(np.zeros((1, bert_dim), dtype=np.float32))
            else:
                embs_list.append(
                    embeddings[eids].astype(np.float32)
                )
            task_idx_list.append(row["task_idx"])

        B        = len(embs_list)
        max_e    = self.pad_to_num_events if self.pad_to_num_events is not None \
                   else max(e.shape[0] for e in embs_list)
        bert_dim = embs_list[0].shape[1]
        padded   = np.zeros((B, max_e, bert_dim), dtype=np.float32)
        mask     = np.zeros((B, max_e),            dtype=np.int64)
        for i, embs in enumerate(embs_list):
            n = embs.shape[0]
            padded[i, :n] = embs
            mask[i, :n]   = 1

        return {
            "event_embs": torch.from_numpy(padded),
            "event_mask": torch.from_numpy(mask),
            "task_idxs":  torch.tensor(task_idx_list, dtype=torch.long),
            "labels":     torch.tensor(labels, dtype=torch.long),
        }


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
    """Batch-level Dataset for evaluation."""

    def __init__(
        self,
        entries:           list[tuple[int, int]],
        index:             LazyDataIndex,
        batch_size:        int,
        pad_to_num_events: int | None = None,
    ):
        self.index             = index
        self.pad_to_num_events = pad_to_num_events
        self._batches: list[list[tuple[int, int]]] = [
            entries[i : i + batch_size]
            for i in range(0, len(entries), batch_size)
        ]

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, j: int) -> dict[str, torch.Tensor]:
        rows       = self.index.read_rows(self._batches[j])
        embeddings = self.index.store.embeddings

        embs_list:     list[np.ndarray] = []
        task_idx_list: list[int]        = []
        for row in rows:
            eids = row["event_ids"]
            if self.pad_to_num_events is not None:
                eids = eids[:self.pad_to_num_events]
            if not eids:
                bert_dim = embeddings.shape[1]
                embs_list.append(np.zeros((1, bert_dim), dtype=np.float32))
            else:
                embs_list.append(
                    embeddings[eids].astype(np.float32)
                )
            task_idx_list.append(row["task_idx"])

        B        = len(embs_list)
        max_e    = self.pad_to_num_events if self.pad_to_num_events is not None \
                   else max(e.shape[0] for e in embs_list)
        bert_dim = embs_list[0].shape[1]
        padded   = np.zeros((B, max_e, bert_dim), dtype=np.float32)
        mask     = np.zeros((B, max_e),            dtype=np.int64)
        for i, embs in enumerate(embs_list):
            n = embs.shape[0]
            padded[i, :n] = embs
            mask[i, :n]   = 1

        return {
            "event_embs": torch.from_numpy(padded),
            "event_mask": torch.from_numpy(mask),
            "task_idxs":  torch.tensor(task_idx_list, dtype=torch.long),
        }


@torch.inference_mode()
def evaluate_ddp(
    encoder:    DiseaseAwareEHREncoder,
    val_index:  LazyDataIndex,
    device:     torch.device,
    args,
    rank:       int,
    world_size: int,
    is_ddp:     bool,
) -> tuple[float, dict[str, float]]:
    """Triplet accuracy distributed across all ranks.

    Returns (overall_acc, {task: acc}).

    Both pos and neg entry lists are shuffled before selecting triplets so that
    oversampled copies (which are consecutive in the index) are not preferentially
    picked together, which would inflate apparent similarity and bias evaluation.
    """
    raw_encoder = encoder.module if isinstance(encoder, DDP) else encoder
    raw_encoder.eval()

    rng = random.Random(args.seed)

    anchors_e:      list[tuple[int, int]] = []
    positives_e:    list[tuple[int, int]] = []
    negatives_e:    list[tuple[int, int]] = []
    task_of_triplet: list[str]            = []

    valid_tasks_for_eval = sorted(
        t for t in val_index.pos
        if len(val_index.pos[t]) >= 2 and val_index.neg.get(t)
    )

    for task in valid_tasks_for_eval:
        pos_entries = val_index.pos[task]
        neg_entries = val_index.neg[task]

        # Shuffle both pools so consecutive oversampled copies are spread out.
        pos_pool = list(range(len(pos_entries)))
        neg_pool = list(range(len(neg_entries)))
        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)

        n = min(args.n_eval_triplets_per_task, len(pos_entries))
        for i in range(n):
            ai = pos_pool[i]
            # Positive: next slot in the shuffled pool (wraps around), ensuring
            # anchor != positive while still respecting the shuffled order.
            pi = pos_pool[(i + 1) % len(pos_pool)]
            # Negative: drawn from the shuffled neg pool (wraps around).
            ni = neg_pool[i % len(neg_pool)]
            anchors_e.append(pos_entries[ai])
            positives_e.append(pos_entries[pi])
            negatives_e.append(neg_entries[ni])
            task_of_triplet.append(task)

    if not anchors_e:
        if rank == 0:
            logger.warning("No eval triplets could be built.")
        return 0.0, {}

    n_triplets    = len(anchors_e)
    my_triple_idx = list(range(rank, n_triplets, world_size))
    my_entries: list[tuple[int, int]] = (
        [anchors_e[t]   for t in my_triple_idx] +
        [positives_e[t] for t in my_triple_idx] +
        [negatives_e[t] for t in my_triple_idx]
    )

    eval_ds = EvalBatchDataset(my_entries, val_index,
                               args.eval_batch_size, args.pad_to_num_events)
    eval_dl = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: b[0],
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_prefetcher = CudaPrefetcher(
        AsyncDataLoader(eval_dl, buffer_size=4),
        device=device, cuda_keys=[],
    )

    all_emb_chunks = []
    for batch in tqdm(eval_prefetcher, desc="Evaluating", disable=(rank != 0),
                      dynamic_ncols=True, total=len(eval_ds)):
        all_emb_chunks.append(
            raw_encoder(
                batch["event_embs"],
                batch["event_mask"],
                batch["task_idxs"],
            ).cpu()
        )

    n_my = len(my_triple_idx)
    if all_emb_chunks and n_my > 0:
        embs    = torch.cat(all_emb_chunks, dim=0)   # (3 * n_my, D)
        d_ap    = (embs[:n_my] - embs[n_my:2*n_my]).norm(dim=1)
        d_an    = (embs[:n_my] - embs[2*n_my:]).norm(dim=1)
        correct = (d_ap < d_an)   # (n_my,) bool

        # ── Per-task local counts ─────────────────────────────────────────────
        my_tasks = [task_of_triplet[t] for t in my_triple_idx]
        local_task_correct: dict[str, torch.Tensor] = {}
        local_task_total:   dict[str, torch.Tensor] = {}
        for task in valid_tasks_for_eval:
            mask = torch.tensor([t == task for t in my_tasks], dtype=torch.bool)
            local_task_correct[task] = correct[mask].sum().to(torch.long).to(device)
            local_task_total[task]   = mask.sum().to(torch.long).to(device)

        local_correct = correct.sum().to(torch.long).to(device)
        local_total   = torch.tensor(n_my, dtype=torch.long, device=device)
    else:
        local_task_correct = {t: torch.tensor(0, dtype=torch.long, device=device)
                              for t in valid_tasks_for_eval}
        local_task_total   = {t: torch.tensor(0, dtype=torch.long, device=device)
                              for t in valid_tasks_for_eval}
        local_correct = torch.tensor(0, dtype=torch.long, device=device)
        local_total   = torch.tensor(0, dtype=torch.long, device=device)

    if is_ddp:
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total,   op=dist.ReduceOp.SUM)
        for task in valid_tasks_for_eval:
            dist.all_reduce(local_task_correct[task], op=dist.ReduceOp.SUM)
            dist.all_reduce(local_task_total[task],   op=dist.ReduceOp.SUM)

    overall_acc = (local_correct.float() / local_total.float()).item() \
                  if local_total.item() > 0 else 0.0

    task_acc: dict[str, float] = {}
    for task in valid_tasks_for_eval:
        n_t = local_task_total[task].item()
        task_acc[task] = (local_task_correct[task].float() / n_t).item() \
                         if n_t > 0 else 0.0

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
    p.add_argument("--checkpoint", default=None,
                   help="Path to a saved checkpoint dir (contains lora/ + extra_modules.pt).")

    # Data
    p.add_argument("--data_paths",     nargs="+", default=None,
                   help="Train parquet files (required unless --eval_only).")
    p.add_argument("--val_data_paths", nargs="+", default=None,
                   help="Validation parquet files.")
    p.add_argument("--val_split",      default="val", choices=["train", "val", "test"])

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
    p.add_argument("--n_eval_triplets_per_task", type=int,   default=100)
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
    if not args.eval_only and not args.data_paths:
        raise ValueError("--data_paths is required unless --eval_only is set.")

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

    # ── Eval-only mode ─────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.val_data_paths:
            raise ValueError("--eval_only requires --val_data_paths.")
        val_index = LazyDataIndex(args.val_data_paths, args.val_split, store)
        val_acc, val_task_acc = evaluate_ddp(
            encoder, val_index,
            device, args, rank, world_size, is_ddp,
        )
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

    # ── Lazy data indices & training dataset ──────────────────────────────────
    train_index = LazyDataIndex(args.data_paths, "train", store)
    train_ds    = EpochShuffledDataset(
        index=train_index,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        pad_to_num_events=args.pad_to_num_events,
    )

    val_index = None
    if args.val_data_paths:
        val_index = LazyDataIndex(args.val_data_paths, args.val_split, store)

    # ── Optimizer + LR scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        fused=True,
    )

    n_batches_per_epoch   = train_ds.n_batches
    n_opt_steps_per_epoch = math.ceil(n_batches_per_epoch / args.grad_accum)
    total_opt_steps       = n_opt_steps_per_epoch * args.epochs
    warmup_steps          = int(total_opt_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_module = BatchHardSoftMarginTripletLoss(
        model=None,
        distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
    )

    if rank == 0:
        logger.info(f"Training: {args.epochs} epochs, "
                    f"{n_batches_per_epoch} batches/epoch, "
                    f"{total_opt_steps} optimizer steps total")
        logger.info("BatchHardSoftMarginTripletLoss (soft margin)")

    # ── Training DataLoader ────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: b[0],
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    opt_step     = 0

    for epoch in range(args.epochs):
        if is_ddp:
            dist.barrier()
        train_ds.set_epoch(epoch)
        encoder.train()

        epoch_loss = 0.0
        optimizer.zero_grad()

        prefetcher = CudaPrefetcher(
            train_loader,
            device=device, cuda_keys=[],
        )
        pbar = tqdm(prefetcher, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=(rank != 0), dynamic_ncols=True, total=n_batches_per_epoch)

        for batch_idx, batch in enumerate(pbar):
            labels_t = batch["labels"]

            is_update_step = (
                (batch_idx + 1) % args.grad_accum == 0
                or (batch_idx + 1) == n_batches_per_epoch
            )
            sync_ctx = nullcontext() if (is_update_step or not is_ddp) \
                       else encoder.no_sync()

            with sync_ctx:
                emb = encoder(
                    batch["event_embs"],
                    batch["event_mask"],
                    batch["task_idxs"],
                )
                loss = loss_module.batch_hard_triplet_soft_margin_loss(labels_t, emb)
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
                    pbar.set_postfix(
                        loss=f"{loss_now:.4f}",
                        gnorm=f"{gnorm_now:.3f}",
                        lr=f"{lr_now:.2e}",
                    )
                    if use_wandb and opt_step % args.log_steps == 0:
                        import wandb
                        wandb.log({
                            "train/loss":  loss_now,
                            "train/gnorm": gnorm_now,
                            "train/lr":    lr_now,
                            "epoch":       epoch + (batch_idx + 1) / n_batches_per_epoch,
                        }, step=opt_step)

            epoch_loss += loss.item() * args.grad_accum

        avg_loss = epoch_loss / n_batches_per_epoch
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")

        # ── Distributed evaluation ─────────────────────────────────────────
        if val_index is not None:
            if is_ddp:
                dist.barrier()
            val_acc, val_task_acc = evaluate_ddp(
                encoder, val_index,
                device, args, rank, world_size, is_ddp,
            )
            if rank == 0:
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
