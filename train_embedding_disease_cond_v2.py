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

from utils.h2d import CudaPrefetcher
from utils.async_dataloader import AsyncDataLoader

import pandas as pd
import pyarrow.parquet as pq
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


# ── Training dataset ──────────────────────────────────────────────────────────

class PreparedDataset(Dataset):
    """Training dataset for prepare_task_data.py output.

    Loads all shards into RAM, groups samples by (task_idx, label).
    Each epoch: pos/neg pools per task are independently shuffled, then a
    batch schedule is built so that every batch contains exactly one task
    and half positives + half negatives.  Batches alternate across tasks.

    Call set_epoch(epoch) before each epoch to regenerate the schedule.

    Schema: task_idx (int16), label (int8), event_ids (list<int32>), source_row (int32)
    """

    def __init__(
        self,
        paths:             list[str],
        store:             EmbeddingStore,
        batch_size:        int,
        rank:              int = 0,
        world_size:        int = 1,
        seed:              int = 42,
        pad_to_num_events: int | None = None,
    ):
        if batch_size < 4 or batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even and ≥4, got {batch_size}")

        self.store             = store
        self.batch_size        = batch_size
        self.half              = batch_size // 2
        self.rank              = rank
        self.world_size        = world_size
        self.seed              = seed
        self.pad_to_num_events = pad_to_num_events

        # pos[task_idx] / neg[task_idx] = list of event_id arrays
        pos: dict[int, list[np.ndarray]] = defaultdict(list)
        neg: dict[int, list[np.ndarray]] = defaultdict(list)

        for path in paths:
            logger.info(f"Loading shard {path} …")
            pf = pq.ParquetFile(path)
            for rg_batch in pf.iter_batches(columns=["task_idx", "label", "event_ids"]):
                df = rg_batch.to_pandas()
                for row in df.itertuples(index=False):
                    eids = np.array(row.event_ids, dtype=np.int32)
                    (pos if row.label == 1 else neg)[int(row.task_idx)].append(eids)
                del df

        self.pos = dict(pos)
        self.neg = dict(neg)

        half = self.half
        self.valid_tasks = sorted(
            t for t in self.pos
            if len(self.pos[t]) >= half and len(self.neg.get(t, [])) >= half
        )
        if not self.valid_tasks:
            raise ValueError(
                f"No task has ≥{half} positives and ≥{half} negatives. "
                f"Reduce --batch_size or provide more data."
            )
        skipped = set(self.pos) - set(self.valid_tasks)
        if skipped:
            logger.warning(f"  Skipped tasks with insufficient data: {sorted(skipped)}")

        idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
        for t in self.valid_tasks:
            logger.info(f"  [{idx_to_name.get(t, t)}] "
                        f"{len(self.pos[t]):,} pos / {len(self.neg[t]):,} neg")

        self._n_per_task = {
            t: min(len(self.pos[t]), len(self.neg[t])) // half
            for t in self.valid_tasks
        }

        # _batches populated by set_epoch(); initialise for epoch 0
        self._batches:  list[tuple[int, int]] = []
        self._pos_perm: dict[int, list[int]]  = {}
        self._neg_perm: dict[int, list[int]]  = {}
        self.n_batches = 0
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Shuffle pos/neg pools per task and rebuild the interleaved batch schedule."""
        g    = torch.Generator()
        g.manual_seed(self.seed + epoch)
        half = self.half

        self._pos_perm = {
            t: torch.randperm(len(self.pos[t]), generator=g).tolist()
            for t in self.valid_tasks
        }
        self._neg_perm = {
            t: torch.randperm(len(self.neg[t]), generator=g).tolist()
            for t in self.valid_tasks
        }

        # Interleave (task, slot) pairs then globally shuffle
        flat = [(t, b) for t in self.valid_tasks for b in range(self._n_per_task[t])]
        perm = torch.randperm(len(flat), generator=g).tolist()
        flat = [flat[i] for i in perm]

        self._batches  = flat[self.rank :: self.world_size]
        self.n_batches = len(self._batches)

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, j: int) -> dict[str, torch.Tensor]:
        task, slot = self._batches[j]
        half       = self.half
        pos_perm   = self._pos_perm[task]
        neg_perm   = self._neg_perm[task]

        eids_list: list[np.ndarray] = []
        labels:    list[int]        = []
        for i in range(half):
            eids_list.append(self.pos[task][pos_perm[slot * half + i]])
            labels.append(1)
        for i in range(half):
            eids_list.append(self.neg[task][neg_perm[slot * half + i]])
            labels.append(0)

        return _collate_event_embs(
            eids_list, [task] * self.batch_size, labels,
            self.store.embeddings, self.pad_to_num_events,
        )


# ── Eval data index ───────────────────────────────────────────────────────────

class EvalDataIndex:
    """In-memory index for build_eval_task_data.py output.

    Schema: patient_id (int64), task_idx (int16), label (int8), event_ids (list<int32>)

    pos[task_idx] = list[(patient_id, event_ids_array)]
    neg[task_idx] = list[(patient_id, event_ids_array)]
    """

    def __init__(self, paths: list[str], store: EmbeddingStore):
        self.store = store
        pos: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        neg: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

        for path in paths:
            logger.info(f"Loading eval data {path} …")
            df = pd.read_parquet(path, columns=["patient_id", "task_idx", "label", "event_ids"])
            for row in df.itertuples(index=False):
                task_idx = int(row.task_idx)
                pid      = int(row.patient_id)
                eids     = np.array(row.event_ids, dtype=np.int32)
                (pos if int(row.label) == 1 else neg)[task_idx].append((pid, eids))
            del df

        self.pos: dict[int, list[tuple[int, np.ndarray]]] = dict(pos)
        self.neg: dict[int, list[tuple[int, np.ndarray]]] = dict(neg)
        self._idx_to_name = {v: k for k, v in TASK_2_IDX.items()}

        for task_idx in sorted(set(list(pos.keys()) + list(neg.keys()))):
            name  = self._idx_to_name.get(task_idx, str(task_idx))
            n_pos = len(pos.get(task_idx, []))
            n_neg = len(neg.get(task_idx, []))
            logger.info(f"  EvalDataIndex [{name}]: {n_pos} pos / {n_neg} neg")


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
def evaluate_ddp(
    encoder:    DiseaseAwareEHREncoder,
    eval_index: EvalDataIndex,
    device:     torch.device,
    args,
    rank:       int,
    world_size: int,
    is_ddp:     bool,
) -> tuple[float, dict[str, float]]:
    """Triplet accuracy distributed across all ranks.

    Returns (overall_acc, {task_name: acc}).

    Triplet constraints:
      - anchor and positive come from *different* patients
      - anchor and negative come from *different* patients

    Both pools are shuffled before selection so consecutive eval samples
    are spread out.
    """
    raw_encoder = encoder.module if isinstance(encoder, DDP) else encoder
    raw_encoder.eval()

    rng = random.Random(args.seed)
    idx_to_name = eval_index._idx_to_name

    # Each entry in these lists is (event_ids_array, task_idx).
    anchors_e:      list[tuple[np.ndarray, int]] = []
    positives_e:    list[tuple[np.ndarray, int]] = []
    negatives_e:    list[tuple[np.ndarray, int]] = []
    task_of_triplet: list[int]                   = []

    valid_task_idxs = sorted(
        t for t in eval_index.pos
        if len(eval_index.pos[t]) >= 2 and eval_index.neg.get(t)
    )

    for task_idx in valid_task_idxs:
        pos_entries = eval_index.pos[task_idx]   # list[(pid, eids)]
        neg_entries = eval_index.neg[task_idx]

        # Shuffle both pools so adjacent duplicates are spread out.
        pos_pool = list(range(len(pos_entries)))
        neg_pool = list(range(len(neg_entries)))
        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)

        n = min(args.n_eval_triplets_per_task, len(pos_entries))
        ni_cursor = 0
        for i in range(n):
            ai           = pos_pool[i]
            anchor_pid   = pos_entries[ai][0]
            anchor_eids  = pos_entries[ai][1]

            # Positive: find a sample from a *different* patient.
            pi = None
            for offset in range(1, len(pos_pool)):
                cand = pos_pool[(i + offset) % len(pos_pool)]
                if pos_entries[cand][0] != anchor_pid:
                    pi = cand
                    break
            if pi is None:
                continue   # all positives are from the same patient — skip

            # Negative: find a sample from a *different* patient.
            ni = None
            for offset in range(len(neg_pool)):
                cand = neg_pool[(ni_cursor + offset) % len(neg_pool)]
                if neg_entries[cand][0] != anchor_pid:
                    ni = cand
                    break
            ni_cursor += 1
            if ni is None:
                continue   # all negatives are from the same patient — skip

            anchors_e.append((anchor_eids,          task_idx))
            positives_e.append((pos_entries[pi][1], task_idx))
            negatives_e.append((neg_entries[ni][1], task_idx))
            task_of_triplet.append(task_idx)

    if not anchors_e:
        if rank == 0:
            logger.warning("No eval triplets could be built.")
        return 0.0, {}

    n_triplets    = len(anchors_e)
    my_triple_idx = list(range(rank, n_triplets, world_size))
    my_entries: list[tuple[np.ndarray, int]] = (
        [anchors_e[t]   for t in my_triple_idx] +
        [positives_e[t] for t in my_triple_idx] +
        [negatives_e[t] for t in my_triple_idx]
    )

    eval_ds = EvalBatchDataset(my_entries, eval_index.store,
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
        my_task_idxs = [task_of_triplet[t] for t in my_triple_idx]
        local_task_correct: dict[int, torch.Tensor] = {}
        local_task_total:   dict[int, torch.Tensor] = {}
        for t_idx in valid_task_idxs:
            mask = torch.tensor([t == t_idx for t in my_task_idxs], dtype=torch.bool)
            local_task_correct[t_idx] = correct[mask].sum().to(torch.long).to(device)
            local_task_total[t_idx]   = mask.sum().to(torch.long).to(device)

        local_correct = correct.sum().to(torch.long).to(device)
        local_total   = torch.tensor(n_my, dtype=torch.long, device=device)
    else:
        local_task_correct = {t: torch.tensor(0, dtype=torch.long, device=device)
                              for t in valid_task_idxs}
        local_task_total   = {t: torch.tensor(0, dtype=torch.long, device=device)
                              for t in valid_task_idxs}
        local_correct = torch.tensor(0, dtype=torch.long, device=device)
        local_total   = torch.tensor(0, dtype=torch.long, device=device)

    if is_ddp:
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total,   op=dist.ReduceOp.SUM)
        for t_idx in valid_task_idxs:
            dist.all_reduce(local_task_correct[t_idx], op=dist.ReduceOp.SUM)
            dist.all_reduce(local_task_total[t_idx],   op=dist.ReduceOp.SUM)

    overall_acc = (local_correct.float() / local_total.float()).item() \
                  if local_total.item() > 0 else 0.0

    task_acc: dict[str, float] = {}
    for t_idx in valid_task_idxs:
        n_t = local_task_total[t_idx].item()
        task_name = idx_to_name.get(t_idx, str(t_idx))
        task_acc[task_name] = (local_task_correct[t_idx].float() / n_t).item() \
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
    p.add_argument("--train_data_paths", nargs="+", default=None,
                   help="Prepared train shard parquets (output of prepare_task_data.py). "
                        "Required unless --eval_only.")
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
    if not args.eval_only and not args.train_data_paths:
        raise ValueError("--train_data_paths is required unless --eval_only is set.")

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
        if not args.eval_data_paths:
            raise ValueError("--eval_only requires --eval_data_paths.")
        eval_index = EvalDataIndex(args.eval_data_paths, store)
        val_acc, val_task_acc = evaluate_ddp(
            encoder, eval_index,
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

    # ── Training dataset ───────────────────────────────────────────────────────
    train_ds = PreparedDataset(
        paths=args.train_data_paths,
        store=store,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        pad_to_num_events=args.pad_to_num_events,
    )

    eval_index = None
    if args.eval_data_paths:
        eval_index = EvalDataIndex(args.eval_data_paths, store)

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

    # ── Sanity-check first batch ───────────────────────────────────────────────
    if rank == 0:
        first_batch = train_ds[0]
        task_ids = first_batch["task_idxs"].tolist()
        labels   = first_batch["labels"].tolist()
        n_pos    = sum(labels)
        n_neg    = len(labels) - n_pos
        unique_tasks = set(task_ids)
        idx_to_name  = {v: k for k, v in TASK_2_IDX.items()}
        assert len(unique_tasks) == 1, \
            f"First batch has {len(unique_tasks)} tasks: {unique_tasks}"
        assert n_pos == n_neg, \
            f"First batch pos/neg imbalance: {n_pos} pos / {n_neg} neg"
        logger.info(
            f"First batch check OK: task={idx_to_name.get(task_ids[0], task_ids[0])}  "
            f"pos={n_pos}  neg={n_neg}"
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
        if eval_index is not None:
            if is_ddp:
                dist.barrier()
            val_acc, val_task_acc = evaluate_ddp(
                encoder, eval_index,
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
