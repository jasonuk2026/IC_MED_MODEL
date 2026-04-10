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


def _wrap_ddp_for_stage(
    model: torch.nn.Module,
    *,
    is_ddp: bool,
    local_rank: int,
    stage: str,
) -> torch.nn.Module:
    """Wrap the model in DDP with stage-specific settings.

    Stage 1 freezes LoRA, so many parameters are intentionally unused and DDP
    must tolerate that. Stage 2 uses a fixed full graph and can use the faster
    no-unused-params reducer path.
    """
    if not is_ddp:
        return model
    if stage == "stage1":
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=True,
        )
    if stage == "stage2":
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=True,
        )
    raise ValueError(f"Unknown DDP stage: {stage}")

# ── Constants ──────────────────────────────────────────────────────────────────

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}

# Richer disease descriptions for text-query conditioning. These are used only
# on the disease/query side (for example in proj_cond_xattn), so we can make the
# query more informative without changing the patient/event encoder itself.
TASK_2_DISEASE_QUERY_TEXT = {
    "new_hypertension": (
        "Disease query: hypertension, also called high blood pressure. "
        "Relevant clinical clues include persistently elevated systolic or "
        "diastolic blood pressure, antihypertensive medications, chronic "
        "cardiovascular risk, kidney disease, and repeated outpatient blood "
        "pressure measurements above the normal range."
    ),
    "new_hyperlipidemia": (
        "Disease query: hyperlipidemia, including high cholesterol, elevated "
        "LDL, elevated triglycerides, and dyslipidemia. Relevant clues include "
        "abnormal lipid panel results, statin or other lipid-lowering therapy, "
        "atherosclerotic cardiovascular risk, and follow-up laboratory testing "
        "for cholesterol and triglycerides."
    ),
    "new_pancan": (
        "Disease query: pancreatic cancer, pancreatic adenocarcinoma, or "
        "malignancy of the pancreas. Relevant clues include pancreatic mass or "
        "neoplasm, biliary obstruction or jaundice, weight loss, abdominal or "
        "back pain, elevated tumor markers, imaging findings, oncologic "
        "evaluation, surgery, chemotherapy, and hospital care related to "
        "pancreatic malignancy."
    ),
    "new_celiac": (
        "Disease query: celiac disease, gluten-sensitive enteropathy, or immune "
        "mediated small bowel disease triggered by gluten. Relevant clues include "
        "malabsorption, chronic diarrhea, abdominal pain, nutritional deficiency, "
        "positive serology such as tissue transglutaminase antibodies, small bowel "
        "biopsy findings, and dietary management with gluten restriction."
    ),
    "new_lupus": (
        "Disease query: systemic lupus erythematosus, also called lupus or SLE. "
        "Relevant clues include autoimmune disease, rash, arthritis, nephritis, "
        "hematologic abnormalities, positive ANA or double-stranded DNA tests, "
        "complement abnormalities, immunosuppressive therapy, and multisystem "
        "inflammatory manifestations."
    ),
    "new_acutemi": (
        "Disease query: acute myocardial infarction, heart attack, STEMI, NSTEMI, "
        "or acute coronary syndrome with myocardial injury. Relevant clues include "
        "chest pain, ischemia, elevated troponin, ECG changes, coronary "
        "catheterization or revascularization, antiplatelet or anticoagulant "
        "therapy, and hospitalization for acute cardiac ischemia."
    ),
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

EVAL_QUERY_PAIR_SCHEMA = pa.schema([
    pa.field("task_idx", pa.int16()),
    pa.field("positive_eids", pa.list_(pa.int32())),
    pa.field("negative_eids", pa.list_(pa.int32())),
])


def precompute_eval_triplets(
    eval_data_paths:      list[str],
    n_triplets_per_task:  int,
    seed:                 int,
    output_path:          Path,
) -> int:
    pos: dict[int, dict[int, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
    neg: dict[int, dict[int, list[list[int]]]] = defaultdict(lambda: defaultdict(list))

    for path in eval_data_paths:
        df = pd.read_parquet(path, columns=["patient_id", "task_idx", "label", "event_ids"])
        for row in df.itertuples(index=False):
            patient_id = int(row.patient_id)
            task_idx = int(row.task_idx)
            eids     = list(row.event_ids)
            (pos if int(row.label) == 1 else neg)[task_idx][patient_id].append(eids)
        del df

    rng = random.Random(seed)
    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    records: list[dict] = []

    for task_idx in sorted(set(list(pos.keys()) + list(neg.keys()))):
        pos_by_pid = pos.get(task_idx, {})
        neg_by_pid = neg.get(task_idx, {})
        pos_pids = list(pos_by_pid.keys())
        neg_pids = list(neg_by_pid.keys())

        if len(pos_pids) < 2 or len(neg_pids) < 1:
            logger.warning(f"  [{idx_to_name.get(task_idx, task_idx)}] "
                           f"insufficient eval data for triplets — skipping")
            continue

        task_count = 0
        for _ in range(n_triplets_per_task):
            anchor_pid, positive_pid = rng.sample(pos_pids, 2)
            candidate_neg_pids = [pid for pid in neg_pids if pid not in {anchor_pid, positive_pid}]
            if not candidate_neg_pids:
                logger.warning(
                    f"  [{idx_to_name.get(task_idx, task_idx)}] insufficient distinct negative "
                    f"patients for patient-level triplets — skipping remaining triplets"
                )
                break
            negative_pid = rng.choice(candidate_neg_pids)

            anchor_eids   = rng.choice(pos_by_pid[anchor_pid])
            positive_eids = rng.choice(pos_by_pid[positive_pid])
            negative_eids = rng.choice(neg_by_pid[negative_pid])
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


def precompute_eval_query_pairs(
    eval_data_paths: list[str],
    n_pairs_per_task: int,
    seed: int,
    output_path: Path,
) -> int:
    pos: dict[int, dict[int, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
    neg: dict[int, dict[int, list[list[int]]]] = defaultdict(lambda: defaultdict(list))

    for path in eval_data_paths:
        df = pd.read_parquet(path, columns=["patient_id", "task_idx", "label", "event_ids"])
        for row in df.itertuples(index=False):
            patient_id = int(row.patient_id)
            task_idx = int(row.task_idx)
            eids = list(row.event_ids)
            (pos if int(row.label) == 1 else neg)[task_idx][patient_id].append(eids)
        del df

    rng = random.Random(seed + 17)
    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    records: list[dict] = []

    for task_idx in sorted(set(list(pos.keys()) + list(neg.keys()))):
        pos_by_pid = pos.get(task_idx, {})
        neg_by_pid = neg.get(task_idx, {})
        pos_pids = list(pos_by_pid.keys())
        neg_pids = list(neg_by_pid.keys())

        if len(pos_pids) < 1 or len(neg_pids) < 1:
            logger.warning(f"  [{idx_to_name.get(task_idx, task_idx)}] insufficient eval data for query pairs — skipping")
            continue

        task_count = 0
        for _ in range(n_pairs_per_task):
            pos_pid = rng.choice(pos_pids)
            neg_pid = rng.choice(neg_pids)
            records.append({
                "task_idx": task_idx,
                "positive_eids": rng.choice(pos_by_pid[pos_pid]),
                "negative_eids": rng.choice(neg_by_pid[neg_pid]),
            })
            task_count += 1

        logger.info(f"  [{idx_to_name.get(task_idx, task_idx)}] {task_count} eval query pairs")

    rng.shuffle(records)
    table = pa.table(
        {
            "task_idx": pa.array([r["task_idx"] for r in records], type=pa.int16()),
            "positive_eids": pa.array([r["positive_eids"] for r in records], type=pa.list_(pa.int32())),
            "negative_eids": pa.array([r["negative_eids"] for r in records], type=pa.list_(pa.int32())),
        },
        schema=EVAL_QUERY_PAIR_SCHEMA,
    )
    pq.write_table(table, str(output_path))
    logger.info(f"  Saved {len(records)} eval query pairs → {output_path}")
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
            *raw_enc.task_input_emb.parameters(),
            raw_enc.task_input_scale,
            *raw_enc.task_cond_emb.parameters(),
            *raw_enc.cond_fuse_1.parameters(),
            *raw_enc.cond_fuse_2.parameters(),
            raw_enc.task_res_scale,
            *raw_enc.film_gamma.parameters(),
            *raw_enc.film_beta.parameters(),
            *raw_enc.task_query_proj.parameters(),
            *raw_enc.disease_head_layers.parameters(),
            *raw_enc.task_cross_attn.parameters(),
            raw_enc.task_xattn_scale,
            *raw_enc.task_proto_emb.parameters(),
            *raw_enc.query_fuse.parameters(),
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


def _disease_retrieval_loss(
    encoder: torch.nn.Module,
    patient_emb: torch.Tensor,
    task_idxs: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Listwise disease-query ↔ patient retrieval loss for a single-task batch.

    Each training batch is prepared from a single task. We embed the disease text
    query for that task, score all patients in the batch against the query, and
    optimise a multi-positive InfoNCE objective where positives are label=1
    patients and negatives are label=0 patients.
    """
    raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
    feat = torch.nn.functional.normalize(patient_emb.float(), p=2, dim=-1)

    if task_idxs.numel() == 0:
        return feat.new_zeros(()), {"q_pos": float("nan"), "q_neg": float("nan")}

    # Prepared training batches are single-task; use the batch task query.
    task_id = task_idxs[0]
    query = raw_enc.encode_task_query(task_id.unsqueeze(0).to(feat.device))
    query = torch.nn.functional.normalize(query.float(), p=2, dim=-1).squeeze(0)  # (D,)

    scores = (feat @ query) / temperature  # (B,)
    pos_mask = labels == 1
    if not pos_mask.any():
        return feat.new_zeros(()), {
            "q_pos": float("nan"),
            "q_neg": scores[~pos_mask].mean().item() if (~pos_mask).any() else float("nan"),
        }

    loss = -torch.logsumexp(scores[pos_mask], dim=0) + torch.logsumexp(scores, dim=0)
    stats = {
        "q_pos": scores[pos_mask].mean().item(),
        "q_neg": scores[~pos_mask].mean().item() if (~pos_mask).any() else float("nan"),
    }
    return loss, stats


def _binary_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC-AUC from scores/labels without sklearn.

    Interpretable as the probability that a random positive receives a higher
    score than a random negative, with ties counting as 0.5.
    """
    scores = scores.float().cpu()
    labels = labels.long().cpu()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    auc = (cmp > 0).float().mean() + 0.5 * (cmp == 0).float().mean()
    return auc.item()


def _precision_recall_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> tuple[float, float]:
    scores = scores.float().cpu()
    labels = labels.long().cpu()
    n = labels.numel()
    if n == 0:
        return float("nan"), float("nan")
    k = min(k, n)
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    topk_labels = labels[topk_idx]
    hits = topk_labels.sum().item()
    total_pos = labels.sum().item()
    precision = hits / max(k, 1)
    recall = hits / max(total_pos, 1) if total_pos > 0 else float("nan")
    return precision, recall


def _forward_embeddings(
    encoder: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    encoder_mode: str,
    *,
    return_pre_emb: bool = False,
):
    """Unified embedding forward for full/proj/raw modes."""
    event_embs = batch["event_embs"].to(device)
    event_mask = batch["event_mask"].to(device)
    task_idxs  = batch["task_idxs"].to(device)

    if encoder_mode == "full":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "proj":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            bypass_qwen=True,
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "proj_cond":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            bypass_qwen=True,
            condition_on_task=True,
            condition_mode="concat",
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "proj_cond_token_pre":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            bypass_qwen=True,
            condition_on_task=True,
            condition_mode="token_preproj",
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "proj_cond_residual":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            bypass_qwen=True,
            condition_on_task=True,
            condition_mode="residual",
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "proj_cond_film":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            bypass_qwen=True,
            condition_on_task=True,
            condition_mode="film",
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "proj_cond_xattn":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            bypass_qwen=True,
            condition_on_task=True,
            condition_mode="xattn",
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "proj_cond_xattn_pool":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            bypass_qwen=True,
            condition_on_task=True,
            condition_mode="xattn_pool",
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "proj_cond_query_proto":
        return encoder(
            event_embs,
            event_mask,
            task_idxs,
            bypass_qwen=True,
            condition_on_task=True,
            condition_mode="query_proto",
            return_pre_emb=return_pre_emb,
        )
    if encoder_mode == "raw":
        mask_f = event_mask.float().unsqueeze(-1)
        pre_emb = (event_embs.float() * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        emb = torch.nn.functional.normalize(pre_emb, p=2, dim=-1)
        if return_pre_emb:
            return emb, pre_emb
        return emb
    raise ValueError(f"Unknown encoder_mode: {encoder_mode}")


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


@torch.inference_mode()
def build_task_text_embs(args, device: torch.device, rank: int, is_ddp: bool) -> torch.Tensor:
    tasks_sorted = sorted(TASK_2_DISEASE_NAME)
    task_text_embs = torch.zeros(len(tasks_sorted), BERT_DIM, dtype=torch.float32, device=device)

    if rank == 0:
        disease_model_name = getattr(args, "disease_model_name", "michiyasunaga/BioLinkBERT-base")
        logger.info(f"Loading disease text encoder: {disease_model_name}")
        disease_tokenizer = AutoTokenizer.from_pretrained(disease_model_name, local_files_only=True)
        disease_model = AutoModel.from_pretrained(disease_model_name, local_files_only=True).to(device)
        disease_model.eval()

        texts = [TASK_2_DISEASE_QUERY_TEXT[t] for t in tasks_sorted]
        enc = disease_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        out = disease_model(**enc)
        special_ids = set(disease_tokenizer.all_special_ids)
        special = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
        for sid in special_ids:
            special |= enc["input_ids"] == sid
        pool_mask = enc["attention_mask"].bool() & ~special
        pool_mask_f = pool_mask.float().unsqueeze(-1)
        task_text_embs = (
            (out.last_hidden_state.float() * pool_mask_f).sum(dim=1)
            / pool_mask_f.sum(dim=1).clamp(min=1e-9)
        )
        logger.info("  Built BioLinkBERT disease embeddings for task queries")
        for task, text in zip(tasks_sorted, texts):
            logger.info(f"  Query text [{task}]: {text}")

    if is_ddp:
        dist.broadcast(task_text_embs, src=0)
    return task_text_embs.cpu()


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


def build_encoder(qwen_model, tokenizer, args, device: torch.device, rank: int, is_ddp: bool) -> DiseaseAwareEHREncoder:
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
    task_text_embs = build_task_text_embs(
        args,
        device,
        rank,
        is_ddp,
    )

    encoder = DiseaseAwareEHREncoder(
        qwen_model       = qwen_model,
        bert_dim         = BERT_DIM,
        qwen_dim         = qwen_dim,
        task_prefix_ids  = task_prefix_ids,
        task_prefix_mask = task_prefix_mask,
        middle_ids       = middle_ids,
        task_text_embs   = task_text_embs,
        shallow_encoder_type = args.shallow_encoder_type,
        shallow_num_layers = args.shallow_num_layers,
        shallow_num_heads = args.shallow_num_heads,
        shallow_intermediate_size = args.shallow_intermediate_size,
        disease_encoder_type = args.disease_encoder_type,
        disease_head_layers = args.disease_head_layers,
        disease_head_intermediate_size = args.disease_head_intermediate_size,
        dtype            = dtype,
    )
    logger.info(
        f"  Prompt template: [{repr(PROMPT_PREFIX)}<disease_name>]{repr(PROMPT_MIDDLE)}<events>"
    )
    if args.shallow_encoder_type == "transformer":
        logger.info(f"  Event encoder: direct event embeddings → shallow Transformer ({BERT_DIM}-dim)  dtype={dtype}")
        logger.info(f"  Qwen-only projection head: RMSNorm → Linear({BERT_DIM}→{BERT_DIM}) → GELU → Linear({BERT_DIM}→{qwen_dim})")
        logger.info(
            f"  Shallow Transformer: layers={args.shallow_num_layers}  heads={args.shallow_num_heads}  "
            f"intermediate={args.shallow_intermediate_size or (BERT_DIM * 4)}  rotary=enabled  bidirectional_attention=True"
        )
    elif args.shallow_encoder_type == "mlp":
        logger.info(f"  Event encoder: direct event embeddings → stacked residual Qwen-style MLP blocks ({BERT_DIM}-dim)  dtype={dtype}")
        logger.info(f"  Qwen-only projection head: RMSNorm → Linear({BERT_DIM}→{BERT_DIM}) → GELU → Linear({BERT_DIM}→{qwen_dim})")
        logger.info(
            f"  Shallow MLP: layers={args.shallow_num_layers}  "
            f"intermediate={args.shallow_intermediate_size or (BERT_DIM * 4)}  residual=enabled"
        )
    else:
        logger.info(
            f"  Event encoder: RMSNorm → Linear({BERT_DIM}→{BERT_DIM}) → GELU → Linear({BERT_DIM}→{qwen_dim})  dtype={dtype}"
        )
    if args.disease_encoder_type == "shared_backbone":
        logger.info("  Disease encoder: shared shallow backbone (siamese) over disease embeddings")
    else:
        logger.info(
            f"  Disease head: base linear projection + {args.disease_head_layers} residual MLP block(s)  "
            f"intermediate={args.disease_head_intermediate_size or 'default'}"
        )
    logger.info(f"  Disease encoder type: {args.disease_encoder_type}")

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
    eval_query_pair_path: Path | None,
    eval_data_paths: list[str] | None,
    store:             EmbeddingStore,
    device:            torch.device,
    args,
) -> tuple[float, dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
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
        return 0.0, {}, {}, {}

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
    all_pre_embs: list[torch.Tensor] = []
    all_proj_embs: list[torch.Tensor] = []
    all_proj_pre_embs: list[torch.Tensor] = []
    all_raw_pre_embs: list[torch.Tensor] = []
    for batch in tqdm(eval_dl, desc="Evaluating", dynamic_ncols=True):
        event_embs = batch["event_embs"].to(device)
        event_mask = batch["event_mask"].to(device)
        task_idxs  = batch["task_idxs"].to(device)

        mode_batch = {
            "event_embs": event_embs,
            "event_mask": event_mask,
            "task_idxs":  task_idxs,
        }
        emb, pre_emb = _forward_embeddings(
            raw_encoder, mode_batch, device, args.encoder_mode, return_pre_emb=True
        )
        proj_emb, proj_pre_emb = _forward_embeddings(
            raw_encoder, mode_batch, device, "proj", return_pre_emb=True
        )
        mask_f = event_mask.float().unsqueeze(-1)
        raw_pre_emb = (event_embs.float() * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        all_embs.append(emb.cpu())
        all_pre_embs.append(pre_emb.cpu())
        all_proj_embs.append(proj_emb.cpu())
        all_proj_pre_embs.append(proj_pre_emb.cpu())
        all_raw_pre_embs.append(raw_pre_emb.cpu())

    embs      = torch.cat(all_embs, dim=0)
    pre_embs  = torch.cat(all_pre_embs, dim=0)
    proj_embs = torch.cat(all_proj_embs, dim=0)
    proj_pre_embs = torch.cat(all_proj_pre_embs, dim=0)
    raw_pre_embs = torch.cat(all_raw_pre_embs, dim=0)

    def _triplet_correct(
        x: torch.Tensor,
        metric: str,
    ) -> torch.Tensor:
        anchors   = x[0::3]
        positives = x[1::3]
        negatives = x[2::3]
        if metric == "l2":
            d_ap = (anchors - positives).norm(dim=1)
            d_an = (anchors - negatives).norm(dim=1)
            return d_ap < d_an
        if metric == "cosine":
            sim_ap = torch.nn.functional.cosine_similarity(anchors, positives, dim=1)
            sim_an = torch.nn.functional.cosine_similarity(anchors, negatives, dim=1)
            return sim_ap > sim_an
        raise ValueError(f"Unknown metric: {metric}")

    correct_main     = _triplet_correct(embs, "l2")
    correct_pre_cos  = _triplet_correct(pre_embs, "cosine")
    correct_pre_l2   = _triplet_correct(pre_embs, "l2")
    correct_proj_cos = _triplet_correct(proj_pre_embs, "cosine")
    correct_proj_l2  = _triplet_correct(proj_pre_embs, "l2")
    correct_raw_cos  = _triplet_correct(raw_pre_embs, "cosine")
    correct_raw_l2   = _triplet_correct(raw_pre_embs, "l2")

    task_idxs_arr = df["task_idx"].values
    idx_to_name   = {v: k for k, v in TASK_2_IDX.items()}
    task_acc: dict[str, float] = {}
    extra_overall = {
        "pre_cosine_triplet_acc": correct_pre_cos.float().mean().item(),
        "pre_l2_triplet_acc":     correct_pre_l2.float().mean().item(),
        "proj_cosine_triplet_acc": correct_proj_cos.float().mean().item(),
        "proj_l2_triplet_acc":     correct_proj_l2.float().mean().item(),
        "raw_cosine_triplet_acc":  correct_raw_cos.float().mean().item(),
        "raw_l2_triplet_acc":      correct_raw_l2.float().mean().item(),
    }
    extra_task: dict[str, dict[str, float]] = {}

    for t_idx in sorted(set(task_idxs_arr.tolist())):
        mask = task_idxs_arr == t_idx
        n_t  = int(mask.sum())
        if n_t > 0:
            torch_mask = torch.from_numpy(mask)
            task_name = idx_to_name.get(t_idx, str(t_idx))
            acc = correct_main[torch_mask].float().mean().item()
            task_acc[task_name] = acc
            extra_task[task_name] = {
                "pre_cosine_triplet_acc": correct_pre_cos[torch_mask].float().mean().item(),
                "pre_l2_triplet_acc":     correct_pre_l2[torch_mask].float().mean().item(),
                "proj_cosine_triplet_acc": correct_proj_cos[torch_mask].float().mean().item(),
                "proj_l2_triplet_acc":     correct_proj_l2[torch_mask].float().mean().item(),
                "raw_cosine_triplet_acc":  correct_raw_cos[torch_mask].float().mean().item(),
                "raw_l2_triplet_acc":      correct_raw_l2[torch_mask].float().mean().item(),
            }

    if eval_query_pair_path is not None and eval_query_pair_path.exists():
        qdf = pd.read_parquet(
            str(eval_query_pair_path),
            columns=["task_idx", "positive_eids", "negative_eids"],
        )
        q_entries: list[tuple[np.ndarray, int]] = []
        for row in qdf.itertuples(index=False):
            t = int(row.task_idx)
            q_entries.append((np.array(row.positive_eids, dtype=np.int32), t))
            q_entries.append((np.array(row.negative_eids, dtype=np.int32), t))

        q_ds = EvalBatchDataset(q_entries, store, args.eval_batch_size, args.pad_to_num_events)
        q_dl = DataLoader(q_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)

        q_pre_embs: list[torch.Tensor] = []
        q_task_idxs: list[torch.Tensor] = []
        for batch in q_dl:
            _, pre_emb = _forward_embeddings(
                raw_encoder, batch, device, args.encoder_mode, return_pre_emb=True
            )
            q_pre_embs.append(pre_emb.cpu())
            q_task_idxs.append(batch["task_idxs"].cpu())

        pair_pre = torch.cat(q_pre_embs, dim=0)          # (2M, D)
        pair_task_idxs = torch.cat(q_task_idxs, dim=0)   # (2M,)
        pos_pre = pair_pre[0::2]
        neg_pre = pair_pre[1::2]
        pair_tasks = pair_task_idxs[0::2]

        all_task_idx = torch.arange(raw_encoder.task_text_embs.size(0), device=device, dtype=torch.long)
        raw_query = raw_encoder.encode_task_query(all_task_idx).float().cpu()
        q_vecs = raw_query[pair_tasks]

        sim_pos = torch.nn.functional.cosine_similarity(q_vecs, pos_pre, dim=1)
        sim_neg = torch.nn.functional.cosine_similarity(q_vecs, neg_pre, dim=1)
        q_cos_correct = sim_pos > sim_neg

        d_pos = (q_vecs - pos_pre).norm(dim=1)
        d_neg = (q_vecs - neg_pre).norm(dim=1)
        q_l2_correct = d_pos < d_neg

        extra_overall["query_cosine_pair_acc"] = q_cos_correct.float().mean().item()
        extra_overall["query_l2_pair_acc"] = q_l2_correct.float().mean().item()

        for t_idx in sorted(set(pair_tasks.tolist())):
            mask = pair_tasks == t_idx
            task_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
            extra_task.setdefault(task_name, {})
            extra_task[task_name]["query_cosine_pair_acc"] = q_cos_correct[mask].float().mean().item()
            extra_task[task_name]["query_l2_pair_acc"] = q_l2_correct[mask].float().mean().item()

    if eval_data_paths:
        val_rows: list[tuple[np.ndarray, int, int]] = []
        for path in eval_data_paths:
            vdf = pd.read_parquet(path, columns=["task_idx", "label", "event_ids"])
            for row in vdf.itertuples(index=False):
                val_rows.append((
                    np.array(row.event_ids, dtype=np.int32),
                    int(row.task_idx),
                    int(row.label),
                ))

        if val_rows:
            val_entries = [(eids, task_idx) for eids, task_idx, _ in val_rows]
            val_labels = torch.tensor([label for _, _, label in val_rows], dtype=torch.long)
            val_task_idxs = torch.tensor([task_idx for _, task_idx, _ in val_rows], dtype=torch.long)
            val_ds = EvalBatchDataset(val_entries, store, args.eval_batch_size, args.pad_to_num_events)
            val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)

            all_val_pre: list[torch.Tensor] = []
            for batch in val_dl:
                _, pre_emb = _forward_embeddings(
                    raw_encoder, batch, device, args.encoder_mode, return_pre_emb=True
                )
                all_val_pre.append(pre_emb.cpu())
            val_pre_embs = torch.cat(all_val_pre, dim=0)

            all_task_idx = torch.arange(raw_encoder.task_text_embs.size(0), device=device, dtype=torch.long)
            query_bank = raw_encoder.encode_task_query(all_task_idx).float().cpu()
            query_vecs = query_bank[val_task_idxs]
            query_scores = torch.nn.functional.cosine_similarity(query_vecs, val_pre_embs, dim=1)

            extra_overall["query_auc"] = _binary_roc_auc(query_scores, val_labels)
            extra_overall["query_num_samples"] = float(val_labels.numel())
            extra_overall["query_num_pos"] = float((val_labels == 1).sum().item())
            extra_overall["query_num_neg"] = float((val_labels == 0).sum().item())
            p10, r10 = _precision_recall_at_k(query_scores, val_labels, 10)
            p50, r50 = _precision_recall_at_k(query_scores, val_labels, 50)
            extra_overall["query_precision_at_10"] = p10
            extra_overall["query_recall_at_10"] = r10
            extra_overall["query_precision_at_50"] = p50
            extra_overall["query_recall_at_50"] = r50
            for t_idx in sorted(set(val_task_idxs.tolist())):
                mask = val_task_idxs == t_idx
                task_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
                extra_task.setdefault(task_name, {})
                task_scores = query_scores[mask]
                task_labels = val_labels[mask]
                extra_task[task_name]["query_num_samples"] = float(task_labels.numel())
                extra_task[task_name]["query_num_pos"] = float((task_labels == 1).sum().item())
                extra_task[task_name]["query_num_neg"] = float((task_labels == 0).sum().item())
                extra_task[task_name]["query_auc"] = _binary_roc_auc(
                    task_scores, task_labels
                )
                tp10, tr10 = _precision_recall_at_k(task_scores, task_labels, 10)
                tp50, tr50 = _precision_recall_at_k(task_scores, task_labels, 50)
                extra_task[task_name]["query_precision_at_10"] = tp10
                extra_task[task_name]["query_recall_at_10"] = tr10
                extra_task[task_name]["query_precision_at_50"] = tp50
                extra_task[task_name]["query_recall_at_50"] = tr50

    overall_acc = correct_main.float().mean().item()
    raw_encoder.train()
    return overall_acc, task_acc, extra_overall, extra_task


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
    p.add_argument("--train_data_epochs", nargs="+", type=int, default=None,
                   help="Optional subset of prepared train data epochs to use, e.g. "
                        "--train_data_epochs 3 or --train_data_epochs 1 3 5. "
                        "Values refer to train_prepared_XXX.parquet indices.")
    p.add_argument("--tasks", nargs="+", default=list(sorted(TASK_2_DISEASE_NAME.keys())))
    p.add_argument("--eval_data_paths",  nargs="+", default=None)

    # BioLinkBERT embeddings
    p.add_argument("--bert_embeddings", required=True)

    # Qwen model
    p.add_argument("--model_name",   default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--shallow_encoder_type", choices=["simple", "mlp", "transformer"], default="transformer",
                   help="Shallow event encoder used before optional Qwen processing. "
                        "'simple' reproduces the original RMSNorm + projection + mean-pool path; "
                        "'mlp' uses stacked Qwen-style residual MLP blocks over events; "
                        "'transformer' uses the newer bidirectional Transformer encoder.")
    p.add_argument("--shallow_num_layers", type=int, default=2,
                   help="Number of lightweight bidirectional Transformer layers over projected events.")
    p.add_argument("--shallow_num_heads", type=int, default=4,
                   help="Number of attention heads in the lightweight event Transformer.")
    p.add_argument("--shallow_intermediate_size", type=int, default=None,
                   help="Intermediate size of the lightweight Transformer MLP. Defaults to 4 * qwen_dim.")
    p.add_argument("--disease_head_layers", type=int, default=0,
                   help="Number of residual MLP blocks on top of disease text embeddings before retrieval/query use.")
    p.add_argument("--disease_head_intermediate_size", type=int, default=None,
                   help="Intermediate size of the disease head MLP blocks. Defaults to 4 * query dim.")
    p.add_argument("--disease_encoder_type", choices=["query_head", "shared_backbone"], default="query_head",
                   help="How to encode disease text embeddings for retrieval. "
                        "'query_head' uses a separate learnable disease-side head; "
                        "'shared_backbone' sends disease embeddings through the same shallow backbone as patients.")
    p.add_argument("--encoder_mode", choices=["full", "proj", "proj_cond", "proj_cond_token_pre", "proj_cond_residual", "proj_cond_film", "proj_cond_xattn", "proj_cond_xattn_pool", "proj_cond_query_proto", "raw"], default="full",
                   help="Embedding path to optimise/evaluate. "
                        "'full' = proj + Qwen, 'proj' = bert_proj mean-pool, "
                        "'proj_cond' = bert_proj mean-pool + lightweight task conditioning, "
                        "'proj_cond_token_pre' = add disease type embedding to each event before bert_proj, "
                        "'proj_cond_residual' = proj + residual task embedding, "
                        "'proj_cond_film' = proj + FiLM task conditioning, "
                        "'proj_cond_xattn' = BioLinkBERT disease-query cross-attention over event projections, "
                        "'proj_cond_xattn_pool' = use disease-query cross-attention pooled vector as output, "
                        "'proj_cond_query_proto' = fuse disease text query with a learned disease prototype before attention pooling, "
                        "'raw' = mean-pool original event embeddings (eval-only).")
    p.add_argument("--disease_model_name", default="michiyasunaga/BioLinkBERT-base",
                   help="BioLinkBERT model used to encode disease text queries for cross-attention.")
    p.add_argument("--train_objective", choices=["patient_metric", "disease_retrieval"], default="patient_metric",
                   help="Training objective. 'patient_metric' keeps the current patient-patient metric learning "
                        "objective; 'disease_retrieval' aligns disease text queries with patient embeddings.")
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
                   help="Deprecated. Stage 1 now runs over the full selected data epoch whenever "
                        "this value is > 0; use 0 to skip Stage 1.")
    p.add_argument("--legacy_stage1_only", action="store_true",
                   help="For shallow encoder modes, run only the legacy Stage 1 optimizer/scheduler "
                        "over the selected training data epoch(s), then evaluate and stop. "
                        "Useful for reproducing older retrieval results.")
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
    p.add_argument("--stage2_triplet_weight", type=float, default=0.1,
                   help="Auxiliary weight for BatchHardTripletLoss on normalised embeddings "
                        "during Stage 2.")
    p.add_argument("--retrieval_temperature", type=float, default=0.1,
                   help="Temperature for disease-query retrieval loss.")
    p.add_argument("--retrieval_supcon_weight", type=float, default=0.2,
                   help="Auxiliary weight for patient-side SupCon when train_objective=disease_retrieval.")

    # Eval
    p.add_argument("--n_eval_triplets_per_task", type=int,   default=256)
    p.add_argument("--eval_batch_size",          type=int,   default=32)
    p.add_argument("--pad_to_num_events",        type=int,   default=None)
    p.add_argument("--stage1_eval_steps",        type=int,   default=0,
                   help="If > 0, run eval every N Stage 1 optimizer steps.")
    p.add_argument("--eval_steps",               type=int,   default=0,
                   help="If > 0, run eval every N Stage 2 optimizer steps.")

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
    if args.encoder_mode == "raw" and not args.eval_only:
        raise ValueError("--encoder_mode raw is eval-only; use --eval_only or choose full/proj.")
    uses_qwen_path = args.encoder_mode == "full"

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
        encoder = build_encoder(qwen_lora, tokenizer, args, device, rank, is_ddp)
        extra = torch.load(Path(args.checkpoint) / "extra_modules.pt", map_location="cpu")
        saved_shallow_type = extra.get("shallow_encoder_type", "simple")
        saved_disease_encoder_type = extra.get("disease_encoder_type", "query_head")
        if saved_shallow_type != args.shallow_encoder_type:
            logger.warning(
                f"Checkpoint shallow_encoder_type={saved_shallow_type} but current args specify "
                f"{args.shallow_encoder_type}; loading may fail or behave unexpectedly."
            )
        if saved_disease_encoder_type != args.disease_encoder_type:
            logger.warning(
                f"Checkpoint disease_encoder_type={saved_disease_encoder_type} but current args specify "
                f"{args.disease_encoder_type}; loading may fail or behave unexpectedly."
            )
        encoder.bert_proj_1.load_state_dict(extra["bert_proj_1"])
        encoder.bert_proj_2.load_state_dict(extra["bert_proj_2"])
        if "input_norm" in extra:
            encoder.input_norm.load_state_dict(extra["input_norm"])
        if "shallow_layers" in extra:
            encoder.shallow_layers.load_state_dict(extra["shallow_layers"])
        if "task_input_emb" in extra:
            encoder.task_input_emb.load_state_dict(extra["task_input_emb"])
        if "task_input_scale" in extra:
            encoder.task_input_scale.data.copy_(extra["task_input_scale"].to(encoder.task_input_scale.dtype))
        if "task_cond_emb" in extra:
            encoder.task_cond_emb.load_state_dict(extra["task_cond_emb"])
        if "cond_fuse_1" in extra:
            encoder.cond_fuse_1.load_state_dict(extra["cond_fuse_1"])
        if "cond_fuse_2" in extra:
            encoder.cond_fuse_2.load_state_dict(extra["cond_fuse_2"])
        if "task_res_scale" in extra:
            encoder.task_res_scale.data.copy_(extra["task_res_scale"].to(encoder.task_res_scale.dtype))
        if "film_gamma" in extra:
            encoder.film_gamma.load_state_dict(extra["film_gamma"])
        if "film_beta" in extra:
            encoder.film_beta.load_state_dict(extra["film_beta"])
        if "task_query_proj" in extra:
            encoder.task_query_proj.load_state_dict(extra["task_query_proj"])
        if "disease_head_layers" in extra:
            encoder.disease_head_layers.load_state_dict(extra["disease_head_layers"])
        if "task_cross_attn" in extra:
            encoder.task_cross_attn.load_state_dict(extra["task_cross_attn"])
        if "task_xattn_scale" in extra:
            encoder.task_xattn_scale.data.copy_(extra["task_xattn_scale"].to(encoder.task_xattn_scale.dtype))
        if "task_proto_emb" in extra:
            encoder.task_proto_emb.load_state_dict(extra["task_proto_emb"])
        if "query_fuse" in extra:
            encoder.query_fuse.load_state_dict(extra["query_fuse"])
    else:
        qwen_lora = setup_lora(qwen_model, args)
        encoder = build_encoder(qwen_lora, tokenizer, args, device, rank, is_ddp)

    encoder = encoder.to(device)

    if args.compile:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        encoder = torch.compile(encoder)

    if not uses_qwen_path and not args.eval_only:
        n_frozen = _set_lora_trainable(encoder, False)
        if rank == 0 and n_frozen > 0:
            logger.info(
                f"encoder_mode={args.encoder_mode}: keeping {n_frozen} LoRA parameter tensors frozen"
            )

    if is_ddp and not args.eval_only:
        initial_stage = "stage1" if args.stage1_steps > 0 else "stage2"
        encoder = _wrap_ddp_for_stage(
            encoder, is_ddp=True, local_rank=local_rank, stage=initial_stage
        )

    # ── Pre-compute eval triplets (rank 0 only) ───────────────────────────────
    eval_triplet_path: Path | None = None
    eval_query_pair_path: Path | None = None
    if args.eval_data_paths and rank == 0:
        eval_triplet_path = run_dir / "eval_triplets.parquet"
        eval_query_pair_path = run_dir / "eval_query_pairs.parquet"
        logger.info("Pre-computing eval triplets …")
        precompute_eval_triplets(
            args.eval_data_paths,
            args.n_eval_triplets_per_task,
            args.seed,
            eval_triplet_path,
        )
        logger.info("Pre-computing eval disease-query pairs …")
        precompute_eval_query_pairs(
            args.eval_data_paths,
            args.n_eval_triplets_per_task,
            args.seed,
            eval_query_pair_path,
        )

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        if eval_triplet_path is None:
            raise ValueError("--eval_only requires --eval_data_paths.")
        val_acc, val_task_acc, extra_overall, extra_task = evaluate_rank0(
            encoder, eval_triplet_path, eval_query_pair_path, args.eval_data_paths, store, device, args
        )
        if rank == 0:
            logger.info(f"Eval triplet accuracy: {val_acc:.4f}")
            logger.info(f"  [pre_emb cosine] triplet accuracy: {extra_overall['pre_cosine_triplet_acc']:.4f}")
            logger.info(f"  [pre_emb l2] triplet accuracy: {extra_overall['pre_l2_triplet_acc']:.4f}")
            logger.info(f"  [bert_proj cosine] triplet accuracy: {extra_overall['proj_cosine_triplet_acc']:.4f}")
            logger.info(f"  [bert_proj l2] triplet accuracy: {extra_overall['proj_l2_triplet_acc']:.4f}")
            logger.info(f"  [raw_event cosine] triplet accuracy: {extra_overall['raw_cosine_triplet_acc']:.4f}")
            logger.info(f"  [raw_event l2] triplet accuracy: {extra_overall['raw_l2_triplet_acc']:.4f}")
            if "query_cosine_pair_acc" in extra_overall:
                logger.info(f"  [disease_query cosine] pair accuracy: {extra_overall['query_cosine_pair_acc']:.4f}")
                logger.info(f"  [disease_query l2] pair accuracy: {extra_overall['query_l2_pair_acc']:.4f}")
            if "query_auc" in extra_overall:
                logger.info(f"  [disease_query auc] {extra_overall['query_auc']:.4f}")
                logger.info(
                    f"  [disease_query counts] "
                    f"n={int(extra_overall.get('query_num_samples', float('nan')))}  "
                    f"pos={int(extra_overall.get('query_num_pos', float('nan')))}  "
                    f"neg={int(extra_overall.get('query_num_neg', float('nan')))}"
                )
                logger.info(
                    f"  [disease_query p@10/r@10] "
                    f"{extra_overall.get('query_precision_at_10', float('nan')):.4f}/"
                    f"{extra_overall.get('query_recall_at_10', float('nan')):.4f}"
                )
                logger.info(
                    f"  [disease_query p@50/r@50] "
                    f"{extra_overall.get('query_precision_at_50', float('nan')):.4f}/"
                    f"{extra_overall.get('query_recall_at_50', float('nan')):.4f}"
                )
            for task, acc in sorted(val_task_acc.items()):
                logger.info(f"  {task}: {acc:.4f}")
                logger.info(
                    f"    pre_cos={extra_task[task]['pre_cosine_triplet_acc']:.4f}  "
                    f"pre_l2={extra_task[task]['pre_l2_triplet_acc']:.4f}  "
                    f"proj_cos={extra_task[task]['proj_cosine_triplet_acc']:.4f}  "
                    f"proj_l2={extra_task[task]['proj_l2_triplet_acc']:.4f}  "
                    f"raw_cos={extra_task[task]['raw_cosine_triplet_acc']:.4f}  "
                    f"raw_l2={extra_task[task]['raw_l2_triplet_acc']:.4f}"
                )
                if "query_cosine_pair_acc" in extra_task[task]:
                    logger.info(
                        f"    query_cos={extra_task[task]['query_cosine_pair_acc']:.4f}  "
                        f"query_l2={extra_task[task]['query_l2_pair_acc']:.4f}"
                    )
                if "query_auc" in extra_task[task]:
                    logger.info(
                        f"    query_auc={extra_task[task]['query_auc']:.4f}  "
                        f"n={int(extra_task[task].get('query_num_samples', float('nan')))}  "
                        f"pos={int(extra_task[task].get('query_num_pos', float('nan')))}  "
                        f"neg={int(extra_task[task].get('query_num_neg', float('nan')))}  "
                        f"p@10={extra_task[task].get('query_precision_at_10', float('nan')):.4f}  "
                        f"r@10={extra_task[task].get('query_recall_at_10', float('nan')):.4f}  "
                        f"p@50={extra_task[task].get('query_precision_at_50', float('nan')):.4f}  "
                        f"r@50={extra_task[task].get('query_recall_at_50', float('nan')):.4f}"
                    )
        if use_wandb and wandb_run:
            import wandb
            log_dict = {
                "eval/triplet_acc":             val_acc,
                "eval/pre_cosine_triplet_acc":  extra_overall["pre_cosine_triplet_acc"],
                "eval/pre_l2_triplet_acc":      extra_overall["pre_l2_triplet_acc"],
                "eval/proj_cosine_triplet_acc": extra_overall["proj_cosine_triplet_acc"],
                "eval/proj_l2_triplet_acc":     extra_overall["proj_l2_triplet_acc"],
                "eval/raw_cosine_triplet_acc":  extra_overall["raw_cosine_triplet_acc"],
                "eval/raw_l2_triplet_acc":      extra_overall["raw_l2_triplet_acc"],
            }
            if "query_cosine_pair_acc" in extra_overall:
                log_dict["eval/query_cosine_pair_acc"] = extra_overall["query_cosine_pair_acc"]
                log_dict["eval/query_l2_pair_acc"] = extra_overall["query_l2_pair_acc"]
            if "query_auc" in extra_overall:
                log_dict["eval/query_auc"] = extra_overall["query_auc"]
                log_dict["eval/query_precision_at_10"] = extra_overall.get("query_precision_at_10", float("nan"))
                log_dict["eval/query_recall_at_10"] = extra_overall.get("query_recall_at_10", float("nan"))
                log_dict["eval/query_precision_at_50"] = extra_overall.get("query_precision_at_50", float("nan"))
                log_dict["eval/query_recall_at_50"] = extra_overall.get("query_recall_at_50", float("nan"))
            for task, acc in val_task_acc.items():
                log_dict[f"eval/{task}/triplet_acc"] = acc
                log_dict[f"eval/{task}/pre_cosine_triplet_acc"] = extra_task[task]["pre_cosine_triplet_acc"]
                log_dict[f"eval/{task}/pre_l2_triplet_acc"] = extra_task[task]["pre_l2_triplet_acc"]
                log_dict[f"eval/{task}/proj_cosine_triplet_acc"] = extra_task[task]["proj_cosine_triplet_acc"]
                log_dict[f"eval/{task}/proj_l2_triplet_acc"] = extra_task[task]["proj_l2_triplet_acc"]
                log_dict[f"eval/{task}/raw_cosine_triplet_acc"] = extra_task[task]["raw_cosine_triplet_acc"]
                log_dict[f"eval/{task}/raw_l2_triplet_acc"] = extra_task[task]["raw_l2_triplet_acc"]
                if "query_cosine_pair_acc" in extra_task[task]:
                    log_dict[f"eval/{task}/query_cosine_pair_acc"] = extra_task[task]["query_cosine_pair_acc"]
                    log_dict[f"eval/{task}/query_l2_pair_acc"] = extra_task[task]["query_l2_pair_acc"]
                if "query_auc" in extra_task[task]:
                    log_dict[f"eval/{task}/query_auc"] = extra_task[task]["query_auc"]
                    log_dict[f"eval/{task}/query_precision_at_10"] = extra_task[task].get("query_precision_at_10", float("nan"))
                    log_dict[f"eval/{task}/query_recall_at_10"] = extra_task[task].get("query_recall_at_10", float("nan"))
                    log_dict[f"eval/{task}/query_precision_at_50"] = extra_task[task].get("query_precision_at_50", float("nan"))
                    log_dict[f"eval/{task}/query_recall_at_50"] = extra_task[task].get("query_recall_at_50", float("nan"))
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
    available_data_epochs = sorted(
        int(p.stem.split("_")[-1]) for p in data_epoch_files
    )
    n_data_epochs    = len(available_data_epochs)
    if n_data_epochs == 0:
        raise ValueError(f"No train_prepared_*.parquet found in {task_dir}")
    if rank == 0:
        logger.info(
            f"Found {n_data_epochs} data epoch(s) (task dir: {task_dir}): "
            f"{available_data_epochs}"
        )

    if args.train_data_epochs is not None:
        requested_epochs = list(dict.fromkeys(args.train_data_epochs))
        missing_epochs = [ep for ep in requested_epochs if ep not in available_data_epochs]
        if missing_epochs:
            raise ValueError(
                f"Requested --train_data_epochs {missing_epochs}, but available prepared "
                f"epochs are {available_data_epochs}"
            )
        selected_data_epochs = requested_epochs
    else:
        selected_data_epochs = available_data_epochs

    if rank == 0:
        logger.info(f"Selected train data epoch(s): {selected_data_epochs}")

    stage1_data_epoch = selected_data_epochs[-1]

    if uses_qwen_path:
        train_schedule = [
            selected_data_epochs[epoch % len(selected_data_epochs)]
            for epoch in range(args.epochs)
        ]
        if rank == 0:
            logger.info(
                f"Qwen path: using staged training with {args.epochs} epoch(s) "
                f"over selected prepared data epoch(s) {selected_data_epochs}"
            )
    else:
        train_schedule = list(selected_data_epochs)
        if rank == 0:
            logger.info(
                f"Shallow encoder path: using single-phase training over all "
                f"selected prepared data epoch(s) {selected_data_epochs} exactly once"
            )
            if args.legacy_stage1_only:
                logger.info(
                    "  legacy_stage1_only enabled: will run only the old Stage 1 optimizer/scheduler "
                    "over the final selected prepared epoch and then stop"
                )
            if args.stage1_steps > 0 and not args.legacy_stage1_only:
                logger.info("  Ignoring --stage1_steps for shallow encoder mode")
            if args.epochs != len(selected_data_epochs):
                logger.info(
                    f"  Ignoring --epochs={args.epochs}; effective training passes = {len(selected_data_epochs)}"
                )

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
    if args.stage1_steps > 0 and (uses_qwen_path or args.legacy_stage1_only):
        if rank == 0:
            logger.info(
                f"Stage 1: full data epoch {stage1_data_epoch} — "
                f"{'LoRA frozen, full frozen-Qwen forward' if uses_qwen_path else 'shallow encoder mode'}"
                f", lr_proj={args.lr_proj}"
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

        # Linear warmup for Stage 1 over the full selected data epoch.
        stage1_ds, stage1_dl = _make_loader(data_epoch=stage1_data_epoch, training_epoch=0)
        s1_total_steps = len(stage1_ds)
        s1_warmup = max(1, int(s1_total_steps * args.warmup_ratio))
        s1_sched  = torch.optim.lr_scheduler.LinearLR(
            stage1_opt, start_factor=1e-3, end_factor=1.0, total_iters=s1_warmup
        )
        if rank == 0:
            logger.info(
                f"  Stage 1 total steps: {s1_total_steps}  "
                f"warmup: {s1_warmup} steps (ratio={args.warmup_ratio})"
            )
        encoder.train()
        stage1_opt.zero_grad()
        s1_step = 0

        pbar_s1 = tqdm(
            stage1_dl,
            desc="Stage 1",
            disable=(rank != 0),
            dynamic_ncols=True,
            total=len(stage1_ds),
        )

        for batch in pbar_s1:
            labels_t = batch["labels"]
            task_idxs_t = batch["task_idxs"].to(device)

            emb, pre_emb = _forward_embeddings(
                encoder, batch, device, args.encoder_mode, return_pre_emb=True
            )
            if args.train_objective == "disease_retrieval":
                retrieval_loss, retrieval_stats = _disease_retrieval_loss(
                    encoder,
                    pre_emb,
                    task_idxs_t,
                    labels_t.to(device),
                    args.retrieval_temperature,
                )
                supcon_aux = _supervised_contrastive_loss(
                    pre_emb,
                    labels_t.to(device),
                    args.stage1_temperature,
                )
                main_loss = retrieval_loss + args.retrieval_supcon_weight * supcon_aux
                main_loss_name = "lret"
                loss_supcon_now = None
            else:
                main_loss = _supervised_contrastive_loss(
                    pre_emb,
                    labels_t.to(device),
                    args.stage1_temperature,
                )
                retrieval_stats = {"q_pos": float("nan"), "q_neg": float("nan")}
                main_loss_name = "lsup"

            # Variance regularisation is applied to the pre-normalisation pooled
            # embedding from the same full-Qwen path used in stage 2.
            if args.var_reg_weight > 0.0:
                std      = torch.sqrt(pre_emb.var(dim=0) + 1e-4)   # (D,)
                var_loss = torch.relu(1.0 - std).mean()
                loss     = main_loss + args.var_reg_weight * var_loss
            else:
                loss = main_loss
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
                    if args.train_objective == "disease_retrieval":
                        loss_retrieval_now = retrieval_loss.item()
                        loss_supcon_aux_now = supcon_aux.item()
                        loss_main_now = main_loss.item()
                    else:
                        loss_supcon_now = main_loss.item()
                        loss_main_now = loss_supcon_now
                    var_loss_now = var_loss.item() if args.var_reg_weight > 0.0 else 0.0

                postfix = {
                    "loss": f"{loss_now:.4f}",
                    main_loss_name: f"{loss_main_now:.4f}",
                    "cdist": f"{metrics['cdist']:.3f}",
                    "dpp": f"{metrics['d_pp']:.3f}",
                    "dpn": f"{metrics['d_pn']:.3f}",
                    "var": f"{metrics['var_all']:.4f}",
                    "prevar": f"{pre_var_now:.4f}",
                    "gnorm": f"{gnorm_now:.3f}",
                    "lr": f"{lr_now:.2e}",
                }
                if args.train_objective == "disease_retrieval":
                    postfix["qpos"] = f"{retrieval_stats['q_pos']:.3f}"
                    postfix["qneg"] = f"{retrieval_stats['q_neg']:.3f}"
                    postfix["lsup"] = f"{loss_supcon_aux_now:.4f}"
                pbar_s1.set_postfix(**postfix)

                if use_wandb and s1_step % args.log_steps == 0:
                    import wandb
                    log_dict = {
                        "stage1/loss":         loss_now,
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
                    }
                    if args.train_objective == "disease_retrieval":
                        log_dict["stage1/loss_retrieval"] = loss_retrieval_now
                        log_dict["stage1/loss_supcon_aux"] = loss_supcon_aux_now
                        log_dict["stage1/query_pos_score"] = retrieval_stats["q_pos"]
                        log_dict["stage1/query_neg_score"] = retrieval_stats["q_neg"]
                    else:
                        log_dict["stage1/loss_supcon"] = loss_supcon_now
                    wandb.log(log_dict, step=s1_step)

                if (
                    eval_triplet_path is not None
                    and rank == 0
                    and args.stage1_eval_steps > 0
                    and s1_step % args.stage1_eval_steps == 0
                ):
                    val_acc, val_task_acc, extra_overall, extra_task = evaluate_rank0(
                        encoder, eval_triplet_path, eval_query_pair_path, args.eval_data_paths, store, device, args
                    )
                    logger.info(f"  [stage1 step {s1_step}] val triplet accuracy: {val_acc:.4f}")
                    logger.info(f"    [pre_emb cosine] {extra_overall['pre_cosine_triplet_acc']:.4f}")
                    logger.info(f"    [pre_emb l2] {extra_overall['pre_l2_triplet_acc']:.4f}")
                    if "query_cosine_pair_acc" in extra_overall:
                        logger.info(f"    [disease_query cosine] {extra_overall['query_cosine_pair_acc']:.4f}")
                        logger.info(f"    [disease_query l2] {extra_overall['query_l2_pair_acc']:.4f}")
                    if "query_auc" in extra_overall:
                        logger.info(f"    [disease_query auc] {extra_overall['query_auc']:.4f}")
                        logger.info(
                            f"    [disease_query counts] "
                            f"n={int(extra_overall.get('query_num_samples', float('nan')))}  "
                            f"pos={int(extra_overall.get('query_num_pos', float('nan')))}  "
                            f"neg={int(extra_overall.get('query_num_neg', float('nan')))}"
                        )
                    for task, acc in sorted(val_task_acc.items()):
                        logger.info(f"    {task}: {acc:.4f}")
                        logger.info(
                            f"      pre_cos={extra_task[task]['pre_cosine_triplet_acc']:.4f}  "
                            f"pre_l2={extra_task[task]['pre_l2_triplet_acc']:.4f}"
                        )
                    if use_wandb:
                        import wandb
                        ld = {
                            "stage1/step_val_triplet_acc":            val_acc,
                            "stage1/step_val_pre_cosine_triplet_acc": extra_overall["pre_cosine_triplet_acc"],
                            "stage1/step_val_pre_l2_triplet_acc":     extra_overall["pre_l2_triplet_acc"],
                        }
                        for task, acc in val_task_acc.items():
                            ld[f"stage1/step_val/{task}/triplet_acc"] = acc
                            ld[f"stage1/step_val/{task}/pre_cosine_triplet_acc"] = extra_task[task]["pre_cosine_triplet_acc"]
                            ld[f"stage1/step_val/{task}/pre_l2_triplet_acc"] = extra_task[task]["pre_l2_triplet_acc"]
                        wandb.log(ld, step=s1_step)

        pbar_s1.close()
        if rank == 0:
            logger.info(f"Stage 1 done ({s1_step} steps). Unfreezing LoRA for stage 2 …")

        if is_ddp:
            encoder = encoder.module
            dist.barrier()
        if uses_qwen_path:
            n_unfrozen = _set_lora_trainable(encoder, True)
            if rank == 0:
                logger.info(f"  Unfroze {n_unfrozen} LoRA parameter tensors")
        else:
            if rank == 0:
                logger.info("  encoder_mode does not use Qwen; keeping LoRA frozen")
        if is_ddp:
            encoder = _wrap_ddp_for_stage(
                encoder, is_ddp=True, local_rank=local_rank, stage="stage2"
            )
            dist.barrier()

        # Eval after stage 1
        if eval_triplet_path is not None and rank == 0:
            val_acc, val_task_acc, extra_overall, extra_task = evaluate_rank0(
                encoder, eval_triplet_path, eval_query_pair_path, args.eval_data_paths, store, device, args
            )
            logger.info(f"  [post-stage1] val triplet accuracy: {val_acc:.4f}")
            logger.info(f"    [pre_emb cosine] {extra_overall['pre_cosine_triplet_acc']:.4f}")
            logger.info(f"    [pre_emb l2] {extra_overall['pre_l2_triplet_acc']:.4f}")
            logger.info(f"    [bert_proj cosine] {extra_overall['proj_cosine_triplet_acc']:.4f}")
            logger.info(f"    [bert_proj l2] {extra_overall['proj_l2_triplet_acc']:.4f}")
            logger.info(f"    [raw_event cosine] {extra_overall['raw_cosine_triplet_acc']:.4f}")
            logger.info(f"    [raw_event l2] {extra_overall['raw_l2_triplet_acc']:.4f}")
            if "query_cosine_pair_acc" in extra_overall:
                logger.info(f"    [disease_query cosine] {extra_overall['query_cosine_pair_acc']:.4f}")
                logger.info(f"    [disease_query l2] {extra_overall['query_l2_pair_acc']:.4f}")
                if "query_auc" in extra_overall:
                    logger.info(f"    [disease_query auc] {extra_overall['query_auc']:.4f}")
                    logger.info(
                        f"    [disease_query counts] "
                        f"n={int(extra_overall.get('query_num_samples', float('nan')))}  "
                        f"pos={int(extra_overall.get('query_num_pos', float('nan')))}  "
                        f"neg={int(extra_overall.get('query_num_neg', float('nan')))}"
                    )
                logger.info(
                    f"    [disease_query p@10/r@10] "
                    f"{extra_overall.get('query_precision_at_10', float('nan')):.4f}/"
                    f"{extra_overall.get('query_recall_at_10', float('nan')):.4f}"
                )
                logger.info(
                    f"    [disease_query p@50/r@50] "
                    f"{extra_overall.get('query_precision_at_50', float('nan')):.4f}/"
                    f"{extra_overall.get('query_recall_at_50', float('nan')):.4f}"
                )
            for task, acc in sorted(val_task_acc.items()):
                logger.info(f"    {task}: {acc:.4f}")
                logger.info(
                    f"      pre_cos={extra_task[task]['pre_cosine_triplet_acc']:.4f}  "
                    f"pre_l2={extra_task[task]['pre_l2_triplet_acc']:.4f}  "
                    f"proj_cos={extra_task[task]['proj_cosine_triplet_acc']:.4f}  "
                    f"proj_l2={extra_task[task]['proj_l2_triplet_acc']:.4f}  "
                    f"raw_cos={extra_task[task]['raw_cosine_triplet_acc']:.4f}  "
                    f"raw_l2={extra_task[task]['raw_l2_triplet_acc']:.4f}"
                )
                if "query_cosine_pair_acc" in extra_task[task] or "query_auc" in extra_task[task]:
                    logger.info(
                        f"      query_cos={extra_task[task].get('query_cosine_pair_acc', float('nan')):.4f}  "
                        f"query_l2={extra_task[task].get('query_l2_pair_acc', float('nan')):.4f}  "
                        f"query_auc={extra_task[task].get('query_auc', float('nan')):.4f}  "
                        f"n={int(extra_task[task].get('query_num_samples', float('nan')))}  "
                        f"pos={int(extra_task[task].get('query_num_pos', float('nan')))}  "
                        f"neg={int(extra_task[task].get('query_num_neg', float('nan')))}  "
                        f"p@10={extra_task[task].get('query_precision_at_10', float('nan')):.4f}  "
                        f"r@10={extra_task[task].get('query_recall_at_10', float('nan')):.4f}  "
                        f"p@50={extra_task[task].get('query_precision_at_50', float('nan')):.4f}  "
                        f"r@50={extra_task[task].get('query_recall_at_50', float('nan')):.4f}"
                    )
            if use_wandb:
                import wandb
                ld = {
                    "stage1/val_triplet_acc":            val_acc,
                    "stage1/val_pre_cosine_triplet_acc": extra_overall["pre_cosine_triplet_acc"],
                    "stage1/val_pre_l2_triplet_acc":     extra_overall["pre_l2_triplet_acc"],
                    "stage1/val_proj_cosine_triplet_acc": extra_overall["proj_cosine_triplet_acc"],
                    "stage1/val_proj_l2_triplet_acc":     extra_overall["proj_l2_triplet_acc"],
                    "stage1/val_raw_cosine_triplet_acc":  extra_overall["raw_cosine_triplet_acc"],
                    "stage1/val_raw_l2_triplet_acc":      extra_overall["raw_l2_triplet_acc"],
                }
                for task, acc in val_task_acc.items():
                    ld[f"stage1/val/{task}/triplet_acc"] = acc
                    ld[f"stage1/val/{task}/pre_cosine_triplet_acc"] = extra_task[task]["pre_cosine_triplet_acc"]
                    ld[f"stage1/val/{task}/pre_l2_triplet_acc"] = extra_task[task]["pre_l2_triplet_acc"]
                    ld[f"stage1/val/{task}/proj_cosine_triplet_acc"] = extra_task[task]["proj_cosine_triplet_acc"]
                    ld[f"stage1/val/{task}/proj_l2_triplet_acc"] = extra_task[task]["proj_l2_triplet_acc"]
                    ld[f"stage1/val/{task}/raw_cosine_triplet_acc"] = extra_task[task]["raw_cosine_triplet_acc"]
                    ld[f"stage1/val/{task}/raw_l2_triplet_acc"] = extra_task[task]["raw_l2_triplet_acc"]
                    if "query_cosine_pair_acc" in extra_task[task]:
                        ld[f"stage1/val/{task}/query_cosine_pair_acc"] = extra_task[task]["query_cosine_pair_acc"]
                        ld[f"stage1/val/{task}/query_l2_pair_acc"] = extra_task[task]["query_l2_pair_acc"]
                    if "query_auc" in extra_task[task]:
                        ld[f"stage1/val/{task}/query_auc"] = extra_task[task]["query_auc"]
                        ld[f"stage1/val/{task}/query_precision_at_10"] = extra_task[task].get("query_precision_at_10", float("nan"))
                        ld[f"stage1/val/{task}/query_recall_at_10"] = extra_task[task].get("query_recall_at_10", float("nan"))
                        ld[f"stage1/val/{task}/query_precision_at_50"] = extra_task[task].get("query_precision_at_50", float("nan"))
                        ld[f"stage1/val/{task}/query_recall_at_50"] = extra_task[task].get("query_recall_at_50", float("nan"))
                wandb.log(ld, step=s1_step)

            raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
            raw_enc.save_checkpoint(run_dir / "stage1")
            logger.info(f"  Saved post-stage1 checkpoint → {run_dir / 'stage1'}")

        if args.legacy_stage1_only and not uses_qwen_path:
            if rank == 0:
                raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
                raw_enc.save_checkpoint(run_dir / "final")
                tokenizer.save_pretrained(str(run_dir / "tokenizer"))
                logger.info("legacy_stage1_only: stopping after post-stage1 evaluation")
                logger.info(f"Done. Best val triplet accuracy: {val_acc:.4f}")

            if use_wandb and wandb_run:
                import wandb
                wandb_run.finish()

            if is_ddp:
                dist.destroy_process_group()
            return

    # Wandb step offset so Stage 2 steps are monotonically after Stage 1 steps.
    wandb_step_offset = s1_step if (args.stage1_steps > 0 and uses_qwen_path) else 0

    # ── Stage 2 optimizer + LR scheduler ─────────────────────────────────────
    scheduled_batches_per_pass: list[int] = []
    for schedule_idx, data_epoch in enumerate(train_schedule):
        est_ds = EpochBatchDataset(
            args.train_data_dir, data_epoch, args.tasks, store,
            args.batch_size, schedule_idx, args.seed, world_size, rank,
            args.pad_to_num_events,
        )
        scheduled_batches_per_pass.append(len(est_ds))
        del est_ds

    total_opt_steps = sum(
        math.ceil(n_batches / args.grad_accum) for n_batches in scheduled_batches_per_pass
    )
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
        if uses_qwen_path:
            logger.info(
                f"Training: {len(train_schedule)} epochs, "
                f"~{scheduled_batches_per_pass[0]} batches/rank/epoch, "
                f"{total_opt_steps} optimizer steps total"
            )
        else:
            logger.info(
                f"Training: single phase, {len(train_schedule)} prepared data epoch(s), "
                f"{sum(scheduled_batches_per_pass)} batches/rank total, "
                f"{total_opt_steps} optimizer steps total"
            )
        logger.info(f"  lr_proj={args.lr_proj}, lr_lora={args.lr_lora}, "
                    f"warmup={warmup_steps} steps")

    # ── Training loop (stage 2) ───────────────────────────────────────────────
    best_val_acc = 0.0
    opt_step     = 0

    for epoch, data_epoch in enumerate(train_schedule):
        if is_ddp:
            dist.barrier()

        if rank == 0:
            if uses_qwen_path:
                logger.info(f"Epoch {epoch+1}/{len(train_schedule)}: data epoch {data_epoch} …")
            else:
                logger.info(
                    f"Pass {epoch+1}/{len(train_schedule)}: prepared data epoch {data_epoch} …"
                )

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
            task_idxs_t = batch["task_idxs"].to(device)

            is_update_step = (
                (batch_idx + 1) % args.grad_accum == 0
                or (batch_idx + 1) == n_batches_this_epoch
            )
            sync_ctx = nullcontext() if (is_update_step or not is_ddp) \
                       else encoder.no_sync()

            with sync_ctx:
                emb, pre_emb = _forward_embeddings(
                    encoder, batch, device, args.encoder_mode, return_pre_emb=True
                )
                if args.train_objective == "disease_retrieval":
                    retrieval_loss, retrieval_stats = _disease_retrieval_loss(
                        encoder,
                        pre_emb,
                        task_idxs_t,
                        labels_t.to(device),
                        args.retrieval_temperature,
                    )
                    supcon_aux = _supervised_contrastive_loss(
                        pre_emb, labels_t.to(device), args.stage1_temperature
                    )
                    main_loss = retrieval_loss + args.retrieval_supcon_weight * supcon_aux
                    triplet_loss = main_loss.new_zeros(())
                    supcon_loss = supcon_aux
                else:
                    supcon_loss = _supervised_contrastive_loss(
                        pre_emb, labels_t.to(device), args.stage1_temperature
                    )
                    triplet_loss = _loss_fn(labels_t.to(device), emb)
                    retrieval_stats = {"q_pos": float("nan"), "q_neg": float("nan")}
                if args.var_reg_weight > 0.0:
                    std = torch.sqrt(pre_emb.var(dim=0) + 1e-4)
                    var_loss = torch.relu(1.0 - std).mean()
                else:
                    var_loss = pre_emb.new_zeros(())

                if args.train_objective == "disease_retrieval":
                    loss = main_loss + args.var_reg_weight * var_loss
                else:
                    loss = (
                        supcon_loss
                        + args.var_reg_weight * var_loss
                        + args.stage2_triplet_weight * triplet_loss
                    )
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
                        if args.train_objective == "disease_retrieval":
                            retrieval_now = retrieval_loss.item()
                            supcon_now = supcon_aux.item()
                            triplet_now = float("nan")
                        else:
                            supcon_now = supcon_loss.item()
                            triplet_now = triplet_loss.item()
                        var_loss_now = var_loss.item() if args.var_reg_weight > 0.0 else 0.0

                    postfix = {
                        "loss": f"{loss_now:.4f}",
                        "cdist": f"{metrics['cdist']:.3f}",
                        "dpp": f"{metrics['d_pp']:.3f}",
                        "dpn": f"{metrics['d_pn']:.3f}",
                        "var": f"{metrics['var_all']:.4f}",
                        "prevar": f"{pre_var_now:.4f}",
                        "gnorm": f"{gnorm_now:.3f}",
                        "lr": f"{lr_now:.2e}",
                    }
                    if args.train_objective == "disease_retrieval":
                        postfix["lret"] = f"{retrieval_now:.4f}"
                        postfix["lsup"] = f"{supcon_now:.4f}"
                        postfix["qpos"] = f"{retrieval_stats['q_pos']:.3f}"
                        postfix["qneg"] = f"{retrieval_stats['q_neg']:.3f}"
                    else:
                        postfix["lsup"] = f"{supcon_now:.4f}"
                        postfix["ltri"] = f"{triplet_now:.4f}"
                    pbar.set_postfix(**postfix)

                    if use_wandb and opt_step % args.log_steps == 0:
                        import wandb
                        log_dict = {
                            "train/loss":         loss_now,
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
                        if args.train_objective == "disease_retrieval":
                            log_dict["train/loss_retrieval"] = retrieval_now
                            log_dict["train/loss_supcon_aux"] = supcon_now
                            log_dict["train/query_pos_score"] = retrieval_stats["q_pos"]
                            log_dict["train/query_neg_score"] = retrieval_stats["q_neg"]
                        else:
                            log_dict["train/loss_supcon"] = supcon_now
                            log_dict["train/loss_triplet"] = triplet_now
                        if len(optimizer.param_groups) > 1:
                            log_dict["train/lr_lora"] = optimizer.param_groups[1]["lr"]
                        wandb.log(log_dict, step=wandb_step_offset + opt_step)

                    if (
                        eval_triplet_path is not None
                        and rank == 0
                        and args.eval_steps > 0
                        and opt_step % args.eval_steps == 0
                    ):
                        val_acc, val_task_acc, extra_overall, extra_task = evaluate_rank0(
                            encoder, eval_triplet_path, eval_query_pair_path, args.eval_data_paths, store, device, args
                        )
                        logger.info(
                            f"  [step {opt_step}] val triplet accuracy: {val_acc:.4f}"
                        )
                        logger.info(f"    [pre_emb cosine] {extra_overall['pre_cosine_triplet_acc']:.4f}")
                        logger.info(f"    [pre_emb l2] {extra_overall['pre_l2_triplet_acc']:.4f}")
                        if "query_cosine_pair_acc" in extra_overall:
                            logger.info(f"    [disease_query cosine] {extra_overall['query_cosine_pair_acc']:.4f}")
                            logger.info(f"    [disease_query l2] {extra_overall['query_l2_pair_acc']:.4f}")
                        if "query_auc" in extra_overall:
                            logger.info(f"    [disease_query auc] {extra_overall['query_auc']:.4f}")
                            logger.info(
                                f"    [disease_query counts] "
                                f"n={int(extra_overall.get('query_num_samples', float('nan')))}  "
                                f"pos={int(extra_overall.get('query_num_pos', float('nan')))}  "
                                f"neg={int(extra_overall.get('query_num_neg', float('nan')))}"
                            )
                            logger.info(
                                f"    [disease_query p@10/r@10] "
                                f"{extra_overall.get('query_precision_at_10', float('nan')):.4f}/"
                                f"{extra_overall.get('query_recall_at_10', float('nan')):.4f}"
                            )
                        for task, acc in sorted(val_task_acc.items()):
                            logger.info(f"    {task}: {acc:.4f}")
                            logger.info(
                                f"      pre_cos={extra_task[task]['pre_cosine_triplet_acc']:.4f}  "
                                f"pre_l2={extra_task[task]['pre_l2_triplet_acc']:.4f}"
                            )
                            if "query_cosine_pair_acc" in extra_task[task] or "query_auc" in extra_task[task]:
                                logger.info(
                                    f"      query_cos={extra_task[task].get('query_cosine_pair_acc', float('nan')):.4f}  "
                                    f"query_l2={extra_task[task].get('query_l2_pair_acc', float('nan')):.4f}  "
                                    f"query_auc={extra_task[task].get('query_auc', float('nan')):.4f}  "
                                    f"n={int(extra_task[task].get('query_num_samples', float('nan')))}  "
                                    f"pos={int(extra_task[task].get('query_num_pos', float('nan')))}  "
                                    f"neg={int(extra_task[task].get('query_num_neg', float('nan')))}  "
                                    f"p@10={extra_task[task].get('query_precision_at_10', float('nan')):.4f}  "
                                    f"r@10={extra_task[task].get('query_recall_at_10', float('nan')):.4f}  "
                                    f"p@50={extra_task[task].get('query_precision_at_50', float('nan')):.4f}  "
                                    f"r@50={extra_task[task].get('query_recall_at_50', float('nan')):.4f}"
                                )
                        if use_wandb:
                            import wandb
                            log_dict = {
                                "val/step_triplet_acc":            val_acc,
                                "val/step_pre_cosine_triplet_acc": extra_overall["pre_cosine_triplet_acc"],
                                "val/step_pre_l2_triplet_acc":     extra_overall["pre_l2_triplet_acc"],
                                "epoch": epoch + (batch_idx + 1) / max(n_batches_this_epoch, 1),
                            }
                            for task, acc in val_task_acc.items():
                                log_dict[f"val/step/{task}/triplet_acc"] = acc
                                log_dict[f"val/step/{task}/pre_cosine_triplet_acc"] = extra_task[task]["pre_cosine_triplet_acc"]
                                log_dict[f"val/step/{task}/pre_l2_triplet_acc"] = extra_task[task]["pre_l2_triplet_acc"]
                                if "query_cosine_pair_acc" in extra_task[task]:
                                    log_dict[f"val/step/{task}/query_cosine_pair_acc"] = extra_task[task]["query_cosine_pair_acc"]
                                    log_dict[f"val/step/{task}/query_l2_pair_acc"] = extra_task[task]["query_l2_pair_acc"]
                                if "query_auc" in extra_task[task]:
                                    log_dict[f"val/step/{task}/query_auc"] = extra_task[task]["query_auc"]
                                    log_dict[f"val/step/{task}/query_precision_at_10"] = extra_task[task].get("query_precision_at_10", float("nan"))
                                    log_dict[f"val/step/{task}/query_recall_at_10"] = extra_task[task].get("query_recall_at_10", float("nan"))
                                    log_dict[f"val/step/{task}/query_precision_at_50"] = extra_task[task].get("query_precision_at_50", float("nan"))
                                    log_dict[f"val/step/{task}/query_recall_at_50"] = extra_task[task].get("query_recall_at_50", float("nan"))
                            wandb.log(log_dict, step=wandb_step_offset + opt_step)

            epoch_loss += loss.item() * args.grad_accum

        n_batches_ran = batch_idx + 1
        avg_loss      = epoch_loss / max(n_batches_ran, 1)
        if rank == 0:
            if uses_qwen_path:
                logger.info(f"Epoch {epoch+1}/{len(train_schedule)}  avg_loss={avg_loss:.4f}")
            else:
                logger.info(f"Pass {epoch+1}/{len(train_schedule)}  avg_loss={avg_loss:.4f}")

        # ── Eval ──────────────────────────────────────────────────────────────
        if eval_triplet_path is not None and rank == 0:
            val_acc, val_task_acc, extra_overall, extra_task = evaluate_rank0(
                encoder, eval_triplet_path, eval_query_pair_path, args.eval_data_paths, store, device, args
            )
            logger.info(f"  val triplet accuracy: {val_acc:.4f}")
            logger.info(f"    [pre_emb cosine] {extra_overall['pre_cosine_triplet_acc']:.4f}")
            logger.info(f"    [pre_emb l2] {extra_overall['pre_l2_triplet_acc']:.4f}")
            logger.info(f"    [bert_proj cosine] {extra_overall['proj_cosine_triplet_acc']:.4f}")
            logger.info(f"    [bert_proj l2] {extra_overall['proj_l2_triplet_acc']:.4f}")
            logger.info(f"    [raw_event cosine] {extra_overall['raw_cosine_triplet_acc']:.4f}")
            logger.info(f"    [raw_event l2] {extra_overall['raw_l2_triplet_acc']:.4f}")
            if "query_cosine_pair_acc" in extra_overall:
                logger.info(f"    [disease_query cosine] {extra_overall['query_cosine_pair_acc']:.4f}")
                logger.info(f"    [disease_query l2] {extra_overall['query_l2_pair_acc']:.4f}")
            if "query_auc" in extra_overall:
                logger.info(f"    [disease_query auc] {extra_overall['query_auc']:.4f}")
                logger.info(
                    f"    [disease_query counts] "
                    f"n={int(extra_overall.get('query_num_samples', float('nan')))}  "
                    f"pos={int(extra_overall.get('query_num_pos', float('nan')))}  "
                    f"neg={int(extra_overall.get('query_num_neg', float('nan')))}"
                )
                logger.info(
                    f"    [disease_query p@10/r@10] "
                    f"{extra_overall.get('query_precision_at_10', float('nan')):.4f}/"
                    f"{extra_overall.get('query_recall_at_10', float('nan')):.4f}"
                )
                logger.info(
                    f"    [disease_query p@50/r@50] "
                    f"{extra_overall.get('query_precision_at_50', float('nan')):.4f}/"
                    f"{extra_overall.get('query_recall_at_50', float('nan')):.4f}"
                )
            for task, acc in sorted(val_task_acc.items()):
                logger.info(f"    {task}: {acc:.4f}")
                logger.info(
                    f"      pre_cos={extra_task[task]['pre_cosine_triplet_acc']:.4f}  "
                    f"pre_l2={extra_task[task]['pre_l2_triplet_acc']:.4f}  "
                    f"proj_cos={extra_task[task]['proj_cosine_triplet_acc']:.4f}  "
                    f"proj_l2={extra_task[task]['proj_l2_triplet_acc']:.4f}  "
                    f"raw_cos={extra_task[task]['raw_cosine_triplet_acc']:.4f}  "
                    f"raw_l2={extra_task[task]['raw_l2_triplet_acc']:.4f}"
                )
                if "query_cosine_pair_acc" in extra_task[task] or "query_auc" in extra_task[task]:
                    logger.info(
                        f"      query_cos={extra_task[task].get('query_cosine_pair_acc', float('nan')):.4f}  "
                        f"query_l2={extra_task[task].get('query_l2_pair_acc', float('nan')):.4f}  "
                        f"query_auc={extra_task[task].get('query_auc', float('nan')):.4f}  "
                        f"n={int(extra_task[task].get('query_num_samples', float('nan')))}  "
                        f"pos={int(extra_task[task].get('query_num_pos', float('nan')))}  "
                        f"neg={int(extra_task[task].get('query_num_neg', float('nan')))}  "
                        f"p@10={extra_task[task].get('query_precision_at_10', float('nan')):.4f}  "
                        f"r@10={extra_task[task].get('query_recall_at_10', float('nan')):.4f}  "
                        f"p@50={extra_task[task].get('query_precision_at_50', float('nan')):.4f}  "
                        f"r@50={extra_task[task].get('query_recall_at_50', float('nan')):.4f}"
                    )
            if use_wandb:
                import wandb
                log_dict = {
                    "val/triplet_acc":            val_acc,
                    "val/pre_cosine_triplet_acc": extra_overall["pre_cosine_triplet_acc"],
                    "val/pre_l2_triplet_acc":     extra_overall["pre_l2_triplet_acc"],
                    "val/proj_cosine_triplet_acc": extra_overall["proj_cosine_triplet_acc"],
                    "val/proj_l2_triplet_acc":     extra_overall["proj_l2_triplet_acc"],
                    "val/raw_cosine_triplet_acc":  extra_overall["raw_cosine_triplet_acc"],
                    "val/raw_l2_triplet_acc":      extra_overall["raw_l2_triplet_acc"],
                    "train/epoch_loss":           avg_loss,
                    "epoch":                      epoch + 1,
                }
                for task, acc in val_task_acc.items():
                    log_dict[f"val/{task}/triplet_acc"] = acc
                    log_dict[f"val/{task}/pre_cosine_triplet_acc"] = extra_task[task]["pre_cosine_triplet_acc"]
                    log_dict[f"val/{task}/pre_l2_triplet_acc"] = extra_task[task]["pre_l2_triplet_acc"]
                    log_dict[f"val/{task}/proj_cosine_triplet_acc"] = extra_task[task]["proj_cosine_triplet_acc"]
                    log_dict[f"val/{task}/proj_l2_triplet_acc"] = extra_task[task]["proj_l2_triplet_acc"]
                    log_dict[f"val/{task}/raw_cosine_triplet_acc"] = extra_task[task]["raw_cosine_triplet_acc"]
                    log_dict[f"val/{task}/raw_l2_triplet_acc"] = extra_task[task]["raw_l2_triplet_acc"]
                    if "query_cosine_pair_acc" in extra_task[task]:
                        log_dict[f"val/{task}/query_cosine_pair_acc"] = extra_task[task]["query_cosine_pair_acc"]
                        log_dict[f"val/{task}/query_l2_pair_acc"] = extra_task[task]["query_l2_pair_acc"]
                    if "query_auc" in extra_task[task]:
                        log_dict[f"val/{task}/query_auc"] = extra_task[task]["query_auc"]
                        log_dict[f"val/{task}/query_precision_at_10"] = extra_task[task].get("query_precision_at_10", float("nan"))
                        log_dict[f"val/{task}/query_recall_at_10"] = extra_task[task].get("query_recall_at_10", float("nan"))
                        log_dict[f"val/{task}/query_precision_at_50"] = extra_task[task].get("query_precision_at_50", float("nan"))
                        log_dict[f"val/{task}/query_recall_at_50"] = extra_task[task].get("query_recall_at_50", float("nan"))
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
