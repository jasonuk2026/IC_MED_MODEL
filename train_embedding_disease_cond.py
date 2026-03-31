#!/usr/bin/env python3
"""
train_embedding_disease_cond.py

Disease-conditioned EHR embedding model using a BioLinkBERT disease virtual token.

Pre-computed BioLinkBERT event embeddings are conditioned on disease-specific
BioLinkBERT embeddings via a lightweight cross-attention layer, then fed into a
Qwen3-Embedding model (LoRA fine-tuned) to produce disease-aware patient
representations trained with BatchAllTripletLoss.

Architecture per sample
───────────────────────
  [event_1, …, event_N]
    → BioLinkBERT emb lookup (768-d)
    → EventProjection  Linear(768 → D_qwen)

  BioLinkBERT(disease_name)  (768-d)
    → DiseaseProjection  Linear(768 → D_qwen)
    → inserted as a single VIRTUAL TOKEN in the Qwen input sequence

  Qwen input sequence (matches prompt template semantics):
    [prefix_embeds | disease_virtual_token | middle_embeds | projected events | <|endoftext|>]

  where prefix = "Please predict disease "
  and   middle = " based on the following events.\nStart of medical events:"

  Text embeddings (prefix, middle, EOS) are pre-computed once from the frozen
  Qwen embedding table at model init and stored as buffers — no embedding
  lookup in the forward pass.

  Qwen self-attention (every layer) naturally allows:
    • each event to attend to the disease token  (event relevance)
    • the disease token to attend to all events  (disease awareness)
  → last-token pool (<|endoftext|>) → L2 normalise → BatchAllTripletLoss

Disease conditioning
─────────────────────
Six disease-name strings are encoded once with BioLinkBERT (mean-pool, no
special tokens) at training start using --bert_model. The resulting 768-d
vectors are stored on CPU and looked up per batch.

The projected disease embedding occupies the position of the disease name in
the prompt template.  At every Qwen layer, each event token can attend to it
with an event-specific weight — providing position-differentiated conditioning
without any extra modules.

Interpretability: after a forward pass, extract Qwen's self-attention weights
at position P (disease token column) to see which events attend to disease most.

New trainable parameters (in addition to Qwen LoRA):
  EventProjection   Linear(768 → D_qwen)   no bias
  DiseaseProjection Linear(768 → D_qwen)   no bias

Usage (single node, 4 GPUs)
─────────────────────────────
  torchrun --nproc_per_node=4 train_embedding_disease_cond.py \\
      --data_paths     data/embedding_inputs/sharded_m500/train_shard_*.parquet \\
      --val_data_paths data/embedding_inputs/sharded_m500/val.parquet \\
      --bert_index     data/biolinkbert_embeddings/event_index.parquet \\
      --bert_embeddings data/biolinkbert_embeddings/embeddings.npy \\
      --bert_model     michiyasunaga/BioLinkBERT-base \\
      --bf16 --flash_attn

Single GPU
──────────
  python train_embedding_disease_cond.py \\
      --data_paths all.parquet \\
      --bert_index     data/biolinkbert_embeddings/event_index.parquet \\
      --bert_embeddings data/biolinkbert_embeddings/embeddings.npy \\
      --bert_model     michiyasunaga/BioLinkBERT-base

NOTE: QLoRA (--qlora) is incompatible with multi-GPU DDP.
"""

import os
import math
import random
import logging
import argparse
from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict, Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from utils.h2d import CudaPrefetcher
from utils.async_dataloader import AsyncDataLoader

import pandas as pd
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, logging as hf_logging
from extract_biolinkbert_embeddings import mean_pool_no_special, normalise_event_key
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sentence_transformers.losses import BatchAllTripletLoss

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

BERT_DIM    = 768     # BioLinkBERT-base hidden size
EOS_TOKEN_ID = 151643  # Qwen3 <|endoftext|> — pooling anchor token

# Prompt template split around the disease virtual token and event virtual tokens:
#   "Please predict disease <DISEASE> based on the following events.\nStart of medical events: <EVENTS>"
PROMPT_PREFIX = "Please predict disease "
PROMPT_MIDDLE = " based on the following events.\nStart of medical events:"

# ── BioLinkBERT event embedding lookup ────────────────────────────────────────

class EventLookup:
    """Memory-mapped lookup: (code, value, unit) → BioLinkBERT embedding (float32).

    Keyed by the raw (code, value, unit) tuple from the index rather than the
    formatted event text, so that description mismatches between concept.csv and
    the event dicts in training parquets can never cause key misses.
    """

    def __init__(self, index_path: str, embeddings_path: str):
        logger.info(f"Loading event index from {index_path} …")
        index_df = pd.read_parquet(index_path)
        # Key: (code, value, unit) — same three columns used during extraction
        self.key2idx: dict[tuple[str, str, str], int] = {
            normalise_event_key(c, v, u): int(eid)
            for c, v, u, eid in zip(
                index_df["code"], index_df["value"], index_df["unit"], index_df["event_id"]
            )
        }
        logger.info(f"  {len(self.key2idx):,} unique (code, value, unit) keys indexed.")

        logger.info(f"Loading embeddings (mmap) from {embeddings_path} …")
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        logger.info(f"  Embeddings shape: {self.embeddings.shape}  dtype: {self.embeddings.dtype}")

    def get(self, e: dict, warn_missing: bool = True) -> np.ndarray | None:
        key = normalise_event_key(e.get("code"), e.get("value"), e.get("unit"))
        idx = self.key2idx.get(key)
        if idx is None:
            if warn_missing:
                raise KeyError(
                    f"Event key not found in BioLinkBERT index.\n"
                    f"  Missing key (code, value, unit): {key!r}"
                )
            return None
        return self.embeddings[idx].copy()


# ── BioLinkBERT disease encoder ───────────────────────────────────────────────

def encode_disease_names(
    bert_model_path: str,
    disease_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Encode disease name strings with BioLinkBERT (mean pool, no special tokens).

    The BioLinkBERT model is loaded onto *device* for encoding, then moved back
    to CPU and deleted so GPU memory is freed before Qwen loads.

    Returns {disease_name: tensor(768,)} on CPU, float32.
    """
    logger.info(f"Encoding {len(disease_names)} disease names with BioLinkBERT "
                f"from {bert_model_path} …")

    bert_tok = AutoTokenizer.from_pretrained(bert_model_path, local_files_only=True)
    bert_mod = AutoModel.from_pretrained(bert_model_path, local_files_only=True)
    bert_mod = bert_mod.to(device).eval()

    special_ids: set[int] = set(bert_tok.all_special_ids)

    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in disease_names:
            enc = bert_tok(
                name, return_tensors="pt", padding=False,
                truncation=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = bert_mod(**enc)
            emb = mean_pool_no_special(
                out.last_hidden_state,
                enc["input_ids"],
                enc["attention_mask"],
                special_ids,
            )   # (1, 768)
            result[name] = emb[0].cpu().float()

    # Free GPU memory before loading Qwen
    bert_mod.cpu()
    del bert_mod
    torch.cuda.empty_cache()

    logger.info("  Disease name encoding complete.")
    return result


# ── Dataset ───────────────────────────────────────────────────────────────────

class EHREmbeddingDataset(Dataset):
    """Lazy dataset: stores event-id index lists instead of materialised embeddings.

    Accepts a list of parquet file paths and processes them one at a time so that
    the raw events column (Python dicts) is never kept in memory across files.
    Embeddings are read from the memory-mapped EventLookup.embeddings array in
    __getitem__, so only one batch worth of float data is ever resident in RAM.

    self.labels  : list[int]  — binary: 0 = negative, 1 = positive
    self.tasks   : list[str]  — task name per sample (for per-disease sampling/eval)
    """

    def __init__(self, paths: list[str], lookup: EventLookup, split: str | None = None):
        self.lookup  = lookup
        self.samples: list[dict] = []
        self.labels:  list[int]  = []
        self.tasks:   list[str]  = []
        for path in paths:
            df = pd.read_parquet(path)
            if split is not None:
                df = df[df["split"] == split]
            for task, events, label in zip(df["task"], df["events"], df["label"]):
                idx_list = []
                for e in events:
                    key = normalise_event_key(e.get("code"), e.get("value"), e.get("unit"))
                    eid = lookup.key2idx.get(key)
                    if eid is None:
                        raise KeyError(f"Missing key: {key}")
                    idx_list.append(eid)

                if not idx_list:
                    raise ValueError("Shouldn't happen")

                disease_name = TASK_2_DISEASE_NAME.get(task, task)
                self.samples.append({
                    "event_ids":    idx_list,   # list[int], resolved lazily in __getitem__
                    "disease_name": disease_name,
                    "task":         task,
                })
                self.labels.append(int(bool(label)))
                self.tasks.append(task)
            del df  # explicitly release raw events dicts before loading next shard
        task_counts = Counter(self.tasks)
        pos_counts  = Counter(t for t, l in zip(self.tasks, self.labels) if l == 1)
        logger.info(f"  {len(self.samples)} samples across {len(task_counts)} tasks")
        for task in sorted(task_counts):
            logger.info(f"    {task}: {pos_counts[task]} pos / "
                        f"{task_counts[task] - pos_counts[task]} neg")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        embs = np.stack([self.lookup.embeddings[i] for i in s["event_ids"]]).astype(np.float32)
        sample = {"event_embs": embs, "disease_name": s["disease_name"], "task": s["task"]}
        return sample, self.labels[idx]


# ── Batch sampler ─────────────────────────────────────────────────────────────

class SingleDiseaseGroupSampler:
    """Each batch contains samples from exactly ONE disease task.

    Within each batch, both positives (label=1) and negatives (label=0) are
    included in equal proportion so BatchAllTripletLoss can build valid triplets.
    Batches from different diseases are never mixed.

    This matches the evaluation setup where triplets are always within a single
    disease cohort, and avoids the model learning trivial cross-disease separation.

    Algorithm per epoch:
      1. For each task, shuffle its positives and negatives independently and
         pre-build a queue of batches.
      2. Interleave batches across tasks in round-robin order (task order
         re-shuffled each round) until all task queues are exhausted.
         This prevents the model from seeing a long run of one disease at a time.
    """

    def __init__(
        self,
        tasks:      list[str],
        labels:     list[int],   # 0 = negative, 1 = positive
        batch_size: int,
        seed:       int  = 42,
    ):
        if batch_size < 4 or batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even and ≥4, got {batch_size}")
        self.batch_size = batch_size
        self.seed       = seed
        self.epoch      = 0

        pos: dict[str, list[int]] = defaultdict(list)
        neg: dict[str, list[int]] = defaultdict(list)
        for idx, (task, lbl) in enumerate(zip(tasks, labels)):
            (pos if lbl == 1 else neg)[task].append(idx)

        # Only keep tasks that have enough data for at least one full batch
        half = batch_size // 2
        self.valid_tasks = sorted(
            t for t in pos
            if len(pos[t]) >= half and len(neg.get(t, [])) >= half
        )
        if not self.valid_tasks:
            raise ValueError(
                f"No disease task has ≥{half} positives and ≥{half} negatives. "
                f"Reduce --batch_size or provide more data."
            )
        self.pos = {t: pos[t] for t in self.valid_tasks}
        self.neg = {t: neg[t] for t in self.valid_tasks}

        skipped = set(pos) - set(self.valid_tasks)
        if skipped:
            logger.warning(f"  SingleDiseaseGroupSampler: skipped tasks with insufficient "
                           f"data (need ≥{half} per class): {sorted(skipped)}")

        # Approximate total batches across all tasks per epoch
        self._n_batches = sum(
            min(len(self.pos[t]), len(self.neg[t])) // half
            for t in self.valid_tasks
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        half = self.batch_size // 2

        # 1. Pre-build per-task batch queues (shuffle samples within each task)
        task_queues: dict[str, deque] = {}
        for task in self.valid_tasks:
            pos_perm = torch.randperm(len(self.pos[task]), generator=g).tolist()
            neg_perm = torch.randperm(len(self.neg[task]), generator=g).tolist()
            pos_q = deque(self.pos[task][i] for i in pos_perm)
            neg_q = deque(self.neg[task][i] for i in neg_perm)
            batches: deque = deque()
            while len(pos_q) >= half and len(neg_q) >= half:
                batches.append(
                    [pos_q.popleft() for _ in range(half)] +
                    [neg_q.popleft() for _ in range(half)]
                )
            task_queues[task] = batches

        # 2. Interleave: each round, yield one batch from each active task in
        #    a freshly shuffled order, then drop exhausted tasks.
        active = list(self.valid_tasks)
        while active:
            order = [active[i]
                     for i in torch.randperm(len(active), generator=g).tolist()]
            next_active = []
            for task in order:
                if task_queues[task]:
                    yield task_queues[task].popleft()
                if task_queues[task]:
                    next_active.append(task)
            active = next_active

    def __len__(self):
        return self._n_batches


# ── Collate ───────────────────────────────────────────────────────────────────

def make_collate_fn(disease_emb_dict: dict[str, torch.Tensor]):
    """Pad event embedding sequences and look up disease embeddings.

    Prompt text embeddings are pre-computed once in DiseaseAwareEHREncoder.__init__
    as registered buffers, so collate no longer needs a tokenizer.

    disease_emb_dict: {disease_name: tensor(768,)} on CPU.
    """

    def collate_fn(batch):
        samples = [b[0] for b in batch]
        labels  = [b[1] for b in batch]

        # ── Pad event embedding sequences ─────────────────────────────────
        event_embs_list = [s["event_embs"] for s in samples]   # list of (n_i, 768)
        max_e    = max(e.shape[0] for e in event_embs_list)
        bert_dim = event_embs_list[0].shape[1]

        padded_event_embs = torch.zeros(len(samples), max_e, bert_dim, dtype=torch.float32)
        event_attn_mask   = torch.zeros(len(samples), max_e, dtype=torch.long)
        for i, embs in enumerate(event_embs_list):
            n = embs.shape[0]
            padded_event_embs[i, :n] = torch.from_numpy(embs)
            event_attn_mask[i, :n]   = 1

        # ── Look up BioLinkBERT disease embeddings ────────────────────────
        disease_embs = torch.stack([
            disease_emb_dict[s["disease_name"]] for s in samples
        ])  # (B, 768)

        return {
            "event_embs":   padded_event_embs,    # (B, max_e, 768) float
            "event_mask":   event_attn_mask,       # (B, max_e)      long
            "disease_embs": disease_embs,          # (B, 768)        float
            "labels":       torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn




# ── Disease-Aware EHR Encoder ─────────────────────────────────────────────────

class DiseaseAwareEHREncoder(nn.Module):
    """Qwen model conditioned on disease via a single projected virtual token.

    The BioLinkBERT disease embedding is projected to Qwen's hidden size and
    inserted at the position of the disease name in the prompt template:

        [prefix_embeds | disease_virtual_token | middle_embeds | projected event embs | EOS]

    where prefix = PROMPT_PREFIX = "Please predict disease "
    and   middle = PROMPT_MIDDLE = " based on the following events.\\nStart of medical events:"

    The text prefix/middle/EOS embeddings are pre-computed once at init time from the
    frozen Qwen embedding table and stored as buffers — no embed_fn calls in forward.

    Qwen's own self-attention at every layer then lets:
      • each event attend to the disease token with an event-specific weight
      • the disease token attend to all events (bidirectional within each layer)

    Interpretability: after inference, extract Qwen's self-attention weights at
    the disease token column (position = len(prefix_tokens)) to see which events
    attended to disease most at each layer.

    Trainable modules beyond Qwen LoRA:
        event_proj    Linear(bert_dim → qwen_dim)   no bias
        disease_proj  Linear(bert_dim → qwen_dim)   no bias
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        bert_dim:   int,
        qwen_dim:   int,
        prefix_ids: torch.Tensor,   # (1, P1)  token ids for PROMPT_PREFIX
        middle_ids: torch.Tensor,   # (1, P2)  token ids for PROMPT_MIDDLE
    ):
        super().__init__()
        self.qwen         = qwen_model
        self.event_proj   = nn.Linear(bert_dim, qwen_dim, bias=False)
        self.disease_proj = nn.Linear(bert_dim, qwen_dim, bias=False)

        nn.init.xavier_uniform_(self.event_proj.weight)
        nn.init.xavier_uniform_(self.disease_proj.weight)

        # Pre-compute frozen text embeddings once; store as buffers so they
        # move to the right device automatically and never appear in the forward graph.
        embed_fn = qwen_model.get_input_embeddings()
        with torch.no_grad():
            self.register_buffer("prefix_embeds", embed_fn(prefix_ids).float())  # (1, P1, D)
            self.register_buffer("middle_embeds", embed_fn(middle_ids).float())  # (1, P2, D)
            eos_ids = torch.full((1, 1), EOS_TOKEN_ID, dtype=torch.long)
            self.register_buffer("eos_embed",    embed_fn(eos_ids).float())       # (1,  1, D)

    def forward(
        self,
        event_embs:    torch.Tensor,   # (B, N, 768)  float
        event_mask:    torch.Tensor,   # (B, N)       long
        disease_embs:  torch.Tensor,   # (B, 768)     float
        compute_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:                 # (B, D_qwen)  L2-normalised embeddings
        B      = event_embs.size(0)
        device = event_embs.device

        # 1. Project event embeddings to Qwen hidden size
        ev_proj   = self.event_proj(event_embs.to(compute_dtype))                    # (B, N, D)

        # 2. Project disease embedding → virtual token (B, 1, D)
        dis_token = self.disease_proj(disease_embs.to(compute_dtype)).unsqueeze(1)

        # 3. Retrieve pre-computed text embeddings (registered buffers, frozen).
        #    Sequence layout matches the prompt template:
        #      "Please predict disease <DISEASE> based on the following events.
        #       Start of medical events: <EVENTS>"
        prefix = self.prefix_embeds.to(compute_dtype).expand(B, -1, -1)  # (B, P1, D)
        middle = self.middle_embeds.to(compute_dtype).expand(B, -1, -1)  # (B, P2, D)
        eos    = self.eos_embed.to(compute_dtype).expand(B, -1, -1)       # (B,  1, D)

        # 4. Assemble: [prefix | disease_token | middle | events | EOS]
        inputs_embeds = torch.cat([prefix, dis_token, middle, ev_proj, eos], dim=1)

        P1 = prefix.size(1)
        P2 = middle.size(1)
        ones = lambda n: torch.ones(B, n, dtype=torch.long, device=device)
        attention_mask = torch.cat(
            [ones(P1), ones(1), ones(P2), event_mask, ones(1)], dim=1
        )

        # 5. Forward through Qwen (inputs_embeds bypasses embed_tokens)
        out = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # 6. Last-token pool (EOS) → L2 normalise
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx   = torch.arange(B, device=device)
        emb = out.last_hidden_state[batch_idx, seq_lengths]
        return F.normalize(emb.float(), p=2, dim=-1)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        raw_qwen = self.qwen.module if isinstance(self.qwen, DDP) else self.qwen
        raw_qwen.save_pretrained(str(save_dir / "lora"))
        torch.save(
            {
                "event_proj":   self.event_proj.state_dict(),
                "disease_proj": self.disease_proj.state_dict(),
            },
            save_dir / "extra_modules.pt",
        )
        logger.info(f"  Saved checkpoint → {save_dir}")

    @classmethod
    def load_from_checkpoint(
        cls,
        save_dir:   Path,
        qwen_base:  nn.Module,
        bert_dim:   int,
        qwen_dim:   int,
        prefix_ids: torch.Tensor,
        middle_ids: torch.Tensor,
    ) -> "DiseaseAwareEHREncoder":
        qwen_lora = PeftModel.from_pretrained(qwen_base, str(save_dir / "lora"))
        encoder   = cls(qwen_lora, bert_dim, qwen_dim, prefix_ids, middle_ids)
        extra     = torch.load(save_dir / "extra_modules.pt", map_location="cpu")
        encoder.event_proj.load_state_dict(extra["event_proj"])
        encoder.disease_proj.load_state_dict(extra["disease_proj"])
        return encoder


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
    """Apply LoRA to the Qwen model; cast leftover fp32 params to bf16 if needed."""
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


def build_encoder(qwen_model, tokenizer, args) -> DiseaseAwareEHREncoder:
    """Construct DiseaseAwareEHREncoder and log total trainable parameter count."""
    qwen_dim = qwen_model.config.hidden_size

    prefix_ids = tokenizer(
        PROMPT_PREFIX, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]  # (1, P1)
    middle_ids = tokenizer(
        PROMPT_MIDDLE, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]  # (1, P2)

    encoder  = DiseaseAwareEHREncoder(
        qwen_model = qwen_model,
        bert_dim   = BERT_DIM,
        qwen_dim   = qwen_dim,
        prefix_ids = prefix_ids,
        middle_ids = middle_ids,
    )
    logger.info(
        f"  Prompt template: {repr(PROMPT_PREFIX)} <DISEASE> {repr(PROMPT_MIDDLE)} <EVENTS>"
    )
    logger.info(f"  disease_proj Linear({BERT_DIM}→{qwen_dim}), event_proj Linear({BERT_DIM}→{qwen_dim})")

    compute_dtype = (
        torch.bfloat16 if args.bf16 else
        torch.float16  if args.fp16 else
        torch.float32
    )
    if compute_dtype != torch.float32:
        encoder.event_proj.to(compute_dtype)
        encoder.disease_proj.to(compute_dtype)

    total     = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"  Total encoder trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)")
    return encoder, compute_dtype


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_ddp(
    encoder:          DiseaseAwareEHREncoder,
    val_dataset:      EHREmbeddingDataset,
    tokenizer,
    disease_emb_dict: dict[str, torch.Tensor],
    device:           torch.device,
    compute_dtype:    torch.dtype,
    args,
    rank:             int,
    world_size:       int,
    is_ddp:           bool,
) -> float:
    """Triplet accuracy distributed across all ranks.

    All ranks build the same triplet list (deterministic seed), then each rank
    encodes its own slice. Local correct/total counts are all_reduced.
    """
    raw_encoder = encoder.module if isinstance(encoder, DDP) else encoder
    raw_encoder.eval()

    # Build within-disease triplets: positives and negatives from the same task only.
    rng    = random.Random(args.seed)
    pos_by_task: dict[str, list] = defaultdict(list)
    neg_by_task: dict[str, list] = defaultdict(list)
    for sample, task, label in zip(val_dataset.samples, val_dataset.tasks, val_dataset.labels):
        (pos_by_task if label == 1 else neg_by_task)[task].append(sample)

    anchors_s, positives_s, negatives_s = [], [], []
    for task in sorted(pos_by_task):
        pos_samples = pos_by_task[task]
        neg_samples = neg_by_task.get(task, [])
        if len(pos_samples) < 2 or not neg_samples:
            continue
        pool = list(range(len(pos_samples)))
        rng.shuffle(pool)
        for i in range(min(args.n_eval_triplets_per_task, len(pos_samples))):
            ai = pool[i % len(pos_samples)]
            pi = rng.choice([j for j in pool if j != ai])
            ni = rng.randint(0, len(neg_samples) - 1)
            anchors_s.append(pos_samples[ai])
            positives_s.append(pos_samples[pi])
            negatives_s.append(neg_samples[ni])

    if not anchors_s:
        if rank == 0:
            logger.warning("No eval triplets could be built.")
        return 0.0

    all_samples = anchors_s + positives_s + negatives_s
    my_idx      = list(range(rank, len(all_samples), world_size))
    my_samples  = [all_samples[i] for i in my_idx]

    collate = make_collate_fn(disease_emb_dict)
    _embs   = val_dataset.lookup.embeddings  # mmap array, shared across workers

    def _resolve(s):
        event_embs = np.stack([_embs[i] for i in s["event_ids"]]).astype(np.float32)
        return {"event_embs": event_embs, "disease_name": s["disease_name"], "task": s["task"]}

    eval_dl = DataLoader(
        my_samples,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate([(_resolve(s), 0) for s in batch]),
        num_workers=4,
        pin_memory=True,
    )
    eval_prefetcher = CudaPrefetcher(
        AsyncDataLoader(eval_dl, buffer_size=2),
        device=device, cuda_keys=[],
    )

    all_embs = []
    for batch in eval_prefetcher:
        emb = raw_encoder(
            batch["event_embs"],
            batch["event_mask"],
            batch["disease_embs"],
            compute_dtype=compute_dtype,
        )
        all_embs.append(emb.cpu())

    n_my = len(my_idx) // 3
    if all_embs and n_my > 0:
        embs     = torch.cat(all_embs, dim=0)    # (3 * n_my, D)
        d_ap     = (embs[:n_my] - embs[n_my:2*n_my]).norm(dim=1)
        d_an     = (embs[:n_my] - embs[2*n_my:3*n_my]).norm(dim=1)
        local_correct = (d_ap < d_an).sum().to(torch.long).to(device)
        local_total   = torch.tensor(n_my, dtype=torch.long, device=device)
    else:
        local_correct = torch.tensor(0, dtype=torch.long, device=device)
        local_total   = torch.tensor(0, dtype=torch.long, device=device)

    if is_ddp:
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total,   op=dist.ReduceOp.SUM)

    acc = (local_correct.float() / local_total.float()).item() \
          if local_total.item() > 0 else 0.0
    raw_encoder.train()
    return acc


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Disease-aware EHR embedding via cross-attention conditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip training; only run evaluation.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a saved checkpoint dir (contains lora/ + extra_modules.pt).")

    # Data
    p.add_argument("--data_paths",     nargs="+", default=None,
                   help="Train parquet files (required unless --eval_only).")
    p.add_argument("--val_data_paths", nargs="+", default=None,
                   help="Validation parquet files.")
    p.add_argument("--val_split",      default="val", choices=["train", "val", "test"])

    # BioLinkBERT (event lookup)
    p.add_argument("--bert_model",      default="michiyasunaga/BioLinkBERT-base",
                   help="BioLinkBERT model path — used to encode disease names at startup.")
    p.add_argument("--bert_index",      required=True,
                   help="event_index.parquet produced by extract_biolinkbert_embeddings.py.")
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
    p.add_argument("--output_dir",   default="output/medical-embedding-disease-cond")
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
    p.add_argument("--triplet_margin",           type=float, default=0.5)
    p.add_argument("--n_eval_triplets_per_task", type=int,   default=100)
    p.add_argument("--eval_batch_size",          type=int,   default=32)

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

    # ── BioLinkBERT event lookup (shared across ranks — read-only mmap) ────────
    lookup = EventLookup(args.bert_index, args.bert_embeddings)

    # ── Pre-compute disease embeddings with BioLinkBERT (rank 0 then broadcast) ─
    # All ranks need the same disease_emb_dict; rank 0 computes, others receive.
    disease_names  = list(TASK_2_DISEASE_NAME.values())
    if rank == 0:
        disease_emb_dict = encode_disease_names(args.bert_model, disease_names, device)
        # Stack for broadcast: (6, 768)
        disease_emb_tensor = torch.stack([disease_emb_dict[n] for n in disease_names])
    else:
        disease_emb_tensor = torch.zeros(len(disease_names), BERT_DIM, dtype=torch.float32)

    if is_ddp:
        disease_emb_tensor = disease_emb_tensor.to(device)
        dist.broadcast(disease_emb_tensor, src=0)
        disease_emb_tensor = disease_emb_tensor.cpu()
        disease_emb_dict   = {n: disease_emb_tensor[i] for i, n in enumerate(disease_names)}

    # ── Data loading ───────────────────────────────────────────────────────────
    my_train_paths = []
    if not args.eval_only:
        my_train_paths = args.data_paths[rank :: world_size]
        if not my_train_paths:
            raise ValueError(f"Rank {rank}: no data paths assigned.")
        logger.info(f"Rank {rank} train paths: {my_train_paths}")

    # ── Build Qwen model + tokenizer ───────────────────────────────────────────
    if rank == 0:
        logger.info(f"Loading Qwen: {args.model_name}")
    qwen_model, tokenizer = load_qwen(args)

    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        qwen_model.config.use_cache = False
        qwen_lora = PeftModel.from_pretrained(qwen_model, str(Path(args.checkpoint) / "lora"))
        encoder, compute_dtype = build_encoder(qwen_lora, tokenizer, args)
        extra = torch.load(Path(args.checkpoint) / "extra_modules.pt", map_location="cpu")
        encoder.event_proj.load_state_dict(extra["event_proj"])
        encoder.disease_proj.load_state_dict(extra["disease_proj"])
    else:
        qwen_lora = setup_lora(qwen_model, args)
        encoder, compute_dtype = build_encoder(qwen_lora, tokenizer, args)

    encoder = encoder.to(device)

    if is_ddp and not args.eval_only:
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=False)

    # ── Eval-only mode ─────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.val_data_paths:
            raise ValueError("--eval_only requires --val_data_paths.")
        val_dataset = EHREmbeddingDataset(args.val_data_paths, lookup, split=args.val_split)
        val_acc = evaluate_ddp(
            encoder, val_dataset, tokenizer, disease_emb_dict,
            device, compute_dtype, args, rank, world_size, is_ddp,
        )
        if rank == 0:
            logger.info(f"Eval triplet accuracy: {val_acc:.4f}")
        if use_wandb and wandb_run:
            import wandb
            wandb.log({"eval/triplet_acc": val_acc})
            wandb_run.finish()
        if is_ddp:
            dist.destroy_process_group()
        return

    # ── Datasets & DataLoaders ─────────────────────────────────────────────────
    train_dataset = EHREmbeddingDataset(my_train_paths, lookup, split="train")
    sampler = SingleDiseaseGroupSampler(
        tasks=train_dataset.tasks,
        labels=train_dataset.labels,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    collate = make_collate_fn(disease_emb_dict)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=1
    )

    val_dataset = None
    if args.val_data_paths:
        val_dataset = EHREmbeddingDataset(args.val_data_paths, lookup, split=args.val_split)

    # ── Optimizer + LR scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    n_batches_per_epoch   = len(train_loader)
    n_opt_steps_per_epoch = math.ceil(n_batches_per_epoch / args.grad_accum)
    total_opt_steps       = n_opt_steps_per_epoch * args.epochs
    warmup_steps          = int(total_opt_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_module = BatchAllTripletLoss(model=None, margin=args.triplet_margin)

    if rank == 0:
        logger.info(f"Training: {args.epochs} epochs, "
                    f"{n_batches_per_epoch} batches/epoch, "
                    f"{total_opt_steps} optimizer steps total")
        logger.info(f"BatchAllTripletLoss margin={args.triplet_margin}")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    opt_step     = 0

    for epoch in range(args.epochs):
        if is_ddp:
            dist.barrier()
        sampler.set_epoch(epoch)
        encoder.train()

        epoch_loss       = 0.0
        optimizer.zero_grad()

        prefetcher = CudaPrefetcher(
            AsyncDataLoader(train_loader, buffer_size=1),
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
                # Call through the DDP wrapper (not encoder.module) so that
                # reducer.prepare_for_backward() fires and no_sync() is honoured.
                emb = encoder(
                    batch["event_embs"],
                    batch["event_mask"],
                    batch["disease_embs"],
                    compute_dtype=compute_dtype,
                )
                loss = loss_module.batch_all_triplet_loss(labels_t, emb)
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
        if val_dataset is not None:
            if is_ddp:
                dist.barrier()
            val_acc = evaluate_ddp(
                encoder, val_dataset, tokenizer, disease_emb_dict,
                device, compute_dtype, args, rank, world_size, is_ddp,
            )
            if rank == 0:
                logger.info(f"  val triplet accuracy: {val_acc:.4f}")
                if use_wandb:
                    import wandb
                    wandb.log({
                        "val/triplet_acc":  val_acc,
                        "train/epoch_loss": avg_loss,
                        "epoch":            epoch + 1,
                    }, step=opt_step)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
                    raw_enc.save_checkpoint(run_dir / "best")
                    logger.info(f"  ↑ New best: {best_val_acc:.4f}")

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
            wandb.summary["best_val_triplet_acc"] = best_val_acc
            wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
