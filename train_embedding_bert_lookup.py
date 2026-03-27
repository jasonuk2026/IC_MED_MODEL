#!/usr/bin/env python3
"""
train_embedding_bert_lookup.py

Variant of train_embedding_custom.py where each event in a patient history is
replaced by its pre-computed BioLinkBERT embedding (from
extract_biolinkbert_embeddings.py) instead of being tokenised as text.

Architecture per sample:
  task prompt text  →  Qwen tokeniser  →  Qwen embed_tokens  ─┐
                                                                ├─→ cat → Qwen layers → last-token pool → triplet loss
  [event_1, ..., event_N]                                      │
    → BioLinkBERT emb lookup (768-d, fp32)                     │
    → ProjectionLayer (Linear 768 → 1024)  ────────────────────┘

Benefits:
  - Context length = (prompt tokens) + (N events)  instead of all event tokens
  - Each event appears exactly once; no stochastic re-sampling needed

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 train_embedding_bert_lookup.py \\
        --data_paths data/embedding_inputs/sharded_m500/train_shard_*.parquet \\
        --val_data_paths data/embedding_inputs/sharded_m500/val.parquet \\
        --bert_index data/biolinkbert_embeddings/event_index.parquet \\
        --bert_embeddings data/biolinkbert_embeddings/embeddings.npy \\
        --bf16

Single GPU:
    python train_embedding_bert_lookup.py --data_paths all.parquet ...
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

import pandas as pd
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, logging as hf_logging
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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

BERT_DIM = 768   # BioLinkBERT-base hidden size


# ── Event text key (must match extract_biolinkbert_embeddings.py) ─────────────

def event_to_text(e: dict) -> str | None:
    """Format a single event dict to the same text key used during extraction."""
    desc  = (e.get("description") or "").strip()
    code  = (e.get("code")        or "").strip()
    value = (e.get("value")       or "").strip()
    unit  = (e.get("unit")        or "").strip()

    if desc and code:
        text = f"{desc} [{code}]"
    elif code:
        text = f"[{code}]"
    elif desc:
        text = desc
    else:
        return None
    if value:
        text += f" | value={value}"
    if unit:
        text += f" | unit={unit}"
    return text


def build_prompt_prefix(disease_name: str) -> str:
    return (
        f"Please predict disease {disease_name} based on the following events.\n"
        "Start of medical events:"
    )


def build_label_map(tasks: list) -> dict:
    label_map = {}
    for idx, task in enumerate(sorted(set(tasks))):
        label_map[(task, False)] = idx * 2
        label_map[(task, True)]  = idx * 2 + 1
    return label_map


# ── BioLinkBERT embedding lookup ──────────────────────────────────────────────

class EventLookup:
    """Memory-mapped lookup from event text → BioLinkBERT embedding (fp32)."""

    def __init__(self, index_path: str, embeddings_path: str):
        logger.info(f"Loading event index from {index_path} ...")
        index_df = pd.read_parquet(index_path)
        self.text2idx: dict[str, int] = dict(
            zip(index_df["event_text"], index_df["event_id"])
        )
        logger.info(f"  {len(self.text2idx):,} unique event texts indexed.")

        logger.info(f"Loading embeddings (mmap) from {embeddings_path} ...")
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        logger.info(f"  Embeddings shape: {self.embeddings.shape}  dtype: {self.embeddings.dtype}")

    def get(self, text: str) -> np.ndarray:
        idx = self.text2idx.get(text)
        if idx is None:
            raise KeyError(
                f"Event text not found in BioLinkBERT index — this indicates a bug "
                f"in extract_biolinkbert_embeddings.py or a mismatch in event_to_text().\n"
                f"  Missing text: {text!r}"
            )
        return self.embeddings[idx].copy()   # copy off mmap to avoid stale refs


# ── Dataset ───────────────────────────────────────────────────────────────────

class EHREmbeddingDataset(Dataset):
    """Each sample: list of BioLinkBERT event embeddings + disease label."""

    def __init__(self, df: pd.DataFrame, lookup: EventLookup, max_events: int):
        label_map = build_label_map(df["task"].tolist())
        self.samples: list[dict] = []
        self.labels:  list[int]  = []
        skipped = 0

        for task, events, label in zip(df["task"], df["events"], df["label"]):
            emb_list = []
            for e in events:
                text = event_to_text(e)
                if text is None:
                    continue
                emb_list.append(lookup.get(text))   # raises KeyError if missing
                if len(emb_list) >= max_events:
                    break

            if not emb_list:
                skipped += 1
                continue

            disease_name = TASK_2_DISEASE_NAME.get(task, task)
            self.samples.append({
                "event_embs":   np.stack(emb_list).astype(np.float32),  # (n, 768)
                "disease_name": disease_name,
                "task":         task,
            })
            self.labels.append(label_map[(task, bool(label))])

        if skipped:
            logger.warning(f"  Skipped {skipped} samples with no valid events (event_to_text returned None for all).")
        counts = Counter(self.labels)
        logger.info(f"  {len(self.samples)} samples, {len(counts)} label classes: "
                    f"{dict(sorted(counts.items()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ── Batch sampler (identical to train_embedding_custom.py) ────────────────────

class GroupByLabelSampler:
    def __init__(self, labels: list, batch_size: int, drop_last: bool = True, seed: int = 42):
        if batch_size < 4 or batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even and ≥4, got {batch_size}")
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.seed       = seed
        self.epoch      = 0

        groups: dict = defaultdict(list)
        for idx, lbl in enumerate(labels):
            groups[lbl].append(idx)

        self.groups = {
            lbl: idxs[: len(idxs) // 2 * 2]
            for lbl, idxs in groups.items() if len(idxs) >= 2
        }
        if len(self.groups) < 2:
            raise ValueError(f"Need ≥2 labels with ≥2 samples each, got {len(self.groups)}")

        pairs = sorted((len(v) // 2 for v in self.groups.values()), reverse=True)
        cap = pairs[1]
        self._stream_len = 2 * sum(min(p, cap) for p in pairs)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        queues: dict = {}
        for lbl, idxs in self.groups.items():
            perm = torch.randperm(len(idxs), generator=g).tolist()
            queues[lbl] = deque(idxs[i] for i in perm)

        remaining = list(queues)
        batch: list = []
        while len(remaining) >= 2:
            order = torch.randperm(len(remaining), generator=g).tolist()
            remaining = [remaining[i] for i in order]
            for lbl in remaining:
                batch.append(queues[lbl].popleft())
                batch.append(queues[lbl].popleft())
                if len(batch) >= self.batch_size:
                    yield batch[: self.batch_size]
                    batch = batch[self.batch_size :]
            remaining = [l for l in remaining if len(queues[l]) >= 2]

        if not self.drop_last and len(batch) >= 4:
            yield batch

    def __len__(self):
        n = self._stream_len // self.batch_size
        if not self.drop_last and self._stream_len % self.batch_size >= 4:
            n += 1
        return n


# ── Collate ───────────────────────────────────────────────────────────────────

def make_collate_fn(tokenizer, max_prompt_len: int):
    """Returns a collate_fn that tokenises prompts and pads event embeddings."""
    # Pre-tokenise all 6 disease prompts and cache them
    prompt_cache: dict[str, torch.Tensor] = {}

    def collate_fn(batch):
        samples = [b[0] for b in batch]
        labels  = [b[1] for b in batch]

        # ── Tokenise task prompts ──────────────────────────────────────────
        prompt_ids_list = []
        for s in samples:
            key = s["disease_name"]
            if key not in prompt_cache:
                prefix = build_prompt_prefix(key)
                ids = tokenizer(
                    prefix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_prompt_len,
                    return_tensors="pt",
                )["input_ids"].squeeze(0)   # (prompt_len,)
                prompt_cache[key] = ids
            prompt_ids_list.append(prompt_cache[key])

        # Pad prompts to same length (left-pad so event embeddings are always at the right)
        max_p = max(t.shape[0] for t in prompt_ids_list)
        pad_id = tokenizer.pad_token_id or 0
        padded_prompt_ids  = torch.full((len(samples), max_p), pad_id, dtype=torch.long)
        prompt_attn_mask   = torch.zeros(len(samples), max_p, dtype=torch.long)
        for i, ids in enumerate(prompt_ids_list):
            padded_prompt_ids[i, -ids.shape[0]:] = ids
            prompt_attn_mask[i, -ids.shape[0]:]  = 1

        # ── Pad event embedding sequences ─────────────────────────────────
        event_embs_list = [s["event_embs"] for s in samples]   # list of (n_i, 768)
        max_e = max(e.shape[0] for e in event_embs_list)
        bert_dim = event_embs_list[0].shape[1]

        padded_event_embs = torch.zeros(len(samples), max_e, bert_dim, dtype=torch.float32)
        event_attn_mask   = torch.zeros(len(samples), max_e, dtype=torch.long)
        for i, embs in enumerate(event_embs_list):
            n = embs.shape[0]
            padded_event_embs[i, :n] = torch.from_numpy(embs)
            event_attn_mask[i, :n]   = 1

        return {
            "prompt_ids":   padded_prompt_ids,   # (B, max_p)
            "prompt_mask":  prompt_attn_mask,     # (B, max_p)
            "event_embs":   padded_event_embs,    # (B, max_e, 768)
            "event_mask":   event_attn_mask,      # (B, max_e)
            "labels":       torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


# ── Projection layer ──────────────────────────────────────────────────────────

class BertProjection(nn.Module):
    """Linear projection from BioLinkBERT hidden size to Qwen hidden size."""

    def __init__(self, bert_dim: int, qwen_dim: int):
        super().__init__()
        self.proj = nn.Linear(bert_dim, qwen_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ── Model setup ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer(args):
    kwargs = {}
    if args.flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if args.qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        args.bf16 = True
    elif args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        kwargs["torch_dtype"] = torch.float16

    model     = AutoModel.from_pretrained(args.model_name, local_files_only=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True,
                                              padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    if args.bf16:
        for name, param in model.named_parameters():
            if param.dtype == torch.float32 and "lora" not in name:
                param.data = param.data.to(torch.bfloat16)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Qwen trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


# ── Forward helpers ───────────────────────────────────────────────────────────

def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx   = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, seq_lengths]


EOS_TOKEN_ID = 151643   # Qwen3 <|endoftext|>, used as the pooling token


def forward_batch(
    qwen_model,
    projection: BertProjection,
    batch: dict,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Build inputs_embeds from prompt tokens + projected BioLinkBERT embeddings
    + <|endoftext|> pooling token, run Qwen, last-token pool, L2-normalise → (B, H).

    Sequence layout (mirrors train_embedding_custom.py):
        [prompt tokens ...] [event_1] ... [event_N] [<|endoftext|>]
    last_token_pool picks the <|endoftext|> position, which is where Qwen3-Embedding
    places its summary representation.
    """

    prompt_ids  = batch["prompt_ids"].to(device)    # (B, max_p)
    prompt_mask = batch["prompt_mask"].to(device)   # (B, max_p)
    event_embs  = batch["event_embs"].to(device)    # (B, max_e, 768)
    event_mask  = batch["event_mask"].to(device)    # (B, max_e)

    B = prompt_ids.size(0)

    # Get Qwen's token embedding table
    raw_model    = qwen_model.module if isinstance(qwen_model, DDP) else qwen_model
    embed_tokens = raw_model.get_input_embeddings()

    # Prompt embeddings from Qwen's vocab
    prompt_embeds = embed_tokens(prompt_ids).to(compute_dtype)     # (B, max_p, H)

    # Project BioLinkBERT embeddings to Qwen dim
    proj_event_embs = projection(event_embs.to(compute_dtype))     # (B, max_e, H)

    # <|endoftext|> pooling token — one per sample, always attended to
    eos_id     = torch.full((B, 1), EOS_TOKEN_ID, dtype=torch.long, device=device)
    eos_embeds = embed_tokens(eos_id).to(compute_dtype)             # (B, 1, H)
    eos_mask   = torch.ones(B, 1, dtype=torch.long, device=device)

    # Concatenate: [prompt | events | <|endoftext|>]
    inputs_embeds  = torch.cat([prompt_embeds, proj_event_embs, eos_embeds], dim=1)
    attention_mask = torch.cat([prompt_mask,   event_mask,      eos_mask  ], dim=1)

    out = qwen_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    emb = last_token_pool(out.last_hidden_state, attention_mask)
    return F.normalize(emb.float(), p=2, dim=-1)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_ddp(
    qwen_model, projection, val_dataset: EHREmbeddingDataset,
    tokenizer, device, args, rank: int, world_size: int, is_ddp: bool,
) -> float:
    raw_model = qwen_model.module if isinstance(qwen_model, DDP) else qwen_model
    raw_model.eval()
    projection.eval()

    # Build triplets: same structure as build_eval_triplets in train_embedding_custom.py
    rng = random.Random(args.seed)
    groups: dict = defaultdict(list)
    for sample, label in zip(val_dataset.samples, val_dataset.labels):
        groups[label].append(sample)

    # Pair labels by task: even idx = negative, odd idx = positive
    task_labels = sorted(set(val_dataset.labels))
    anchors_s, positives_s, negatives_s = [], [], []
    for lbl in task_labels:
        if lbl % 2 == 0:
            continue   # skip negatives as anchor pool
        pos_samples = groups[lbl]
        neg_samples = [s for other_lbl, ss in groups.items()
                       if other_lbl != lbl for s in ss]
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

    # Encode in batches
    collate = make_collate_fn(tokenizer, args.max_prompt_len)
    compute_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    all_embs = []
    for i in range(0, len(my_samples), args.eval_batch_size):
        mini = my_samples[i : i + args.eval_batch_size]
        fake_labels = [0] * len(mini)
        batch = collate(list(zip(mini, fake_labels)))
        emb = forward_batch(qwen_model, projection, batch, device, compute_dtype)
        all_embs.append(emb)

    n_my = len(my_idx) // 3
    if all_embs and n_my > 0:
        embs = torch.cat(all_embs, dim=0)
        d_ap = (embs[:n_my] - embs[n_my:2*n_my]).norm(dim=1)
        d_an = (embs[:n_my] - embs[2*n_my:]).norm(dim=1)
        local_correct = (d_ap < d_an).sum().to(torch.long).to(device)
        local_total   = torch.tensor(n_my, dtype=torch.long, device=device)
    else:
        local_correct = torch.tensor(0, dtype=torch.long, device=device)
        local_total   = torch.tensor(0, dtype=torch.long, device=device)

    if is_ddp:
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total,   op=dist.ReduceOp.SUM)

    acc = (local_correct.float() / local_total.float()).item() if local_total.item() > 0 else 0.0
    raw_model.train()
    projection.train()
    return acc


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(qwen_model, projection, tokenizer, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    raw = qwen_model.module if isinstance(qwen_model, DDP) else qwen_model
    raw.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    torch.save(projection.state_dict(), save_dir / "projection.pt")
    logger.info(f"  Saved checkpoint → {save_dir}")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument("--data_paths",     nargs="+", default=None)
    p.add_argument("--val_data_paths", nargs="+", default=None)
    p.add_argument("--val_split",      default="val", choices=["train", "val", "test"])

    # BioLinkBERT embedding lookup
    p.add_argument("--bert_index",      default="data/biolinkbert_embeddings/event_index.parquet")
    p.add_argument("--bert_embeddings", default="data/biolinkbert_embeddings/embeddings.npy")
    p.add_argument("--max_events",      type=int, default=500,
                   help="Max events per sample (truncate if longer).")
    p.add_argument("--max_prompt_len",  type=int, default=32,
                   help="Max tokens for the task prompt prefix.")

    # Model
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
    p.add_argument("--output_dir",   default="output/medical-embedding-bert-lookup")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--proj_lr",      type=float, default=1e-3,
                   help="Learning rate for the projection layer (typically higher than LoRA lr).")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--log_steps",    type=int,   default=10)

    # Loss / eval
    p.add_argument("--triplet_margin",           type=float, default=0.5)
    p.add_argument("--n_eval_triplets_per_task", type=int,   default=100)
    p.add_argument("--eval_batch_size",          type=int,   default=4)

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
        raise RuntimeError("QLoRA is not compatible with multi-GPU DDP.")
    if not args.data_paths:
        raise ValueError("--data_paths is required.")

    compute_dtype = (torch.bfloat16 if args.bf16
                     else torch.float16 if args.fp16
                     else torch.float32)

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
        )
        wandb.save(os.path.abspath(__file__), base_path=os.path.dirname(os.path.abspath(__file__)))
        logger.info(f"wandb run: {wandb_run.url}")

    # ── BioLinkBERT lookup (all ranks share, mmap = no extra RAM per rank) ────
    lookup = EventLookup(args.bert_index, args.bert_embeddings)

    # ── Data ──────────────────────────────────────────────────────────────────
    my_train_paths = args.data_paths[rank :: world_size]
    logger.info(f"Rank {rank} train paths: {my_train_paths}")
    train_df = pd.concat(
        [pd.read_parquet(p).pipe(lambda df: df[df["split"] == "train"])
         for p in my_train_paths],
        ignore_index=True,
    )
    logger.info(f"Rank {rank}: {len(train_df)} train rows")

    val_dataset = None
    if args.val_data_paths:
        val_df = pd.concat(
            [pd.read_parquet(p).pipe(lambda df: df[df["split"] == args.val_split])
             for p in args.val_data_paths],
            ignore_index=True,
        )
        if rank == 0:
            logger.info(f"Val rows: {len(val_df)}")
        val_dataset = EHREmbeddingDataset(val_df, lookup, args.max_events)

    train_dataset = EHREmbeddingDataset(train_df, lookup, args.max_events)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model_name}")
    qwen_model, tokenizer = load_model_and_tokenizer(args)
    qwen_model = setup_lora(qwen_model, args)

    qwen_dim   = qwen_model.config.hidden_size
    projection = BertProjection(BERT_DIM, qwen_dim)

    qwen_model = qwen_model.to(device)
    projection = projection.to(device)
    if args.bf16:
        projection = projection.to(torch.bfloat16)
    elif args.fp16:
        projection = projection.to(torch.float16)

    if is_ddp:
        qwen_model = DDP(qwen_model, device_ids=[local_rank], output_device=local_rank,
                         find_unused_parameters=False)
        projection = DDP(projection, device_ids=[local_rank], output_device=local_rank,
                         find_unused_parameters=False)

    proj_params  = list((projection.module if is_ddp else projection).parameters())
    lora_params  = [p for p in qwen_model.parameters() if p.requires_grad]
    proj_trainable = sum(p.numel() for p in proj_params)
    logger.info(f"  Projection trainable params: {proj_trainable:,} "
                f"({BERT_DIM}→{qwen_dim})")

    # ── DataLoader ────────────────────────────────────────────────────────────
    collate_fn = make_collate_fn(tokenizer, args.max_prompt_len)
    sampler = GroupByLabelSampler(
        labels=train_dataset.labels, batch_size=args.batch_size, drop_last=True, seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
    )

    # ── Optimizer: two param groups (projection gets higher lr) ───────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params,  "lr": args.lr},
            {"params": proj_params,  "lr": args.proj_lr},
        ],
        weight_decay=args.weight_decay,
    )

    n_batches_per_epoch   = len(train_loader)
    n_opt_steps_per_epoch = math.ceil(n_batches_per_epoch / args.grad_accum)
    total_opt_steps       = n_opt_steps_per_epoch * args.epochs
    warmup_steps          = int(total_opt_steps * args.warmup_ratio)

    def lr_lambda(opt_step: int) -> float:
        if opt_step < warmup_steps:
            return opt_step / max(warmup_steps, 1)
        progress = (opt_step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)

    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_module  = BatchAllTripletLoss(model=None, margin=args.triplet_margin)

    logger.info(f"Training: {args.epochs} epochs, "
                f"{n_batches_per_epoch} batches/epoch, "
                f"{total_opt_steps} optimizer steps total")
    logger.info(f"BatchAllTripletLoss margin={args.triplet_margin}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    opt_step     = 0

    for epoch in range(args.epochs):
        if is_ddp:
            dist.barrier()
        sampler.set_epoch(epoch)
        qwen_model.train()
        projection.train()

        epoch_loss       = 0.0
        n_opt_this_epoch = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=(rank != 0), dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            labels_t = batch["labels"].to(device)

            is_update_step = (batch_idx + 1) % args.grad_accum == 0 \
                             or (batch_idx + 1) == n_batches_per_epoch
            sync_ctx = nullcontext() if (is_update_step or not is_ddp) \
                       else qwen_model.no_sync()

            with sync_ctx:
                emb  = forward_batch(qwen_model, projection, batch, device, compute_dtype)
                loss = loss_module.batch_all_triplet_loss(labels_t, emb)
                loss = loss / args.grad_accum
                loss.backward()

            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    lora_params + proj_params, args.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step         += 1
                n_opt_this_epoch += 1

                if rank == 0:
                    lr_now   = scheduler.get_last_lr()[0]
                    loss_now = loss.item() * args.grad_accum
                    pbar.set_postfix(
                        loss=f"{loss_now:.4f}",
                        gnorm=f"{grad_norm.item():.3f}",
                        lr=f"{lr_now:.2e}",
                    )
                    if use_wandb and opt_step % args.log_steps == 0:
                        import wandb
                        wandb.log({
                            "train/loss":  loss_now,
                            "train/gnorm": grad_norm.item(),
                            "train/lr":    lr_now,
                            "epoch":       epoch + (batch_idx + 1) / n_batches_per_epoch,
                        }, step=opt_step)

            epoch_loss += loss.item() * args.grad_accum

        avg_loss = epoch_loss / n_batches_per_epoch
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")

        # ── Evaluation ────────────────────────────────────────────────────────
        if val_dataset is not None:
            if is_ddp:
                dist.barrier()
            val_acc = evaluate_ddp(
                qwen_model, projection, val_dataset,
                tokenizer, device, args, rank, world_size, is_ddp,
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
                    save_checkpoint(qwen_model, projection, tokenizer, run_dir / "best")
                    logger.info(f"  ↑ New best: {best_val_acc:.4f}")

        if rank == 0:
            save_checkpoint(qwen_model, projection, tokenizer, run_dir / f"epoch_{epoch+1}")

        if is_ddp:
            dist.barrier()

    if rank == 0:
        save_checkpoint(qwen_model, projection, tokenizer, run_dir / "final")
        logger.info(f"Done. Best val triplet accuracy: {best_val_acc:.4f}")
        if use_wandb and wandb_run:
            import wandb
            wandb.summary["best_val_triplet_acc"] = best_val_acc
            wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
