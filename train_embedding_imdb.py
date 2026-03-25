#!/usr/bin/env python3
"""
train_embedding_imdb.py — Same training loop as train_embedding_custom.py,
but using the Stanford IMDB sentiment dataset instead of EHR parquet files.

Purpose: sanity-check that the custom training infrastructure actually works
on a well-understood benchmark before diagnosing EHR-specific issues.

Dataset: stanfordnlp/imdb
  - 25,000 train reviews  (label 0=neg, 1=pos)
  - 25,000 test  reviews  (used as validation here)
  - Plain text, binary labels

Usage:
  # Single GPU
  CUDA_VISIBLE_DEVICES=0 python train_embedding_imdb.py --bf16 --flash_attn

  # Limit data for a quick smoke-test
  CUDA_VISIBLE_DEVICES=0 python train_embedding_imdb.py --bf16 --max_train 2000 --max_val 500

  # Multi-GPU DDP
  torchrun --nproc_per_node=2 train_embedding_imdb.py --bf16 --flash_attn
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

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers.losses import BatchAllTripletLoss

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# ── Dataset ───────────────────────────────────────────────────────────────────

class IMDBDataset(Dataset):
    def __init__(self, hf_split, max_samples: int = None, seed: int = 42):
        data = list(hf_split)
        if max_samples and len(data) > max_samples:
            rng = random.Random(seed)
            data = rng.sample(data, max_samples)
        self.texts:  list[str] = [d["text"]  for d in data]
        self.labels: list[int] = [d["label"] for d in data]
        counts = Counter(self.labels)
        logger.info(f"  {len(self.texts)} samples, label distribution: {dict(sorted(counts.items()))}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ── Batch sampler ─────────────────────────────────────────────────────────────

class GroupByLabelSampler:
    """Produces batches with multiple labels × ≥2 samples per label.

    Standalone port of sentence-transformers' GroupByLabelBatchSampler.
    No DDP awareness needed — data is already split by rank before this runs.

    Algorithm: shuffle samples within each label, then interleave labels in
    round-robin (2 samples per label per round) until fewer than 2 labels
    remain.  The stream is chunked into batches of exactly `batch_size`.
    """

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

        # Keep only labels with ≥2 samples; trim to even length for clean pairs
        self.groups = {
            lbl: idxs[: len(idxs) // 2 * 2]
            for lbl, idxs in groups.items() if len(idxs) >= 2
        }
        if len(self.groups) < 2:
            raise ValueError(
                f"Need ≥2 labels with ≥2 samples each, got {len(self.groups)}"
            )

        # Pre-compute approximate stream length for __len__
        pairs = sorted((len(v) // 2 for v in self.groups.values()), reverse=True)
        cap = pairs[1]  # round-robin stops when only 1 label remains
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
            # Shuffle label visit order each round for diverse batches
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


def collate_fn(batch):
    texts  = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    return texts, labels


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
        kwargs["dtype"] = torch.bfloat16
    elif args.fp16:
        kwargs["dtype"] = torch.float16

    model     = AutoModel.from_pretrained(args.model_name, local_files_only=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True, padding_side="right")
    return model, tokenizer


def setup_model(model, tokenizer, args):
    """Attach LoRA (no special tokens needed for plain-text IMDB)."""
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    if args.qlora:
        prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    modules_to_save = []
    if args.train_norm:
        norm_module_names = [name.rsplit(".weight", 1)[0]
                             for name, _ in model.named_parameters()
                             if "norm.weight" in name]
        modules_to_save += norm_module_names
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save if modules_to_save else None,
    ))

    # Cast any fp32 params left behind
    if args.bf16:
        for name, param in model.named_parameters():
            if param.dtype == torch.float32 and "lora" not in name:
                param.data = param.data.to(torch.bfloat16)

    # ── Gradient checkpointing ────────────────────────────────────────────────
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  requires_grad params : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


# ── Encoding ──────────────────────────────────────────────────────────────────

def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-token pooling for decoder models with right-padding."""
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx   = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, seq_lengths]


@torch.inference_mode()
def encode_texts(model, tokenizer, texts: list, device, batch_size: int) -> torch.Tensor:
    """Encode texts to L2-normalised embeddings without gradients (for eval)."""
    all_emb = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, padding=True, return_tensors="pt").to(device)
        out = model(**enc)
        emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb.float(), p=2, dim=-1)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ── Evaluation ────────────────────────────────────────────────────────────────

def build_eval_triplets(val_dataset: IMDBDataset, n_triplets: int, seed: int):
    """Build (anchor, positive, negative) triplets from the val set."""
    rng    = random.Random(seed)
    groups = defaultdict(list)
    for text, label in zip(val_dataset.texts, val_dataset.labels):
        groups[label].append(text)

    pos = groups[1]   # positive reviews
    neg = groups[0]   # negative reviews
    if len(pos) < 2 or len(neg) < 1:
        return [], [], []

    n = min(n_triplets, len(pos))
    anchors, positives, negatives = [], [], []
    pool = list(range(len(pos)))
    rng.shuffle(pool)
    for i in range(n):
        ai = pool[i % len(pos)]
        pi = rng.choice([j for j in pool if j != ai])
        ni = rng.randint(0, len(neg) - 1)
        anchors.append(pos[ai])
        positives.append(pos[pi])
        negatives.append(neg[ni])

    # Also add triplets with neg as anchor (symmetric)
    pool_neg = list(range(len(neg)))
    rng.shuffle(pool_neg)
    for i in range(n):
        ai = pool_neg[i % len(neg)]
        pi = rng.choice([j for j in pool_neg if j != ai])
        ni = rng.randint(0, len(pos) - 1)
        anchors.append(neg[ai])
        positives.append(neg[pi])
        negatives.append(pos[ni])

    return anchors, positives, negatives


def evaluate(model, tokenizer, val_dataset: IMDBDataset, device, args) -> float:
    """Triplet accuracy: fraction where d(anchor, positive) < d(anchor, negative)."""
    anchors, positives, negatives = build_eval_triplets(
        val_dataset, n_triplets=args.n_eval_triplets, seed=args.seed
    )
    if not anchors:
        logger.warning("No eval triplets could be built.")
        return 0.0

    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    all_texts = anchors + positives + negatives
    embs = encode_texts(raw_model, tokenizer, all_texts, device, batch_size=args.eval_batch_size)
    n     = len(anchors)
    d_ap  = (embs[:n] - embs[n:2*n]).norm(dim=1)
    d_an  = (embs[:n] - embs[2*n:]).norm(dim=1)
    acc   = (d_ap < d_an).float().mean().item()

    raw_model.train()
    return acc


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    raw.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info(f"  Saved checkpoint → {save_dir}")


# ── Arg parse ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument("--dataset",      default="stanfordnlp/imdb",
                   help="HuggingFace dataset identifier.")
    p.add_argument("--max_train",    type=int, default=None,
                   help="Randomly subsample train set (useful for quick tests).")
    p.add_argument("--max_val",      type=int, default=2000,
                   help="Randomly subsample val (test) set.")

    # Model
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
    p.add_argument("--train_norm",          action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Training
    p.add_argument("--output_dir",   default="output/imdb-embedding")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)

    # Loss / Eval
    p.add_argument("--triplet_margin",   type=float, default=0.5)
    p.add_argument("--n_eval_triplets",  type=int,   default=200)
    p.add_argument("--eval_batch_size",  type=int,   default=8)

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

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run dir: {run_dir}")
        logger.info(f"World size: {world_size}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    raw = load_dataset(args.dataset)

    # DDP: split train samples across ranks by index
    train_split = raw["train"]
    all_indices = list(range(len(train_split)))
    my_indices  = all_indices[rank :: world_size]
    my_train_split = train_split.select(my_indices)
    logger.info(f"Rank {rank}: {len(my_train_split)} train samples (of {len(train_split)} total)")

    train_dataset = IMDBDataset(my_train_split, max_samples=args.max_train, seed=args.seed)

    val_dataset = None
    if rank == 0:
        val_dataset = IMDBDataset(raw["test"], max_samples=args.max_val, seed=args.seed)

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args)
    model = setup_model(model, tokenizer, args)
    model = model.to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # ── DataLoader ─────────────────────────────────────────────────────────────
    sampler = GroupByLabelSampler(
        labels=train_dataset.labels,
        batch_size=args.batch_size,
        drop_last=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # ── Optimizer + LR scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
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

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ──────────────────────────────────────────────────────────
    logger.info(f"Training: {args.epochs} epochs, "
                f"{n_batches_per_epoch} batches/epoch, "
                f"{total_opt_steps} optimizer steps total")

    loss_module = BatchAllTripletLoss(model=None, margin=args.triplet_margin)
    logger.info(f"BatchAllTripletLoss margin={args.triplet_margin}")

    best_val_acc = 0.0
    opt_step     = 0

    for epoch in range(args.epochs):
        if is_ddp:
            dist.barrier()
        sampler.set_epoch(epoch)
        model.train()

        epoch_loss       = 0.0
        n_opt_this_epoch = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=(rank != 0), dynamic_ncols=True)
        for batch_idx, (texts, labels) in enumerate(pbar):

            enc = tokenizer(
                texts, padding=True, truncation=False,
                return_tensors="pt",
            ).to(device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            is_update_step = (batch_idx + 1) % args.grad_accum == 0 \
                             or (batch_idx + 1) == n_batches_per_epoch
            sync_ctx = nullcontext() if (is_update_step or not is_ddp) \
                       else model.no_sync()

            with sync_ctx:
                out  = model(**enc)
                emb  = last_token_pool(out.last_hidden_state, enc["attention_mask"])
                emb  = F.normalize(emb, p=2, dim=-1)
                loss = loss_module.batch_all_triplet_loss(labels_t, emb)
                loss = loss / args.grad_accum

                loss.backward()

            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    args.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step         += 1
                n_opt_this_epoch += 1

            epoch_loss += loss.item() * args.grad_accum

            if rank == 0 and is_update_step:
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}",
                                 gnorm=f"{grad_norm:.3f}", lr=f"{lr_now:.2e}")

        avg_loss = epoch_loss / n_batches_per_epoch
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")

            if val_dataset is not None:
                val_acc = evaluate(model, tokenizer, val_dataset, device, args)
                logger.info(f"  val triplet accuracy: {val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(model, tokenizer, run_dir / "best")
                    logger.info(f"  ↑ New best: {best_val_acc:.4f}")

            save_checkpoint(model, tokenizer, run_dir / f"epoch_{epoch+1}")

        if is_ddp:
            dist.barrier()

    if rank == 0:
        save_checkpoint(model, tokenizer, run_dir / "final")
        logger.info(f"Done. Best val triplet accuracy: {best_val_acc:.4f}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
