#!/usr/bin/env python3
"""
train_embedding_custom.py — Transparent DDP training loop for medical EHR embeddings.

Replaces SentenceTransformerTrainer with a plain PyTorch training loop so every
step is explicit and inspectable.

Loss: BatchAllTripletLoss — all valid (anchor, positive, negative) triplets in
  the batch contribute to the loss (not just the hardest pair).

DDP: each rank loads its own slice of the parquet files. Because data is split
  by rank BEFORE the GroupByLabel sampler runs, the sampler can operate
  independently per rank without any distributed-awareness.

  torchrun --nproc_per_node=N train_embedding_custom.py \\
      --data_paths r0.parquet r1.parquet ... \\
      --bf16 --flash_attn

Single GPU:
  python train_embedding_custom.py --data_paths all.parquet ...

NOTE: QLoRA (--qlora) is incompatible with multi-GPU DDP because 4-bit weights
  cannot be all-reduced. Use single GPU for QLoRA.
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

import pandas as pd
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

# ── Constants ─────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = ["<|disease_start|>", "<|disease_end|>", "<|event_start|>", "<|event_end|>"]

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}


# _loss_module is created inside train() so --triplet_margin is respected.


# ── Text formatting ────────────────────────────────────────────────────────────

def format_events(events: list) -> str:
    lines = []
    for e in events:
        desc  = e.get("description") or ""
        code  = e.get("code") or ""
        value = e.get("value") or ""
        unit  = e.get("unit") or ""
        if desc and code:
            line = f"{desc} [{code}]"
        elif code:
            line = f"[{code}]"
        elif desc:
            line = desc
        else:
            continue
        if value:
            line += f" | value={value}"
        if unit:
            line += f" | unit={unit}"
        lines.append(line)
    return "\n".join(lines)


def build_prompt(disease_name: str, events: list) -> str:
    return (
        f"<|disease_start|>{disease_name}<|disease_end|>"
        f"<|event_start|>{format_events(events)}<|event_end|>"
    )


def build_label_map(tasks: list) -> dict:
    """(task, polarity) → unique int so different diseases don't merge clusters."""
    label_map = {}
    for idx, task in enumerate(sorted(set(tasks))):
        label_map[(task, False)] = idx * 2
        label_map[(task, True)]  = idx * 2 + 1
    return label_map


# ── Dataset ───────────────────────────────────────────────────────────────────

class EHRDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        label_map = build_label_map(df["task"].tolist())
        self.texts:  list[str] = []
        self.labels: list[int] = []
        skipped = 0
        for task, events, label in zip(df["task"], df["events"], df["label"]):
            if not format_events(events).strip():
                skipped += 1
                continue
            disease_name = TASK_2_DISEASE_NAME.get(task, task)
            self.texts.append(build_prompt(disease_name, events))
            self.labels.append(label_map[(task, bool(label))])
        if skipped:
            logger.warning(f"  Skipped {skipped} samples with empty events.")
        counts = Counter(self.labels)
        logger.info(f"  {len(self.texts)} samples, {len(counts)} label classes: "
                    f"{dict(sorted(counts.items()))}")

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
    """Add special tokens, resize embeddings, attach LoRA."""
    # ── Special tokens ────────────────────────────────────────────────────────
    vocab_size_before = len(tokenizer)
    tokenizer.add_special_tokens({"extra_special_tokens": SPECIAL_TOKENS})
    vocab_size_after = len(tokenizer)
    model.resize_token_embeddings(vocab_size_after)
    logger.info(f"Vocabulary: {vocab_size_before} → {vocab_size_after} "
                f"(+{vocab_size_after - vocab_size_before} special tokens: {SPECIAL_TOKENS})")
    for tok, phrase in [
        ("<|disease_start|>", "disease name"),
        ("<|disease_end|>",   "disease name end"),
        ("<|event_start|>",   "medical event start"),
        ("<|event_end|>",     "<|endoftext|>"),
    ]:
        ids      = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        mean_emb = model.embed_tokens.weight.data[ids].mean(dim=0)
        tok_id   = tokenizer.convert_tokens_to_ids(tok)
        model.embed_tokens.weight.data[tok_id] = mean_emb
    pad_id = tokenizer.convert_tokens_to_ids("<|event_end|>")
    tokenizer.pad_token_id = pad_id

    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    if args.qlora:
        prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    # Collect module names for modules_to_save: always include embed_tokens; optionally norm layers
    modules_to_save = ["embed_tokens"]
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
        modules_to_save=modules_to_save,
    ))

    # Cast any fp32 params left behind (e.g. norm layers from prepare_model_for_kbit_training)
    if args.bf16:
        for name, param in model.named_parameters():
            if param.dtype == torch.float32 and "lora" not in name:
                param.data = param.data.to(torch.bfloat16)

    # ── Optional: unfreeze norm layers ────────────────────────────────────────
    if args.train_norm:
        n_norm = 0
        for name, param in model.named_parameters():
            if "norm.weight" in name:
                param.requires_grad = True
                n_norm += param.numel()
        logger.info(f"  Norm weights unfrozen: {n_norm:,} extra trainable params")

    # ── Optional: train new special token embeddings only ─────────────────────
    if args.train_new_tokens:
        embed = model.base_model.model.embed_tokens
        embed.weight.requires_grad = True
        special_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS]
        def _mask_grad(grad, ids=special_ids):
            mask = torch.zeros_like(grad)
            mask[ids] = 1.0
            return grad * mask
        embed.weight.register_hook(_mask_grad)
        logger.info(f"  Special token embeddings trainable (ids={special_ids})")

    # ── Gradient checkpointing ────────────────────────────────────────────────
    if args.gradient_checkpointing:
        # enable_input_require_grads() is required for LoRA + gradient checkpointing:
        # token embeddings have requires_grad=False by default, so without this hook
        # PyTorch skips gradient computation for the first checkpoint segment entirely.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # embed_tokens.weight has requires_grad=True but only 4 rows are effectively
    # updated (gradient hook zeros the rest). Count effective params separately.
    embed_weight = model.base_model.model.embed_tokens.weight
    effective_trainable = trainable
    if args.train_new_tokens and embed_weight.requires_grad:
        effective_trainable = trainable - embed_weight.numel() + len(SPECIAL_TOKENS) * embed_weight.shape[1]
    logger.info(f"  requires_grad params : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    logger.info(f"  effectively updated  : {effective_trainable:,} / {total:,} ({100*effective_trainable/total:.2f}%)")
    return model


# ── Encoding ──────────────────────────────────────────────────────────────────

def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-token pooling for decoder models with right-padding.
    Uses attention_mask to locate the actual last non-padding token per sample.
    """
    seq_lengths = attention_mask.sum(dim=1) - 1   # 0-indexed position of last real token
    batch_idx   = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, seq_lengths]


@torch.inference_mode()
def encode_texts(model, tokenizer, texts: list, device, batch_size: int) -> torch.Tensor:
    """Encode texts to L2-normalised embeddings without gradients (for eval)."""
    all_emb = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb.float(), p=2, dim=-1)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ── Evaluation ────────────────────────────────────────────────────────────────

def build_eval_triplets(val_df: pd.DataFrame, n_per_task: int, seed: int):
    rng    = random.Random(seed)
    groups = defaultdict(list)
    for task, events, label in zip(val_df["task"], val_df["events"], val_df["label"]):
        if not format_events(events).strip():
            continue
        groups[(task, bool(label))].append(
            build_prompt(TASK_2_DISEASE_NAME.get(task, task), events)
        )
    anchors, positives, negatives = [], [], []
    for task in sorted({t for t, _ in groups}):
        pos = groups[(task, True)]
        if len(pos) < 2:
            continue
        neg = [t for (tsk, pol), texts in groups.items()
               if not (tsk == task and pol) for t in texts]
        if not neg:
            continue
        n    = min(n_per_task, len(pos))
        pool = list(range(len(pos)))
        rng.shuffle(pool)
        for i in range(n):
            ai = pool[i % len(pos)]
            pi = rng.choice([j for j in pool if j != ai])
            ni = rng.randint(0, len(neg) - 1)
            anchors.append(pos[ai])
            positives.append(pos[pi])
            negatives.append(neg[ni])
    return anchors, positives, negatives


def evaluate(model, tokenizer, val_df: pd.DataFrame, device, args) -> float:
    """Triplet accuracy: fraction where d(anchor, positive) < d(anchor, negative)."""
    anchors, positives, negatives = build_eval_triplets(
        val_df, n_per_task=args.n_eval_triplets_per_task, seed=args.seed
    )
    if not anchors:
        logger.warning("No eval triplets could be built.")
        return 0.0

    amp_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    amp_ctx   = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()

    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    all_texts = anchors + positives + negatives
    embs = encode_texts(raw_model, tokenizer, all_texts, device,
                        batch_size=args.eval_batch_size)
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
    p.add_argument("--data_paths", nargs="+", required=True,
                   help="Train parquet files. Rank r loads paths[r::world_size].")
    p.add_argument("--val_data_paths", nargs="+", default=None,
                   help="Val parquet files (evaluated on rank 0 only).")

    # Model
    p.add_argument("--model_name",     default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--flash_attn",     action="store_true")
    p.add_argument("--bf16",           action="store_true")
    p.add_argument("--fp16",           action="store_true")
    p.add_argument("--qlora",          action="store_true",
                   help="4-bit NF4 QLoRA. Single GPU only.")

    # LoRA
    p.add_argument("--lora_r",              type=int,   default=4)
    p.add_argument("--lora_alpha",          type=int,   default=8)
    p.add_argument("--lora_dropout",        type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--train_norm",          action="store_true")
    p.add_argument("--train_new_tokens",    action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Training
    p.add_argument("--output_dir",   default="output/medical-embedding-custom")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--log_steps",    type=int,   default=10)

    # Loss
    p.add_argument("--triplet_margin",           type=float, default=0.5)
    p.add_argument("--n_eval_triplets_per_task", type=int,   default=100)
    p.add_argument("--eval_batch_size",          type=int,   default=4)

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
        raise RuntimeError("QLoRA is not compatible with multi-GPU DDP. Use single GPU.")

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run dir: {run_dir}")
        logger.info(f"World size: {world_size}")

    # ── Data: rank r loads paths[r::world_size] ────────────────────────────────
    my_train_paths = args.data_paths[rank :: world_size]
    if not my_train_paths:
        raise ValueError(f"Rank {rank}: no data paths assigned "
                         f"({len(args.data_paths)} paths, {world_size} ranks).")
    logger.info(f"Rank {rank} train paths: {my_train_paths}")

    train_df = pd.concat(
        [pd.read_parquet(p).pipe(lambda df: df[df["split"] == "train"])
         for p in my_train_paths],
        ignore_index=True,
    )
    logger.info(f"Rank {rank}: {len(train_df)} train rows")

    val_df = None
    if args.val_data_paths and rank == 0:
        val_df = pd.concat(
            [pd.read_parquet(p).pipe(lambda df: df[df["split"] == "val"])
             for p in args.val_data_paths],
            ignore_index=True,
        )
        logger.info(f"Val rows: {len(val_df)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args)
    model = setup_model(model, tokenizer, args)
    model = model.to(device)

    if is_ddp:
        # find_unused_parameters=True needed because frozen base-model params
        # don't receive gradients and would otherwise cause DDP to hang.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # ── DataLoader ─────────────────────────────────────────────────────────────
    train_dataset = EHRDataset(train_df)
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

    n_batches_per_epoch  = len(train_loader)
    n_opt_steps_per_epoch = math.ceil(n_batches_per_epoch / args.grad_accum)
    total_opt_steps      = n_opt_steps_per_epoch * args.epochs
    warmup_steps         = int(total_opt_steps * args.warmup_ratio)

    def lr_lambda(opt_step: int) -> float:
        if opt_step < warmup_steps:
            return opt_step / max(warmup_steps, 1)
        progress = (opt_step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)   # linear decay to 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ──────────────────────────────────────────────────────────
    logger.info(f"Training: {args.epochs} epochs, "
                f"{n_batches_per_epoch} batches/epoch, "
                f"{total_opt_steps} optimizer steps total")

    loss_module  = BatchAllTripletLoss(model=None, margin=args.triplet_margin)
    logger.info(f"BatchAllTripletLoss margin={args.triplet_margin}")

    best_val_acc = 0.0
    opt_step     = 0

    for epoch in range(args.epochs):
        if is_ddp:
            dist.barrier()
        sampler.set_epoch(epoch)
        model.train()

        epoch_loss     = 0.0
        n_opt_this_epoch = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=(rank != 0), dynamic_ncols=True)
        for batch_idx, (texts, labels) in enumerate(pbar):

            # Tokenise on CPU, move to device
            enc = tokenizer(
                texts, padding=True, truncation=False,
                return_tensors="pt", add_special_tokens=False
            ).to(device)
            # enc      = {k: v.to(device) for k, v in enc.items()}
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            # Skip gradient sync on accumulation steps (saves NCCL overhead)
            is_update_step = (batch_idx + 1) % args.grad_accum == 0 \
                             or (batch_idx + 1) == n_batches_per_epoch
            sync_ctx = nullcontext() if (is_update_step or not is_ddp) \
                       else model.no_sync()

            with sync_ctx:
                out  = model(**enc)
                emb  = last_token_pool(out.last_hidden_state, enc["attention_mask"])
                emb  = F.normalize(emb, p=2, dim=-1)
                loss = loss_module.batch_all_triplet_loss(labels_t, emb)
                loss = loss / args.grad_accum   # scale for accumulation

                loss.backward()

            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    args.grad_clip,
                )
                optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                opt_step       += 1
                n_opt_this_epoch += 1

            epoch_loss += loss.item() * args.grad_accum

            if rank == 0 and is_update_step:
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}",
                                 gnorm=f"{grad_norm:.3f}", lr=f"{lr_now:.2e}")

        avg_loss = epoch_loss / n_batches_per_epoch
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")

            # ── Eval (rank 0 only) ────────────────────────────────────────────
            if val_df is not None:
                val_acc = evaluate(model, tokenizer, val_df, device, args)
                logger.info(f"  val triplet accuracy: {val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(model, tokenizer, run_dir / "best")
                    logger.info(f"  ↑ New best: {best_val_acc:.4f}")

            save_checkpoint(model, tokenizer, run_dir / f"epoch_{epoch+1}")

        if is_ddp:
            dist.barrier()   # Wait for rank 0 eval + checkpoint before next epoch

    if rank == 0:
        save_checkpoint(model, tokenizer, run_dir / "final")
        logger.info(f"Done. Best val triplet accuracy: {best_val_acc:.4f}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
