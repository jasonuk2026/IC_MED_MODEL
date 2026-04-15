#!/usr/bin/env python3
"""
train_cpt.py — Continued Pre-Training (CPT) of a causal LM on EHRSHOT token blocks.

Input parquet schema (output of dataset/build_cpt_blocks_from_csv.py):
    patient_id  int64
    block_idx   int32
    num_tokens  int32
    input_ids   list<int32>   (all blocks are exactly block_size long)

Each block is treated as a standard next-token-prediction example:
    input  = input_ids[:-1]
    labels = input_ids[1:]

Usage — single GPU
──────────────────
    python train_cpt.py \
        --data_path EHRSHOT_ASSETS/cpt_blocks/qwen3_0.6b_block2048.parquet \
        --model_name Qwen/Qwen3-0.6B-Base \
        --output_dir output/cpt_qwen3_0.6b \
        --bf16 --gradient_checkpointing

Usage — multi-GPU DDP (e.g. 4 GPUs)
──────────────────────────────────
    torchrun --nproc_per_node=4 train_cpt.py \
        --data_path EHRSHOT_ASSETS/cpt_blocks/qwen3_0.6b_block2048.parquet \
        --model_name Qwen/Qwen3-0.6B-Base \
        --output_dir output/cpt_qwen3_0.6b \
        --bf16 --gradient_checkpointing
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_warning()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

class CPTDataset(Dataset):
    """Lazy parquet dataset with per-worker, per-row-group caching.

    Only row group metadata is read at construction time.  Each DataLoader
    worker opens the file independently after forking and caches the one row
    group it is currently reading.  Because the DataLoader assigns contiguous
    index ranges to each worker, cache hit rate is ~100% after the first miss
    per row group, and IO is fully overlapped by prefetching.
    """

    def __init__(self, parquet_path: str):
        self.path = parquet_path
        # Read only metadata — no data loaded here
        pf = pq.ParquetFile(parquet_path)
        meta = pf.metadata
        offsets, total = [], 0
        for i in range(meta.num_row_groups):
            offsets.append(total)
            total += meta.row_group(i).num_rows
        self.row_group_offsets = offsets
        self.total_rows = total
        logger.info(
            f"Dataset: {total:,} blocks across {len(offsets)} row groups  ({parquet_path})"
        )
        # Worker-local state — each forked worker gets its own copy
        self._pf: pq.ParquetFile | None = None
        self._cached_rg_idx: int = -1
        self._cached_col = None

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Binary search: which row group contains idx?
        offsets = self.row_group_offsets
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        rg_idx     = lo
        row_in_rg  = idx - offsets[rg_idx]

        # Open file lazily (once per worker, after fork)
        if self._pf is None:
            self._pf = pq.ParquetFile(self.path)

        # Cache one row group at a time
        if rg_idx != self._cached_rg_idx:
            rg = self._pf.read_row_group(rg_idx, columns=["input_ids"])
            self._cached_col    = rg.column("input_ids")
            self._cached_rg_idx = rg_idx

        return torch.tensor(self._cached_col[row_in_rg].as_py(), dtype=torch.long)


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(output_dir: Path, step: int, model, optimizer, scheduler, args):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)  # unwrap torch.compile
    raw_model.save_pretrained(ckpt_dir)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / "trainer_state.pt",
    )
    logger.info(f"Checkpoint saved → {ckpt_dir}")


def load_checkpoint(ckpt_dir: Path, optimizer, scheduler):
    state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info(f"Resumed from step {state['step']} ({ckpt_dir})")
    return state["step"]


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DDP Continued Pre-Training on EHRSHOT token blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / model
    p.add_argument("--data_path",   required=True, help="Parquet file from build_cpt_blocks_from_csv.py")
    p.add_argument("--model_name",  required=True, help="HF model name or local path")
    p.add_argument("--output_dir",  required=True, help="Directory for checkpoints and final model")
    p.add_argument("--local_files_only", action="store_true")
    # Training
    p.add_argument("--epochs",           type=int,   default=1)
    p.add_argument("--batch_size",       type=int,   default=8,    help="Per-GPU batch size")
    p.add_argument("--grad_accum",       type=int,   default=8,    help="Gradient accumulation steps")
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=0.1)
    p.add_argument("--max_grad_norm",    type=float, default=1.0)
    p.add_argument("--warmup_ratio",     type=float, default=0.05, help="Fraction of total steps used for warmup")
    # Efficiency
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--fa2", action="store_true", help="Use Flash Attention 2")
    p.add_argument("--fa3", action="store_true", help="Use Flash Attention 3")
    p.add_argument("--compile",    action="store_true", help="torch.compile the model")
    p.add_argument("--num_workers",  type=int, default=4, help="DataLoader workers per GPU")
    # Checkpointing
    p.add_argument("--save_steps",    type=int,   default=0,    help="Save checkpoint every N optimizer steps (0 = disable)")
    p.add_argument("--save_at_steps", type=int,   nargs="+",    help="Save checkpoint at these exact step numbers, e.g. --save_at_steps 1000 5000 10000")
    p.add_argument("--resume_from",   default=None,             help="Path to checkpoint dir to resume from")
    # Logging
    p.add_argument("--log_steps",    type=int, default=10)
    p.add_argument("--wandb_project",  default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags",     nargs="+", default=None)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Distributed setup ──────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp     = world_size > 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None

    if is_ddp:
        dist.init_process_group(backend="nccl", device_id=device)

    if rank != 0:
        logging.disable(logging.CRITICAL)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"World size: {world_size}  |  device: {device}")

    # ── wandb ──────────────────────────────────────────────────────────────────
    use_wandb = (args.wandb_project is not None) and (rank == 0)
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"cpt-{datetime.now().strftime('%m%d-%H%M')}",
            tags=args.wandb_tags,
            config=vars(args),
        )

    # ── Dataset & DataLoader ───────────────────────────────────────────────────
    dataset = CPTDataset(args.data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp \
              else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_name}")
    dtype = torch.bfloat16
    attn_implementation = "eager"
    if args.fa2:
        attn_implementation = "flash_attention_2"
    if args.fa3:
        attn_implementation = 'flash_attention_3'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=args.local_files_only,
    ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.compile:
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model params: {n_params:.1f}M")

    # ── Optimizer & schedule ───────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(f"Total optimizer steps: {total_steps:,}  warmup: {warmup_steps:,}")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum * world_size}")

    # ── Resume ─────────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        raw_model = model.module if is_ddp else model
        raw_model.from_pretrained(ckpt_path)
        global_step = load_checkpoint(ckpt_path, optimizer, scheduler)

    # ── Training loop ──────────────────────────────────────────────────────────
    # tokens per optimizer step across all GPUs
    tokens_per_step: int | None = None   # set on first batch once block_size is known

    # Timing: skip first WARMUP_STEPS steps so torch.compile JIT doesn't skew stats
    TIMING_WARMUP = 10
    total_training_time = 0.0            # cumulative wall time (excluding warmup steps)

    model.train()
    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)

        running_loss = 0.0
        micro_steps  = 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0), dynamic_ncols=True)

        t0: float | None = None  # start of current grad_accum window

        for batch_idx, input_ids in enumerate(pbar):
            # input_ids: (B, block_size) int64
            input_ids = input_ids.to(device)
            labels    = input_ids.clone()

            # derive tokens_per_step once
            if tokens_per_step is None:
                block_size = input_ids.shape[1]
                tokens_per_step = args.batch_size * block_size * world_size * args.grad_accum

            # start timer at the first micro-step of each accumulation window
            if micro_steps % args.grad_accum == 0:
                synchronize()
                t0 = time.time()

            outputs = model(input_ids=input_ids, labels=labels)
            loss    = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum  # .item() is a CPU-GPU sync
            micro_steps  += 1

            # Optimizer step every grad_accum micro-batches
            if micro_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                synchronize()
                dt = time.time() - t0

                global_step += 1
                if global_step > TIMING_WARMUP:
                    total_training_time += dt

                if rank == 0 and global_step % args.log_steps == 0:
                    avg_loss  = running_loss / micro_steps
                    lr_now    = scheduler.get_last_lr()[0]
                    tok_per_s = int(tokens_per_step / dt)
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        tok_s=f"{tok_per_s:,}",
                        lr=f"{lr_now:.2e}",
                        step=global_step,
                    )
                    if use_wandb:
                        wandb_run.log(
                            {"train/loss": avg_loss, "train/lr": lr_now, "train/tok_per_s": tok_per_s},
                            step=global_step,
                        )
                    running_loss = 0.0
                    micro_steps  = 0

                if rank == 0:
                    should_save = (
                        (args.save_steps > 0 and global_step % args.save_steps == 0) or
                        (args.save_at_steps and global_step in args.save_at_steps)
                    )
                    if should_save:
                        save_checkpoint(output_dir, global_step, model, optimizer, scheduler, args)

    if rank == 0:
        logger.info(f"Total training time (excl. first {TIMING_WARMUP} steps): {total_training_time/60:.2f}m")

    # ── Final save ─────────────────────────────────────────────────────────────
    if rank == 0:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if is_ddp else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)  # unwrap torch.compile
        raw_model.save_pretrained(final_dir)
        AutoTokenizer.from_pretrained(
            args.model_name, local_files_only=args.local_files_only
        ).save_pretrained(final_dir)
        logger.info(f"Final model saved → {final_dir}")
        if use_wandb:
            wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
