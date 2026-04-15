#!/usr/bin/env python3
"""
train_stage2.py — Stage 2: multi-objective EHR event-level pre-training.

Three joint losses
──────────────────
  1. CPT   (--lambda_cpt)
       Standard causal LM next-token prediction on the full unmasked sequence.

  2. JEPA  (--lambda_jepa)
       • The last --future_len tokens are the "target" window.
       • --mask_ratio fraction of those positions are replaced with [MASK].
       • Online encoder encodes the masked sequence → predictor MLP → predictions.
       • EMA teacher encodes the full sequence (no grad) → targets.
       • MSE between predictor output and teacher output at masked positions only.

  3. RED   (--lambda_red)
       • Each EHR event ends with EOS (our end-of-event separator token).
       • The EOS hidden state (from the full-sequence forward pass) should equal
         the mean of all preceding token hidden states in the same event.
       • MSE averaged over all events in the batch.

EMA teacher
──────────────────
  A separate copy of the online model (no grad, no DDP, no compile).
  Updated after every optimizer step:
      θ_teacher ← τ·θ_teacher + (1−τ)·θ_online

Usage — single GPU
──────────────────
    python train_stage2.py \\
        --data_path EHRSHOT_ASSETS/cpt_blocks/qwen3_0.6b_block2048.parquet \\
        --model_name output/cpt_qwen3_0.6b/final \\
        --output_dir output/stage2_qwen3_0.6b \\
        --gradient_checkpointing --flash_attn

Usage — multi-GPU DDP
──────────────────────
    torchrun --nproc_per_node=4 train_stage2.py ...
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision("high")
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


# ── Dataset (lazy parquet, identical to train_cpt.py) ─────────────────────────

class CPTDataset(Dataset):
    def __init__(self, parquet_path: str):
        pf = pq.ParquetFile(parquet_path)
        meta = pf.metadata
        offsets, total = [], 0
        for i in range(meta.num_row_groups):
            offsets.append(total)
            total += meta.row_group(i).num_rows
        self.path = parquet_path
        self.row_group_offsets = offsets
        self.total_rows = total
        logger.info(f"Dataset: {total:,} blocks across {len(offsets)} row groups")
        self._pf = None
        self._cached_rg_idx = -1
        self._cached_col = None

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> torch.Tensor:
        offsets = self.row_group_offsets
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        rg_idx = lo
        row_in_rg = idx - offsets[rg_idx]
        if self._pf is None:
            self._pf = pq.ParquetFile(self.path)
        if rg_idx != self._cached_rg_idx:
            rg = self._pf.read_row_group(rg_idx, columns=["input_ids"])
            self._cached_col = rg.column("input_ids")
            self._cached_rg_idx = rg_idx
        return torch.tensor(self._cached_col[row_in_rg].as_py(), dtype=torch.long)


# ── Predictor (position-wise 2-layer MLP) ─────────────────────────────────────

class Predictor(nn.Module):
    """Maps online encoder hidden states → predicted target embeddings."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── EMA helpers ────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(online: nn.Module, teacher: nn.Module, decay: float) -> None:
    for p_o, p_t in zip(online.parameters(), teacher.parameters()):
        p_t.data.mul_(decay).add_(p_o.data, alpha=1.0 - decay)


# ── Collate: event boundaries + JEPA masking ──────────────────────────────────

def collate_fn(
    batch: list[torch.Tensor],
    eos_token_id: int,
    mask_token_id: int,
    future_len: int,
    mask_ratio: float,
) -> dict:
    full_input_ids = torch.stack(batch)           # (B, T)
    B, T = full_input_ids.shape
    future_start = max(0, T - future_len)

    # ── Event boundaries ──
    # For each sample: list of (event_start, red_pos) where red_pos is the
    # position of the EOS token that closes the event.
    event_boundaries: list[list[tuple[int, int]]] = []
    for b in range(B):
        ids = full_input_ids[b].tolist()
        bounds: list[tuple[int, int]] = []
        ev_start = 0
        for pos, tok in enumerate(ids):
            if tok == eos_token_id:
                if pos > ev_start:          # skip zero-length events
                    bounds.append((ev_start, pos))
                ev_start = pos + 1
        event_boundaries.append(bounds)

    # ── JEPA masking ──
    masked_input_ids = full_input_ids.clone()
    masked_positions = torch.zeros(B, T, dtype=torch.bool)
    n_mask = max(1, int(future_len * mask_ratio))
    for b in range(B):
        idx = torch.randperm(future_len, device=full_input_ids.device)[:n_mask]
        positions = future_start + idx
        masked_input_ids[b, positions] = mask_token_id
        masked_positions[b, positions] = True

    return {
        "full_input_ids":   full_input_ids,
        "masked_input_ids": masked_input_ids,
        "masked_positions": masked_positions,   # (B, T) bool
        "event_boundaries": event_boundaries,   # list[list[(start, red_pos)]]
    }


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(output_dir: Path, step: int, online, predictor, optimizer, scheduler, args):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP and torch.compile before saving
    raw = online.module if isinstance(online, DDP) else online
    raw = getattr(raw, "_orig_mod", raw)
    raw.save_pretrained(ckpt_dir)
    torch.save(predictor.state_dict(), ckpt_dir / "predictor.pt")
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        ckpt_dir / "trainer_state.pt",
    )
    logger.info(f"Checkpoint saved → {ckpt_dir}")


def load_checkpoint(ckpt_dir: Path, predictor, optimizer, scheduler):
    state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    predictor.load_state_dict(torch.load(ckpt_dir / "predictor.pt", map_location="cpu"))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info(f"Resumed from step {state['step']} ({ckpt_dir})")
    return state["step"]


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2 multi-objective EHR foundation model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / model
    p.add_argument("--data_path",   required=True)
    p.add_argument("--model_name",  required=True, help="HF name or local path (e.g. CPT checkpoint)")
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--local_files_only", action="store_true")
    # Training
    p.add_argument("--epochs",       type=int,   default=1)
    p.add_argument("--batch_size",   type=int,   default=4,   help="Per-GPU batch size")
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm",type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    # Loss weights
    p.add_argument("--lambda_cpt",  type=float, default=1.0)
    p.add_argument("--lambda_jepa", type=float, default=1.0)
    p.add_argument("--lambda_red",  type=float, default=1.0)
    # JEPA config
    p.add_argument("--future_len",  type=int,   default=256,  help="Last N tokens as JEPA target window")
    p.add_argument("--mask_ratio",  type=float, default=0.5,  help="Fraction of future tokens to mask")
    p.add_argument("--ema_decay",   type=float, default=0.999)
    p.add_argument("--pred_mlp_ratio", type=float, default=4.0, help="Predictor MLP inner dim ratio")
    # Efficiency
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn",  action="store_true")
    p.add_argument("--compile",     action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    # Checkpointing
    p.add_argument("--save_steps",    type=int, default=0,   help="Save every N steps (0 = disable)")
    p.add_argument("--save_at_steps", type=int, nargs="+",   help="Save at these exact steps")
    p.add_argument("--resume_from",   default=None)
    # Logging
    p.add_argument("--log_steps",     type=int, default=10)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name",default=None)
    p.add_argument("--wandb_tags",    nargs="+", default=None)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp     = world_size > 1
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
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
            name=args.wandb_run_name or f"stage2-{datetime.now().strftime('%m%d-%H%M')}",
            tags=args.wandb_tags,
            config=vars(args),
        )

    # ── Tokenizer + [MASK] token ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    vocab_resized = False
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
        vocab_resized = True
        logger.info(f"Added <mask> token (id={tokenizer.mask_token_id})")
    mask_token_id = tokenizer.mask_token_id
    eos_token_id  = tokenizer.eos_token_id

    # ── Dataset & DataLoader ───────────────────────────────────────────────────
    dataset = CPTDataset(args.data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) \
              if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(
            collate_fn,
            eos_token_id=eos_token_id,
            mask_token_id=mask_token_id,
            future_len=args.future_len,
            mask_ratio=args.mask_ratio,
        ),
    )

    # ── Online encoder ─────────────────────────────────────────────────────────
    logger.info(f"Loading online encoder: {args.model_name}")
    dtype = torch.bfloat16
    online = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        local_files_only=args.local_files_only,
    ).to(device)
    if vocab_resized:
        online.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        online.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.compile:
        online = torch.compile(online)
        logger.info("torch.compile enabled for online encoder")
    if is_ddp:
        online = DDP(online, device_ids=[local_rank], output_device=local_rank)

    # ── EMA teacher (no grad, no DDP, no compile) ──────────────────────────────
    logger.info("Building EMA teacher (separate weights, no grad)")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        local_files_only=args.local_files_only,
    ).to(device)
    if vocab_resized:
        teacher.resize_token_embeddings(len(tokenizer))
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # ── Predictor (position-wise MLP, online only) ─────────────────────────────
    hidden_size = online.config.hidden_size if not is_ddp else online.module.config.hidden_size
    predictor = Predictor(hidden_size, args.pred_mlp_ratio).to(device).to(dtype)
    if is_ddp:
        predictor = DDP(predictor, device_ids=[local_rank], output_device=local_rank)

    n_online  = sum(p.numel() for p in online.parameters()) / 1e6
    n_pred    = sum(p.numel() for p in predictor.parameters()) / 1e6
    logger.info(f"Online encoder: {n_online:.1f}M  |  Predictor: {n_pred:.2f}M")

    # ── Optimizer ──────────────────────────────────────────────────────────────
    all_params = list(online.parameters()) + list(predictor.parameters())
    decay_p    = [p for p in all_params if p.requires_grad and p.dim() >= 2]
    nodecay_p  = [p for p in all_params if p.requires_grad and p.dim() < 2]
    optimizer  = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": args.weight_decay},
         {"params": nodecay_p, "weight_decay": 0.0}],
        lr=args.lr,
    )

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"Total steps: {total_steps:,}  warmup: {warmup_steps:,}")
    logger.info(f"Effective batch: {args.batch_size * args.grad_accum * world_size}")

    # ── Resume ─────────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume_from:
        ckpt = Path(args.resume_from)
        raw = online.module if is_ddp else online
        raw = getattr(raw, "_orig_mod", raw)
        raw.from_pretrained(ckpt)
        global_step = load_checkpoint(ckpt, predictor, optimizer, scheduler)

    # ── Unwrap helpers for EMA update ─────────────────────────────────────────
    def get_raw_online():
        m = online.module if is_ddp else online
        return getattr(m, "_orig_mod", m)

    # ── Training loop ──────────────────────────────────────────────────────────
    TIMING_WARMUP    = 10
    total_train_time = 0.0
    tokens_per_step: int | None = None

    online.train()
    predictor.train()

    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)

        run_loss = run_cpt = run_jepa = run_red = 0.0
        micro_steps = 0
        optimizer.zero_grad()
        t0: float | None = None

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0), dynamic_ncols=True)

        for batch in pbar:
            full_ids    = batch["full_input_ids"].to(device)
            masked_ids  = batch["masked_input_ids"].to(device)
            mask_pos    = batch["masked_positions"].to(device)    # (B, T) bool
            ev_bounds   = batch["event_boundaries"]               # list[list[(start, red_pos)]]
            B, T        = full_ids.shape

            if tokens_per_step is None:
                tokens_per_step = args.batch_size * T * world_size * args.grad_accum

            if micro_steps % args.grad_accum == 0:
                synchronize()
                t0 = time.time()

            # ── Part 1 + Part 3: full sequence forward ─────────────────────
            full_out = online(
                input_ids=full_ids,
                labels=full_ids,
                output_hidden_states=True,
            )
            loss_cpt  = full_out.loss
            full_h    = full_out.hidden_states[-1]    # (B, T, D)

            # ── Part 3: RED-align loss ─────────────────────────────────────
            loss_red  = full_h.new_zeros(())
            n_events  = 0
            for b in range(B):
                for (ev_start, red_pos) in ev_bounds[b]:
                    # token hiddens inside the event, excluding the EOS itself
                    # +1 to skip the opening token (mirrors user's reference code)
                    ev_h = full_h[b, ev_start + 1 : red_pos]
                    if ev_h.shape[0] == 0:
                        continue
                    avg_h = ev_h.mean(dim=0)
                    loss_red = loss_red + F.mse_loss(full_h[b, red_pos], avg_h.detach())
                    n_events += 1
            if n_events > 0:
                loss_red = loss_red / n_events

            # ── Part 2: JEPA forward ────────────────────────────────────────
            masked_out = online(
                input_ids=masked_ids,
                output_hidden_states=True,
            )
            online_h   = masked_out.hidden_states[-1]             # (B, T, D)
            pred_h     = predictor(online_h)                      # (B, T, D)

            with torch.no_grad():
                teacher_h = teacher(
                    input_ids=full_ids,
                    output_hidden_states=True,
                ).hidden_states[-1]                               # (B, T, D)

            # MSE only at masked positions
            pred_masked   = pred_h[mask_pos]                      # (N, D)
            target_masked = teacher_h[mask_pos].detach()          # (N, D)
            loss_jepa = F.mse_loss(pred_masked, target_masked)

            # ── Combined loss ───────────────────────────────────────────────
            loss = (
                args.lambda_cpt  * loss_cpt  +
                args.lambda_jepa * loss_jepa +
                args.lambda_red  * loss_red
            ) / args.grad_accum
            loss.backward()

            run_loss  += loss.item() * args.grad_accum
            run_cpt   += loss_cpt.item()
            run_jepa  += loss_jepa.item()
            run_red   += loss_red.item() if isinstance(loss_red, torch.Tensor) else float(loss_red)
            micro_steps += 1

            # ── Optimizer step ─────────────────────────────────────────────
            if micro_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(online.parameters()) + list(predictor.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA teacher update
                update_ema(get_raw_online(), teacher, args.ema_decay)

                synchronize()
                dt = time.time() - t0
                global_step += 1
                if global_step > TIMING_WARMUP:
                    total_train_time += dt

                if rank == 0 and global_step % args.log_steps == 0:
                    n = micro_steps
                    tok_s = int(tokens_per_step / dt)
                    lr_now = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        loss=f"{run_loss/n:.4f}",
                        cpt=f"{run_cpt/n:.4f}",
                        jepa=f"{run_jepa/n:.4f}",
                        red=f"{run_red/n:.4f}",
                        tok_s=f"{tok_s:,}",
                        lr=f"{lr_now:.2e}",
                    )
                    if use_wandb:
                        wandb_run.log({
                            "train/loss":      run_loss  / n,
                            "train/loss_cpt":  run_cpt   / n,
                            "train/loss_jepa": run_jepa  / n,
                            "train/loss_red":  run_red   / n,
                            "train/lr":        lr_now,
                            "train/tok_per_s": tok_s,
                        }, step=global_step)
                    run_loss = run_cpt = run_jepa = run_red = 0.0
                    micro_steps = 0

                if rank == 0:
                    should_save = (
                        (args.save_steps > 0 and global_step % args.save_steps == 0) or
                        (args.save_at_steps and global_step in args.save_at_steps)
                    )
                    if should_save:
                        save_checkpoint(output_dir, global_step, online, predictor, optimizer, scheduler, args)

    # ── Final save ─────────────────────────────────────────────────────────────
    if rank == 0:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw = get_raw_online()
        raw.save_pretrained(final_dir)
        torch.save(
            (predictor.module if is_ddp else predictor).state_dict(),
            final_dir / "predictor.pt",
        )
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved → {final_dir}")
        logger.info(f"Total training time (excl. first {TIMING_WARMUP} steps): {total_train_time/60:.2f}m")
        if use_wandb:
            wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
