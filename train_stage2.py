#!/usr/bin/env python3
"""
train_stage2.py — Stage 2: multi-objective EHR event-level pre-training.

Three joint losses
──────────────────
  1. CPT   (--lambda_cpt)
       Standard causal LM next-token prediction on the full unmasked sequence.

  2. JEPA  (--lambda_jepa)
       • For each sample, randomly choose --num_mask_events complete events.
       • All non-EOS tokens inside those events are replaced with a learnable
         mask embedding (no tokenizer/vocab change).
       • Online encoder encodes the masked sequence → predictor MLP → predictions.
       • EMA teacher encodes the full sequence (no grad) → targets.
       • MSE between predictor output and teacher output at the masked token
         positions only.

  3. RED   (--lambda_red)
       • Each masked EHR event keeps its EOS token visible.
       • Average the predictor outputs across the masked tokens of that event.
       • Regress that average to the teacher EOS hidden state for the same event.
       • MSE averaged over all masked events in the batch.

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


class LearnableMaskEmbedding(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.mask = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        nn.init.normal_(self.mask, mean=0.0, std=0.02)

    def apply(self, input_ids: torch.Tensor, embed_tokens: nn.Module, masked_positions: torch.Tensor) -> torch.Tensor:
        inputs_embeds = embed_tokens(input_ids)
        if masked_positions.any():
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[masked_positions] = self.mask
        return inputs_embeds


# ── EMA helpers ────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(online: nn.Module, teacher: nn.Module, decay: float) -> None:
    for p_o, p_t in zip(online.parameters(), teacher.parameters()):
        p_t.data.mul_(decay).add_(p_o.data, alpha=1.0 - decay)


# ── Collate: event boundaries + event-level masking ───────────────────────────

def collate_fn(
    batch: list[torch.Tensor],
    eos_token_id: int,
    num_mask_events: int,
) -> dict:
    full_input_ids = torch.stack(batch)           # (B, T)
    B, T = full_input_ids.shape

    # ── Event boundaries ──
    # For each sample: list of (event_start, eos_pos) where eos_pos is the
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

    # ── Event-level JEPA masking ──
    masked_positions = torch.zeros(B, T, dtype=torch.bool)
    masked_event_boundaries: list[list[tuple[int, int]]] = []
    for b in range(B):
        bounds = event_boundaries[b]
        if not bounds:
            masked_event_boundaries.append([])
            continue
        n_select = min(num_mask_events, len(bounds))
        chosen_idx = torch.randperm(len(bounds), device=full_input_ids.device)[:n_select].tolist()
        chosen_bounds = [bounds[i] for i in chosen_idx]
        chosen_bounds.sort()
        for ev_start, eos_pos in chosen_bounds:
            if eos_pos > ev_start:
                masked_positions[b, ev_start:eos_pos] = True
        masked_event_boundaries.append(chosen_bounds)

    return {
        "full_input_ids":          full_input_ids,
        "masked_positions":        masked_positions,        # (B, T) bool
        "event_boundaries":        event_boundaries,        # list[list[(start, eos_pos)]]
        "masked_event_boundaries": masked_event_boundaries, # list[list[(start, eos_pos)]]
    }


def _decode_ids(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _render_masked_preview(tokenizer, ids: list[int], mask_positions: list[int], mask_placeholder: str) -> str:
    pieces: list[str] = []
    pos_set = set(mask_positions)
    i = 0
    while i < len(ids):
        if i in pos_set:
            j = i
            while j < len(ids) and j in pos_set:
                j += 1
            pieces.append(mask_placeholder)
            i = j
        else:
            j = i
            while j < len(ids) and j not in pos_set:
                j += 1
            pieces.append(_decode_ids(tokenizer, ids[i:j]))
            i = j
    return "".join(pieces)


def log_objective_preview(
    tokenizer,
    batch: dict,
    *,
    eos_token_id: int,
    future_len: int,
    mask_placeholder: str = "<LEARNABLE_MASK>",
) -> None:
    full_ids = batch["full_input_ids"]
    masked_positions = batch["masked_positions"]
    masked_event_boundaries = batch["masked_event_boundaries"]
    if full_ids.size(0) == 0:
        return

    sample_idx = int(torch.randint(0, full_ids.size(0), (1,)).item())
    sample_ids = full_ids[sample_idx].tolist()
    sample_mask = masked_positions[sample_idx].tolist()
    future_start = max(0, len(sample_ids) - future_len)
    future_ids = sample_ids[future_start:]

    logger.info("")
    logger.info("=" * 72)
    logger.info("Stage2 Objective Preview")
    logger.info("=" * 72)
    logger.info("sample_idx=%d  total_seq_len=%d  cpt_preview_len=%d", sample_idx, len(sample_ids), len(future_ids))
    logger.info("CPT preview: the final %d token(s) of the full sequence are used in causal LM training.", len(future_ids))
    logger.info("  CPT future token ids: %s", future_ids[:64] if len(future_ids) > 64 else future_ids)
    logger.info("  CPT future text:")
    logger.info("  %s", _decode_ids(tokenizer, future_ids))

    sample_masked_bounds = masked_event_boundaries[sample_idx]
    logger.info("JEPA preview: %d token position(s) are masked across %d sampled event(s).", sum(sample_mask), len(sample_masked_bounds))
    if sample_masked_bounds:
        ev_start, eos_pos = sample_masked_bounds[0]
        masked_event_plus_eos = sample_ids[ev_start : eos_pos + 1]
        masked_event_rel_mask_positions = [i for i, flag in enumerate(sample_mask[ev_start : eos_pos + 1]) if flag]
        logger.info("  Example masked event span: start=%d end=%d (inclusive of EOS)", ev_start, eos_pos)
        logger.info("  Masked event+EOS token ids: %s", masked_event_plus_eos[:64] if len(masked_event_plus_eos) > 64 else masked_event_plus_eos)
        logger.info("  JEPA masked event text preview:")
        logger.info("  %s", _render_masked_preview(tokenizer, masked_event_plus_eos, masked_event_rel_mask_positions, mask_placeholder))
    else:
        logger.info("  No complete event was selected for masking in this sample.")

    eos_token_text = _decode_ids(tokenizer, [eos_token_id])
    logger.info("RED preview: eos_token_id=%d  eos_token_text=%r", eos_token_id, eos_token_text)
    if sample_masked_bounds:
        ev_start, eos_pos = sample_masked_bounds[0]
        event_plus_eos = sample_ids[ev_start : eos_pos + 1]
        event_body = sample_ids[ev_start:eos_pos]
        logger.info("  Example masked event span: start=%d end=%d (inclusive of EOS)", ev_start, eos_pos)
        logger.info("  Event+EOS token ids: %s", event_plus_eos[:64] if len(event_plus_eos) > 64 else event_plus_eos)
        logger.info("  Event+EOS text:")
        logger.info("  %s", _decode_ids(tokenizer, event_plus_eos))
        logger.info("  RED target meaning: average the predictor outputs over the masked event tokens and regress that vector to the teacher EOS hidden state.")
        logger.info("  Event body text to be averaged (without EOS):")
        logger.info("  %s", _decode_ids(tokenizer, event_body))
    else:
        logger.info("  No masked complete event was found in the sampled sequence, so RED preview is unavailable for this example.")
    logger.info("=" * 72)


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(output_dir: Path, step: int, micro_step: int, online, predictor, mask_embedding, optimizer, scheduler, args):
    ckpt_dir = output_dir / f"checkpoint-{micro_step}"
    tmp_ckpt_dir = output_dir / f".checkpoint-{micro_step}.tmp"
    if tmp_ckpt_dir.exists():
        import shutil
        shutil.rmtree(tmp_ckpt_dir)
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP and torch.compile before saving
    raw = online.module if isinstance(online, DDP) else online
    raw = getattr(raw, "_orig_mod", raw)
    raw.save_pretrained(tmp_ckpt_dir)
    torch.save(predictor.state_dict(), tmp_ckpt_dir / "predictor.pt")
    torch.save(mask_embedding.state_dict(), tmp_ckpt_dir / "mask_embedding.pt")
    torch.save(
        {
            "step": step,
            "micro_step": micro_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        tmp_ckpt_dir / "trainer_state.pt",
    )
    ready_marker = tmp_ckpt_dir / ".ready"
    ready_marker.write_text("ok\n")
    if ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir)
    tmp_ckpt_dir.rename(ckpt_dir)
    logger.info(f"Checkpoint saved → {ckpt_dir} (optimizer_step={step}, micro_step={micro_step})")


def load_checkpoint(ckpt_dir: Path, predictor, mask_embedding, optimizer, scheduler):
    state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    predictor.load_state_dict(torch.load(ckpt_dir / "predictor.pt", map_location="cpu"))
    mask_path = ckpt_dir / "mask_embedding.pt"
    if mask_path.exists():
        mask_embedding.load_state_dict(torch.load(mask_path, map_location="cpu"))
    else:
        logger.warning("mask_embedding.pt not found in %s; using freshly initialized mask embedding", ckpt_dir)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info(
        "Resumed from optimizer_step=%s micro_step=%s (%s)",
        state.get("step", 0),
        state.get("micro_step", 0),
        ckpt_dir,
    )
    return state.get("step", 0), state.get("micro_step", 0)


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
    p.add_argument("--future_len",  type=int,   default=256,  help="Only used for CPT preview logging")
    p.add_argument("--num_mask_events", type=int, default=1, help="Randomly mask this many complete events per sample")
    p.add_argument("--ema_decay",   type=float, default=0.996)
    p.add_argument("--pred_mlp_ratio", type=float, default=1.0, help="Predictor MLP inner dim ratio")
    # Efficiency
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn",  action="store_true")
    p.add_argument("--compile",     action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    # Checkpointing
    p.add_argument("--save_steps",    type=int, default=0,   help="Save every N optimizer steps (0 = disable)")
    p.add_argument("--save_micro_steps", type=int, default=0,
                   help="Save every N actual micro-batch steps, but checkpoint only after the next optimizer update.")
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

    # ── Tokenizer (no mask token added; JEPA uses a learnable mask embedding) ──
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    eos_token_id  = tokenizer.pad_token_id

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
            num_mask_events=args.num_mask_events,
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
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # ── Predictor (position-wise MLP, online only) ─────────────────────────────
    hidden_size = online.config.hidden_size if not is_ddp else online.module.config.hidden_size
    predictor = Predictor(hidden_size, args.pred_mlp_ratio).to(device).to(dtype)
    mask_embedding = LearnableMaskEmbedding(hidden_size, dtype).to(device)
    if is_ddp:
        predictor = DDP(predictor, device_ids=[local_rank], output_device=local_rank)

    n_online  = sum(p.numel() for p in online.parameters()) / 1e6
    n_pred    = sum(p.numel() for p in predictor.parameters()) / 1e6
    n_mask    = sum(p.numel() for p in mask_embedding.parameters()) / 1e6
    logger.info(f"Online encoder: {n_online:.1f}M  |  Predictor: {n_pred:.2f}M  |  Mask embedding: {n_mask:.4f}M")

    # ── Optimizer ──────────────────────────────────────────────────────────────
    all_params = list(online.parameters()) + list(predictor.parameters()) + list(mask_embedding.parameters())
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
    global_micro_step = 0
    if args.resume_from:
        ckpt = Path(args.resume_from)
        raw = online.module if is_ddp else online
        raw = getattr(raw, "_orig_mod", raw)
        raw.from_pretrained(ckpt)
        global_step, global_micro_step = load_checkpoint(ckpt, predictor, mask_embedding, optimizer, scheduler)

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
    preview_printed = False
    next_micro_save_target = None
    if args.save_micro_steps > 0:
        next_micro_save_target = ((global_micro_step // args.save_micro_steps) + 1) * args.save_micro_steps

    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)

        run_loss = run_cpt = run_jepa = run_red = 0.0
        micro_steps = 0
        optimizer.zero_grad()
        t0: float | None = None

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0), dynamic_ncols=True)

        for batch in pbar:
            if rank == 0 and not preview_printed:
                log_objective_preview(
                    tokenizer,
                    batch,
                    eos_token_id=eos_token_id,
                    future_len=args.future_len,
                )
                preview_printed = True

            full_ids    = batch["full_input_ids"].to(device)
            mask_pos    = batch["masked_positions"].to(device)    # (B, T) bool
            masked_ev_bounds = batch["masked_event_boundaries"]   # list[list[(start, eos_pos)]]
            B, T        = full_ids.shape

            if tokens_per_step is None:
                tokens_per_step = args.batch_size * T * world_size * args.grad_accum

            if micro_steps % args.grad_accum == 0:
                synchronize()
                t0 = time.time()

            # ── Part 1: full-sequence CPT ──────────────────────────────────
            full_out = online(input_ids=full_ids, labels=full_ids)
            loss_cpt = full_out.loss

            loss_jepa = torch.zeros((), device=device, dtype=torch.float32)
            loss_red = torch.zeros((), device=device, dtype=torch.float32)

            # ── Part 2 + Part 3: event-masked predictor/teacher alignment ──
            if args.lambda_jepa > 0 or args.lambda_red > 0:
                embed_tokens = get_raw_online().get_input_embeddings()
                masked_embeds = mask_embedding.apply(full_ids, embed_tokens, mask_pos)
                masked_out = online(
                    inputs_embeds=masked_embeds,
                    output_hidden_states=True,
                )
                online_h = masked_out.hidden_states[-1]          # (B, T, D)
                pred_h = predictor(online_h)                     # (B, T, D)

                with torch.no_grad():
                    teacher_h = teacher(
                        input_ids=full_ids,
                        output_hidden_states=True,
                    ).hidden_states[-1]                          # (B, T, D)

                if args.lambda_jepa > 0:
                    pred_masked = pred_h[mask_pos]               # (N, D)
                    target_masked = teacher_h[mask_pos].detach() # (N, D)
                    if pred_masked.shape[0] > 0:
                        loss_jepa = F.mse_loss(pred_masked, target_masked)

                if args.lambda_red > 0:
                    red_terms = []
                    for b in range(B):
                        for ev_start, eos_pos in masked_ev_bounds[b]:
                            pred_event = pred_h[b, ev_start:eos_pos]
                            if pred_event.shape[0] == 0:
                                continue
                            red_terms.append(
                                F.mse_loss(pred_event.mean(dim=0), teacher_h[b, eos_pos].detach())
                            )
                    if red_terms:
                        loss_red = torch.stack(red_terms).mean()

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
            global_micro_step += 1

            # ── Optimizer step ─────────────────────────────────────────────
            if micro_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(online.parameters()) + list(predictor.parameters()) + list(mask_embedding.parameters()),
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
                    should_save_optimizer = (
                        (args.save_steps > 0 and global_step % args.save_steps == 0) or
                        (args.save_at_steps and global_step in args.save_at_steps)
                    )
                    should_save_micro = (
                        next_micro_save_target is not None and global_micro_step >= next_micro_save_target
                    )
                    if should_save_optimizer or should_save_micro:
                        save_checkpoint(
                            output_dir,
                            global_step,
                            global_micro_step,
                            online,
                            predictor,
                            mask_embedding,
                            optimizer,
                            scheduler,
                            args,
                        )
                        if should_save_micro:
                            while next_micro_save_target is not None and global_micro_step >= next_micro_save_target:
                                next_micro_save_target += args.save_micro_steps

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
        torch.save(mask_embedding.state_dict(), final_dir / "mask_embedding.pt")
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved → {final_dir}")
        logger.info(f"Total training time (excl. first {TIMING_WARMUP} steps): {total_train_time/60:.2f}m")
        if use_wandb:
            wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
