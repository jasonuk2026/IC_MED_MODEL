#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_warning()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


class PatientEventDataset(Dataset):
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
        logger.info("Dataset: %s rows across %d row groups", f"{total:,}", len(offsets))
        self._pf = None
        self._cached_rg_idx = -1
        self._cached_col = None

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> list[list[int]]:
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
            rg = self._pf.read_row_group(rg_idx, columns=["event_token_ids"])
            self._cached_col = rg.column("event_token_ids")
            self._cached_rg_idx = rg_idx
        return [[int(x) for x in ev] for ev in self._cached_col[row_in_rg].as_py()]


def truncate_tokens(token_ids: list[int], max_event_tokens: int, truncate_side: str) -> list[int]:
    if len(token_ids) <= max_event_tokens:
        return token_ids
    if truncate_side == "last":
        return token_ids[-max_event_tokens:]
    return token_ids[:max_event_tokens]


def collate_concat_event_sequences(
    batch: list[list[list[int]]],
    *,
    pad_token_id: int,
    max_events: int,
    max_event_tokens: int,
    sequence_truncate_side: str,
    event_truncate_side: str,
):
    batch_size = len(batch)
    sample_token_ids: list[list[int]] = []
    sample_token_event_idx: list[list[int]] = []
    sequence_event_mask = torch.zeros((batch_size, max_events), dtype=torch.long)
    num_real_events = 0
    max_total_tokens = 1

    for sample_idx, sample_events in enumerate(batch):
        events = sample_events
        if len(events) > max_events:
            events = events[-max_events:] if sequence_truncate_side == "last" else events[:max_events]
        concat_ids: list[int] = []
        concat_event_idx: list[int] = []
        for event_idx, event_ids in enumerate(events[:max_events]):
            if not event_ids:
                continue
            ids = truncate_tokens(event_ids, max_event_tokens, event_truncate_side)
            concat_ids.extend(ids)
            concat_event_idx.extend([event_idx] * len(ids))
            sequence_event_mask[sample_idx, event_idx] = 1
            num_real_events += 1
        sample_token_ids.append(concat_ids)
        sample_token_event_idx.append(concat_event_idx)
        max_total_tokens = max(max_total_tokens, len(concat_ids))

    input_ids = torch.full((batch_size, max_total_tokens), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_total_tokens), dtype=torch.long)
    token_event_index = torch.full((batch_size, max_total_tokens), -1, dtype=torch.long)

    for sample_idx, (ids, event_idx) in enumerate(zip(sample_token_ids, sample_token_event_idx)):
        if not ids:
            continue
        n = len(ids)
        input_ids[sample_idx, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[sample_idx, :n] = 1
        token_event_index[sample_idx, :n] = torch.tensor(event_idx, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_event_index": token_event_index,
        "sequence_event_mask": sequence_event_mask,
        "num_real_events": int(num_real_events),
    }


class SmallCausalTransformerPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        max_events: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position_embed = nn.Embedding(max_events, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, event_embeddings: torch.Tensor, sequence_event_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = event_embeddings.shape
        position_ids = torch.arange(seq_len, device=event_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.input_proj(event_embeddings) + self.position_embed(position_ids)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=event_embeddings.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=(sequence_event_mask == 0),
        )
        hidden = self.norm(hidden)
        return self.output_proj(hidden)


class NextEventConcatMeanModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        max_events: int,
        predictor_hidden_size: int,
        predictor_num_heads: int,
        predictor_ffn_dim: int,
        predictor_num_layers: int,
        predictor_dropout: float,
        freeze_event_encoder: bool,
        torch_dtype: torch.dtype,
        attn_implementation: str,
        local_files_only: bool,
    ):
        super().__init__()
        self.event_encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
            use_cache=False,
        )
        self.hidden_size = int(self.event_encoder.config.hidden_size)
        self.freeze_event_encoder = bool(freeze_event_encoder)
        if self.freeze_event_encoder:
            for p in self.event_encoder.parameters():
                p.requires_grad = False
            self.event_encoder.eval()

        if predictor_hidden_size <= 0:
            raise ValueError("predictor_hidden_size must be > 0")
        if predictor_hidden_size % predictor_num_heads != 0:
            raise ValueError("predictor_hidden_size must be divisible by predictor_num_heads")
        predictor_ffn_dim = predictor_hidden_size if predictor_ffn_dim <= 0 else predictor_ffn_dim

        self.predictor = SmallCausalTransformerPredictor(
            input_dim=self.hidden_size,
            hidden_dim=predictor_hidden_size,
            num_heads=predictor_num_heads,
            ffn_dim=predictor_ffn_dim,
            num_layers=predictor_num_layers,
            max_events=max_events,
            dropout=predictor_dropout,
        )

    def save_checkpoint(self, save_dir: Path, args_dict: dict):
        save_dir.mkdir(parents=True, exist_ok=True)
        event_encoder = self.event_encoder
        if hasattr(event_encoder, "_orig_mod"):
            event_encoder = event_encoder._orig_mod
        payload = {
            "state_dict": self.state_dict(),
            "args": args_dict,
            "event_encoder_config": event_encoder.config.to_dict(),
        }
        torch.save(payload, save_dir / "model.pt")
        (save_dir / "training_args.json").write_text(json.dumps(args_dict, indent=2, sort_keys=True))
        logger.info("Saved checkpoint -> %s", save_dir)

    def pool_events_from_tokens(
        self,
        token_hidden: torch.Tensor,
        token_event_index: torch.Tensor,
        sequence_event_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, hidden_size = token_hidden.shape
        max_events = sequence_event_mask.shape[1]
        flat_hidden = token_hidden.reshape(-1, hidden_size)
        flat_event_index = token_event_index.reshape(-1)
        sample_ids = torch.arange(batch_size, device=token_hidden.device).unsqueeze(1).expand_as(token_event_index)
        flat_sample_ids = sample_ids.reshape(-1)
        valid = flat_event_index >= 0
        seg_ids = flat_sample_ids[valid] * max_events + flat_event_index[valid]

        sums = torch.zeros(batch_size * max_events, hidden_size, dtype=token_hidden.dtype, device=token_hidden.device)
        counts = torch.zeros(batch_size * max_events, 1, dtype=token_hidden.dtype, device=token_hidden.device)
        sums.index_add_(0, seg_ids, flat_hidden[valid])
        counts.index_add_(0, seg_ids, torch.ones((int(valid.sum().item()), 1), dtype=token_hidden.dtype, device=token_hidden.device))
        pooled = sums / counts.clamp(min=1.0)
        return pooled.reshape(batch_size, max_events, hidden_size)

    def encode_sequence_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_event_index: torch.Tensor,
        sequence_event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_ctx = torch.no_grad() if self.freeze_event_encoder else torch.enable_grad()
        with encoder_ctx:
            outputs = self.event_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
            token_hidden = outputs.last_hidden_state
        if self.freeze_event_encoder:
            token_hidden = token_hidden.detach()
        event_embeddings = self.pool_events_from_tokens(token_hidden, token_event_index, sequence_event_mask)
        seq_embeddings = self.predictor(event_embeddings, sequence_event_mask)
        return seq_embeddings, event_embeddings

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_event_index: torch.Tensor,
        sequence_event_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        seq_hidden, event_embeddings = self.encode_sequence_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_event_index=token_event_index,
            sequence_event_mask=sequence_event_mask,
        )
        pred = seq_hidden[:, :-1, :]
        target = event_embeddings[:, 1:, :].detach()
        valid_pairs = (sequence_event_mask[:, :-1] > 0) & (sequence_event_mask[:, 1:] > 0)
        if not valid_pairs.any():
            zero = pred.sum() * 0.0
            return {
                "loss": zero,
                "mean_cosine": zero,
                "num_pairs": 0.0,
            }
        cosine = F.cosine_similarity(pred[valid_pairs], target[valid_pairs], dim=-1)
        loss = -cosine.mean()
        return {
            "loss": loss,
            "mean_cosine": cosine.mean(),
            "num_pairs": float(valid_pairs.sum().item()),
        }


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train next-event prediction by concatenating all event tokens per sample, mean-pooling per event, and predicting next event embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_event_tokens", type=int, default=32)
    p.add_argument("--max_events", type=int, default=1024)
    p.add_argument("--sequence_truncate_side", choices=["first", "last"], default="last")
    p.add_argument("--event_truncate_side", choices=["first", "last"], default="last")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--freeze_event_encoder", action="store_true")
    p.add_argument("--predictor_hidden_size", type=int, default=128)
    p.add_argument("--predictor_num_heads", type=int, default=4)
    p.add_argument("--predictor_ffn_dim", type=int, default=0, help="0 means use predictor_hidden_size.")
    p.add_argument("--predictor_num_layers", type=int, default=1)
    p.add_argument("--predictor_dropout", type=float, default=0.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_ddp:
        dist.init_process_group(backend="nccl", device_id=device)
    if rank != 0:
        logging.disable(logging.CRITICAL)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("World size: %d | device: %s", world_size, device)

    use_wandb = (args.wandb_project is not None) and (rank == 0)
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"next-event-concat-mean-{datetime.now().strftime('%m%d-%H%M')}",
            tags=args.wandb_tags,
            config=vars(args),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(f"Tokenizer {args.model_name} has no pad_token_id")

    dataset = PatientEventDataset(args.data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=lambda batch: collate_concat_event_sequences(
            batch,
            pad_token_id=pad_token_id,
            max_events=args.max_events,
            max_event_tokens=args.max_event_tokens,
            sequence_truncate_side=args.sequence_truncate_side,
            event_truncate_side=args.event_truncate_side,
        ),
    )

    torch_dtype = torch.float32
    autocast_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
        autocast_dtype = torch.float16

    attn_implementation = "flash_attention_2" if args.flash_attn else "eager"
    logger.info("Loading concat-mean event encoder backbone: %s", args.model_name)
    model = NextEventConcatMeanModel(
        model_name=args.model_name,
        max_events=args.max_events,
        predictor_hidden_size=args.predictor_hidden_size,
        predictor_num_heads=args.predictor_num_heads,
        predictor_ffn_dim=args.predictor_ffn_dim,
        predictor_num_layers=args.predictor_num_layers,
        predictor_dropout=args.predictor_dropout,
        freeze_event_encoder=args.freeze_event_encoder,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        local_files_only=args.local_files_only,
    ).to(device)

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        logger.info(
            "Model: total_params=%.1fM trainable_params=%.3fM freeze_event_encoder=%s predictor_hidden=%d predictor_layers=%d predictor_heads=%d max_events=%d max_event_tokens=%d compile=%s",
            total_params,
            trainable_params,
            args.freeze_event_encoder,
            args.predictor_hidden_size,
            args.predictor_num_layers,
            args.predictor_num_heads,
            args.max_events,
            args.max_event_tokens,
            args.compile,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    best_loss = float("inf")
    raw_model = unwrap_model(model)
    if raw_model.freeze_event_encoder:
        raw_model.event_encoder.eval()
    model.train()

    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)
        if raw_model.freeze_event_encoder:
            raw_model.event_encoder.eval()
        optimizer.zero_grad()
        run_loss = 0.0
        run_cos = 0.0
        run_pairs = 0.0
        run_batches = 0
        epoch_loss_sum = 0.0
        epoch_cos_sum = 0.0
        epoch_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=(rank != 0), dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            if batch["num_real_events"] == 0:
                continue
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            autocast_enabled = torch.cuda.is_available() and (autocast_dtype is not None)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_event_index=batch["token_event_index"],
                    sequence_event_mask=batch["sequence_event_mask"],
                )
                loss = outputs["loss"] / args.grad_accum
            loss.backward()

            run_loss += float(outputs["loss"].item())
            run_cos += float(outputs["mean_cosine"].item())
            run_pairs += float(outputs["num_pairs"])
            run_batches += 1
            epoch_loss_sum += float(outputs["loss"].item())
            epoch_cos_sum += float(outputs["mean_cosine"].item())
            epoch_batches += 1

            if ((batch_idx + 1) % args.grad_accum == 0) or ((batch_idx + 1) == len(loader)):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % args.log_steps == 0:
                    avg_loss = run_loss / max(run_batches, 1)
                    avg_cos = run_cos / max(run_batches, 1)
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        cosine=f"{avg_cos:.4f}",
                        pairs=f"{int(run_pairs)}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": avg_loss,
                                "train/mean_cosine": avg_cos,
                                "train/num_pairs": run_pairs / max(run_batches, 1),
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )
                    run_loss = 0.0
                    run_cos = 0.0
                    run_pairs = 0.0
                    run_batches = 0

        if rank == 0:
            raw_model = unwrap_model(model)
            epoch_avg_loss = epoch_loss_sum / max(epoch_batches, 1)
            epoch_avg_cos = epoch_cos_sum / max(epoch_batches, 1)
            logger.info(
                "Epoch %d/%d summary: avg_loss=%.4f avg_cosine=%.4f",
                epoch + 1,
                args.epochs,
                epoch_avg_loss,
                epoch_avg_cos,
            )
            if args.save_every_epoch:
                raw_model.save_checkpoint(output_dir / f"epoch_{epoch + 1}", vars(args))
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                raw_model.save_checkpoint(output_dir / "best", vars(args))

    if rank == 0:
        raw_model = unwrap_model(model)
        raw_model.save_checkpoint(output_dir / "final", vars(args))
        if wandb_run is not None:
            wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
