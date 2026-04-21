#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time
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
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from transformers.models.qwen3.modeling_qwen3 import create_causal_mask, create_sliding_window_causal_mask

hf_logging.set_verbosity_warning()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
        logger.info("Dataset: %s blocks across %d row groups", f"{total:,}", len(offsets))
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


def split_events(input_ids: torch.Tensor, eos_token_id: int) -> list[list[int]]:
    ids = input_ids.tolist()
    events = []
    start = 0
    for pos, tok in enumerate(ids):
        if tok == eos_token_id:
            if pos >= start:
                event = ids[start : pos + 1]
                if event:
                    events.append(event)
            start = pos + 1
    return events


def truncate_event_tokens(event_ids: list[int], max_event_tokens: int, eos_token_id: int) -> list[int]:
    if len(event_ids) <= max_event_tokens:
        return event_ids
    trimmed = event_ids[-max_event_tokens:]
    if trimmed[-1] != eos_token_id:
        trimmed[-1] = eos_token_id
    return trimmed


def collate_event_sequences(
    batch: list[torch.Tensor],
    *,
    eos_token_id: int,
    max_events: int,
    max_event_tokens: int,
    truncate_side: str,
):
    patient_events = []
    for sample in batch:
        events = split_events(sample, eos_token_id)
        if max_events is not None and len(events) > max_events:
            events = events[-max_events:] if truncate_side == "last" else events[:max_events]
        events = [truncate_event_tokens(ev, max_event_tokens, eos_token_id) for ev in events]
        if len(events) < 2:
            events = []
        patient_events.append(events)

    batch_size = len(patient_events)
    max_num_events = max((len(events) for events in patient_events), default=0)
    if max_num_events == 0:
        max_num_events = 1

    flat_events = []
    flat_to_sample = []
    flat_to_event_idx = []
    for sample_idx, events in enumerate(patient_events):
        for event_idx, event_ids in enumerate(events):
            flat_events.append(event_ids)
            flat_to_sample.append(sample_idx)
            flat_to_event_idx.append(event_idx)

    if flat_events:
        max_len = max(len(ev) for ev in flat_events)
        event_input_ids = torch.full((len(flat_events), max_len), eos_token_id, dtype=torch.long)
        event_attention_mask = torch.zeros((len(flat_events), max_len), dtype=torch.long)
        event_last_pos = torch.zeros(len(flat_events), dtype=torch.long)
        for row_idx, event_ids in enumerate(flat_events):
            n = len(event_ids)
            event_input_ids[row_idx, :n] = torch.tensor(event_ids, dtype=torch.long)
            event_attention_mask[row_idx, :n] = 1
            event_last_pos[row_idx] = n - 1
    else:
        event_input_ids = torch.full((1, 1), eos_token_id, dtype=torch.long)
        event_attention_mask = torch.zeros((1, 1), dtype=torch.long)
        event_last_pos = torch.zeros(1, dtype=torch.long)

    sequence_event_mask = torch.zeros((batch_size, max_num_events), dtype=torch.long)
    sequence_index_map = torch.full((batch_size, max_num_events), -1, dtype=torch.long)
    for flat_idx, (sample_idx, event_idx) in enumerate(zip(flat_to_sample, flat_to_event_idx)):
        sequence_event_mask[sample_idx, event_idx] = 1
        sequence_index_map[sample_idx, event_idx] = flat_idx

    return {
        "event_input_ids": event_input_ids,
        "event_attention_mask": event_attention_mask,
        "event_last_pos": event_last_pos,
        "sequence_event_mask": sequence_event_mask,
        "sequence_index_map": sequence_index_map,
        "num_real_events": int(len(flat_events)),
    }


def build_mask_mapping(config, hidden_states, attention_mask, position_ids):
    mask_kwargs = {
        "config": config,
        "inputs_embeds": hidden_states,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    mask_map = {"full_attention": create_causal_mask(**mask_kwargs)}
    if "sliding_attention" in config.layer_types:
        mask_map["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return mask_map


class NextEventCosineModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        event_encoder_layers: int,
        torch_dtype: torch.dtype,
        attn_implementation: str,
        local_files_only: bool,
    ):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        self.model = self.backbone.model
        self.event_encoder_layers = event_encoder_layers
        self.hidden_size = self.backbone.config.hidden_size

        total_layers = len(self.model.layers)
        if event_encoder_layers <= 0 or event_encoder_layers >= total_layers:
            raise ValueError(
                "event_encoder_layers must be in [1, %d), got %d" % (total_layers, event_encoder_layers)
            )
        self.total_layers = total_layers

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        logger.info("Saved checkpoint -> %s", save_dir)

    def _run_layers(self, hidden_states, attention_mask, position_ids, start_idx: int, end_idx: int):
        mask_map = build_mask_mapping(self.backbone.config, hidden_states, attention_mask, position_ids)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for layer_idx in range(start_idx, end_idx):
            layer = self.model.layers[layer_idx]
            hidden_states = layer(
                hidden_states,
                attention_mask=mask_map[self.backbone.config.layer_types[layer_idx]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
        return hidden_states

    def encode_events(self, event_input_ids, event_attention_mask, event_last_pos):
        hidden_states = self.model.embed_tokens(event_input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        hidden_states = self._run_layers(
            hidden_states,
            event_attention_mask,
            position_ids,
            0,
            self.event_encoder_layers,
        )
        row_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[row_idx, event_last_pos]

    def model_event_sequence(self, event_embeddings, sequence_event_mask):
        position_ids = torch.arange(event_embeddings.shape[1], device=event_embeddings.device).unsqueeze(0)
        hidden_states = self._run_layers(
            event_embeddings,
            sequence_event_mask,
            position_ids,
            self.event_encoder_layers,
            self.total_layers,
        )
        return hidden_states

    def forward(
        self,
        *,
        event_input_ids,
        event_attention_mask,
        event_last_pos,
        sequence_event_mask,
        sequence_index_map,
    ):
        real_event_embeddings = self.encode_events(event_input_ids, event_attention_mask, event_last_pos)
        batch_size, max_num_events = sequence_index_map.shape
        event_embeddings = torch.zeros(
            batch_size,
            max_num_events,
            self.hidden_size,
            dtype=real_event_embeddings.dtype,
            device=real_event_embeddings.device,
        )
        valid = sequence_index_map >= 0
        if valid.any():
            event_embeddings[valid] = real_event_embeddings[sequence_index_map[valid]]

        seq_hidden = self.model_event_sequence(event_embeddings, sequence_event_mask)
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
        description="Train a next-event predictor by splitting Qwen into event encoder layers and sequence modeling layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--event_encoder_layers", type=int, default=2)
    p.add_argument("--max_event_tokens", type=int, default=128)
    p.add_argument("--max_events", type=int, default=1024)
    p.add_argument("--truncate_side", choices=["first", "last"], default="last")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--event_encoder_batch_size", type=int, default=0,
                   help="Reserved for future chunking; 0 means encode all events in a training batch together.")
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
    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

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
            name=args.wandb_run_name or f"next-event-cosine-{datetime.now().strftime('%m%d-%H%M')}",
            tags=args.wandb_tags,
            config=vars(args),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer %s has no pad_token_id/eos delimiter configured" % args.model_name)

    dataset = CPTDataset(args.data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=lambda batch: collate_event_sequences(
            batch,
            eos_token_id=eos_token_id,
            max_events=args.max_events,
            max_event_tokens=args.max_event_tokens,
            truncate_side=args.truncate_side,
        ),
    )

    torch_dtype = torch.float32
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    attn_implementation = "flash_attention_2" if args.flash_attn else "eager"
    logger.info("Loading backbone: %s", args.model_name)
    model = NextEventCosineModel(
        model_name=args.model_name,
        event_encoder_layers=args.event_encoder_layers,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        local_files_only=args.local_files_only,
    ).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        logger.info(
            "Next-event cosine model: params=%.1fM event_encoder_layers=%d max_event_tokens=%d max_events=%d",
            n_params,
            args.event_encoder_layers,
            args.max_event_tokens,
            args.max_events,
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
    model.train()

    for epoch in range(args.epochs):
        if is_ddp:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()
        run_loss = 0.0
        run_cos = 0.0
        run_pairs = 0.0
        epoch_loss_sum = 0.0
        epoch_cos_sum = 0.0
        epoch_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=(rank != 0), dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            if batch["num_real_events"] == 0:
                continue
            outputs = model(
                event_input_ids=batch["event_input_ids"].to(device),
                event_attention_mask=batch["event_attention_mask"].to(device),
                event_last_pos=batch["event_last_pos"].to(device),
                sequence_event_mask=batch["sequence_event_mask"].to(device),
                sequence_index_map=batch["sequence_index_map"].to(device),
            )
            loss = outputs["loss"] / args.grad_accum
            loss.backward()

            run_loss += float(outputs["loss"].item())
            run_cos += float(outputs["mean_cosine"].item())
            run_pairs += float(outputs["num_pairs"])
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
                    avg_loss = run_loss / max(args.log_steps, 1)
                    avg_cos = run_cos / max(args.log_steps, 1)
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
                                "train/num_pairs": run_pairs / max(args.log_steps, 1),
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )
                    run_loss = 0.0
                    run_cos = 0.0
                    run_pairs = 0.0

        if rank == 0:
            raw_model = model.module if isinstance(model, DDP) else model
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
                raw_model.save_checkpoint(output_dir / f"epoch_{epoch + 1}")
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                raw_model.save_checkpoint(output_dir / "best")

    if rank == 0:
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.save_checkpoint(output_dir / "final")
        if wandb_run is not None:
            wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
