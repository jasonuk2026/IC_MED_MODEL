#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_cross_attention_classifier import DiseaseEventCrossAttentionClassifier
from train_disease_concat_classifier import SampleLevelPreparedDataset, evaluate_classifier, _log_eval
from train_embedding_disease_cond_v2 import (
    BERT_DIM,
    TASK_2_DISEASE_NAME,
    EmbeddingStore,
    build_task_text_embs,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _compute_pos_weight(train_data_dir: str, tasks: list[str], selected_epochs: list[int]) -> tuple[float, int, int]:
    pos = 0
    neg = 0
    for task in tasks:
        task_dir = Path(train_data_dir) / task
        for data_epoch in selected_epochs:
            path = task_dir / f"train_prepared_{data_epoch:03d}.parquet"
            if not path.exists():
                raise ValueError(f"Missing prepared file: {path}")
            labels = pd.read_parquet(path, columns=["label"])["label"].to_numpy()
            pos += int((labels == 1).sum())
            neg += int((labels == 0).sum())
    if pos <= 0:
        raise ValueError("Cannot auto-compute pos_weight: no positive samples found.")
    return float(neg / pos), pos, neg


def parse_args():
    p = argparse.ArgumentParser(
        description="Disease query cross-attention + gated fusion classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--train_data_dir", default=None)
    p.add_argument("--train_data_epochs", nargs="+", type=int, default=None)
    p.add_argument("--tasks", nargs="+", default=list(sorted(TASK_2_DISEASE_NAME.keys())))
    p.add_argument("--eval_data_paths", nargs="+", default=None)
    p.add_argument("--bert_embeddings", required=True)
    p.add_argument("--disease_model_name", default="michiyasunaga/BioLinkBERT-base")

    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--patient_layers", type=int, default=1)
    p.add_argument("--head_layers", type=int, default=1)
    p.add_argument("--intermediate_size", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--output_dir", default="output/disease-cross-attn-classifier")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos_weight", type=float, default=None)
    p.add_argument("--pad_to_num_events", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)

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
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if not args.eval_only and args.train_data_dir is None:
        raise ValueError("--train_data_dir is required unless --eval_only is set.")
    if args.eval_data_paths is None:
        raise ValueError("--eval_data_paths is required")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run dir: %s", run_dir)

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
        logger.info("wandb run: %s", wandb_run.url)

    store = EmbeddingStore(args.bert_embeddings)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    task_text_embs = build_task_text_embs(args, device, rank, is_ddp)

    if args.checkpoint:
        model = DiseaseEventCrossAttentionClassifier.load_checkpoint(
            Path(args.checkpoint),
            task_text_embs=task_text_embs,
            device=device,
            dtype=dtype,
        )
    else:
        model = DiseaseEventCrossAttentionClassifier(
            bert_dim=BERT_DIM,
            task_text_embs=task_text_embs,
            hidden_size=args.hidden_size,
            patient_layers=args.patient_layers,
            head_layers=args.head_layers,
            intermediate_size=args.intermediate_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
            dtype=dtype,
        ).to(device)

    if rank == 0:
        logger.info(
            "Cross-attn classifier: hidden=%d patient_layers=%d head_layers=%d heads=%d intermediate=%s dropout=%.3f dtype=%s",
            args.hidden_size,
            args.patient_layers,
            args.head_layers,
            args.num_heads,
            args.intermediate_size or (args.hidden_size * 4),
            args.dropout,
            dtype,
        )
        logger.info("Training batches use sample-level shuffle; a batch may mix multiple tasks.")

    if is_ddp and not args.eval_only:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, static_graph=True)

    if args.eval_only:
        overall, per_task = evaluate_classifier(
            model,
            eval_data_paths=args.eval_data_paths,
            store=store,
            args=args,
            device=device,
        )
        if rank == 0:
            _log_eval("", overall, per_task)
        if is_ddp:
            dist.destroy_process_group()
        return

    first_task = sorted(args.tasks)[0]
    task_dir = Path(args.train_data_dir) / first_task
    data_epoch_files = sorted(task_dir.glob("train_prepared_*.parquet"))
    available_epochs = sorted(int(p.stem.split("_")[-1]) for p in data_epoch_files)
    if not available_epochs:
        raise ValueError(f"No train_prepared_*.parquet found in {task_dir}")
    if args.train_data_epochs is not None:
        selected_epochs = list(dict.fromkeys(args.train_data_epochs))
        missing = [ep for ep in selected_epochs if ep not in available_epochs]
        if missing:
            raise ValueError(f"Requested epochs {missing}, available={available_epochs}")
    else:
        selected_epochs = available_epochs[: args.epochs]

    if rank == 0:
        logger.info("Selected train data epoch(s): %s", selected_epochs)

    if args.pos_weight is None:
        args.pos_weight, train_pos, train_neg = _compute_pos_weight(args.train_data_dir, args.tasks, selected_epochs)
    else:
        train_pos = None
        train_neg = None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )

    schedule_lengths = []
    for schedule_idx, data_epoch in enumerate(selected_epochs):
        ds = SampleLevelPreparedDataset(
            args.train_data_dir, data_epoch, args.tasks, store,
            args.batch_size, schedule_idx, args.seed, world_size, rank,
            args.pad_to_num_events,
        )
        schedule_lengths.append(len(ds))
    total_steps = sum(math.ceil(n / args.grad_accum) for n in schedule_lengths)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    pos_weight = torch.tensor(args.pos_weight, device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if rank == 0:
        if train_pos is not None and train_neg is not None:
            logger.info(
                "Using auto pos_weight=%.4f from training counts pos=%d neg=%d",
                args.pos_weight,
                train_pos,
                train_neg,
            )
        else:
            logger.info("Using manual pos_weight=%.4f", args.pos_weight)

    global_step = 0
    best_auc = float("-inf")

    for schedule_idx, data_epoch in enumerate(selected_epochs):
        if rank == 0:
            logger.info("Pass %d/%d: prepared data epoch %d …", schedule_idx + 1, len(selected_epochs), data_epoch)

        epoch_ds = SampleLevelPreparedDataset(
            args.train_data_dir, data_epoch, args.tasks, store,
            args.batch_size, schedule_idx, args.seed, world_size, rank,
            args.pad_to_num_events,
        )
        train_loader = DataLoader(
            epoch_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda b: b[0],
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {schedule_idx + 1}/{len(selected_epochs)}", disable=(rank != 0), dynamic_ncols=True, total=len(epoch_ds))

        for batch_idx, batch in enumerate(pbar):
            labels = batch["labels"].to(device).float()
            logits = model(
                batch["event_embs"].to(device),
                batch["event_mask"].to(device),
                batch["task_idxs"].to(device),
            )
            loss = loss_fn(logits, labels) / args.grad_accum
            loss.backward()

            is_update_step = ((batch_idx + 1) % args.grad_accum == 0) or ((batch_idx + 1) == len(epoch_ds))
            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
                if global_step < warmup_steps:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            else:
                grad_norm = float("nan")

            epoch_loss += loss.item() * args.grad_accum
            if rank == 0:
                probs = torch.sigmoid(logits.float())
                pbar.set_postfix(
                    loss=f"{loss.item() * args.grad_accum:.4f}",
                    posp=f"{probs[labels == 1].mean().item():.3f}" if (labels == 1).any() else "nan",
                    negp=f"{probs[labels == 0].mean().item():.3f}" if (labels == 0).any() else "nan",
                    gnorm=f"{float(grad_norm):.3f}" if not isinstance(grad_norm, float) or not math.isnan(grad_norm) else "nan",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if rank == 0:
            logger.info("Epoch %d/%d avg_loss=%.4f", schedule_idx + 1, len(selected_epochs), epoch_loss / max(len(epoch_ds), 1))
        overall, per_task = evaluate_classifier(
            model,
            eval_data_paths=args.eval_data_paths,
            store=store,
            args=args,
            device=device,
        )
        if rank == 0:
            _log_eval("  ", overall, per_task)
            raw_model = model.module if isinstance(model, DDP) else model
            epoch_dir = run_dir / f"epoch_{schedule_idx + 1}"
            raw_model.save_checkpoint(epoch_dir)
            if overall["auc"] > best_auc:
                best_auc = overall["auc"]
                raw_model.save_checkpoint(run_dir / "best")
            if use_wandb:
                import wandb
                log_dict = {
                    "epoch": schedule_idx + 1,
                    "train/avg_loss": epoch_loss / max(len(epoch_ds), 1),
                    "eval/auc": overall["auc"],
                    "eval/accuracy": overall["accuracy"],
                    "eval/pos_prob_mean": overall["pos_prob_mean"],
                    "eval/neg_prob_mean": overall["neg_prob_mean"],
                }
                for task, stats in per_task.items():
                    log_dict[f"eval/{task}/auc"] = stats["auc"]
                    log_dict[f"eval/{task}/accuracy"] = stats["accuracy"]
                wandb.log(log_dict)

    if use_wandb:
        wandb_run.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
