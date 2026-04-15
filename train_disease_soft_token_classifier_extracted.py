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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_soft_token_classifier import DiseaseEventSoftTokenClassifier
from train_disease_concat_classifier import _log_eval
from train_embedding_disease_cond_v2 import (
    TASK_2_DISEASE_NAME,
    TASK_2_IDX,
    EmbeddingStore,
    _binary_roc_auc,
    build_task_text_embs,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


DEFAULT_TASKS = list(sorted(TASK_2_DISEASE_NAME.keys()))


def parse_binary_label(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value != 0)
    if isinstance(value, (float, np.floating)):
        return int(float(value) != 0.0)

    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return 1
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return 0

    try:
        return int(float(text) != 0.0)
    except ValueError as exc:
        raise ValueError(f"Unsupported binary label value: {value!r}") from exc


def collate_latest_leftpad(
    eids_list: list[np.ndarray],
    task_idxs: list[int],
    labels: list[int] | None,
    embeddings: np.ndarray,
    max_events: int,
) -> dict[str, torch.Tensor]:
    bert_dim = embeddings.shape[1]
    batch_size = len(eids_list)
    padded = np.zeros((batch_size, max_events, bert_dim), dtype=np.float32)
    mask = np.zeros((batch_size, max_events), dtype=np.int64)

    for i, eids in enumerate(eids_list):
        if len(eids) > max_events:
            eids = eids[-max_events:]
        if len(eids) == 0:
            continue
        embs = embeddings[eids].astype(np.float32)
        n = embs.shape[0]
        padded[i, max_events - n :] = embs
        mask[i, max_events - n :] = 1

    out = {
        "event_embs": torch.from_numpy(padded),
        "event_mask": torch.from_numpy(mask),
        "task_idxs": torch.tensor(task_idxs, dtype=torch.long),
    }
    if labels is not None:
        out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out


class ExtractedTaskDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        tasks: list[str],
        store: EmbeddingStore,
        batch_size: int,
        training_epoch: int,
        seed: int,
        world_size: int,
        rank: int,
        max_events: int,
    ):
        self.embeddings = store.embeddings
        self.max_events = max_events
        self.batch_size = batch_size
        rows: list[tuple[np.ndarray, int, int]] = []
        for task in sorted(tasks):
            p_parquet = Path(data_dir) / task / f"{split}.parquet"
            if not p_parquet.exists():
                raise FileNotFoundError(f"Missing extracted task parquet: {p_parquet}")
            df = pd.read_parquet(p_parquet, columns=["label", "event_ids"])
            for row in df.itertuples(index=False):
                rows.append((
                    np.array(row.event_ids, dtype=np.int32),
                    int(TASK_2_IDX[task]),
                    parse_binary_label(row.label),
                ))

        rng = random.Random(seed + training_epoch * 1337)
        rng.shuffle(rows)
        if world_size > 1:
            rows = rows[rank::world_size]
        self.samples = rows

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.batch_size
        end = min(len(self.samples), start + self.batch_size)
        batch_rows = self.samples[start:end]
        eids_list = [x[0] for x in batch_rows]
        task_idxs = [x[1] for x in batch_rows]
        labels = [x[2] for x in batch_rows]
        return collate_latest_leftpad(
            eids_list=eids_list,
            task_idxs=task_idxs,
            labels=labels,
            embeddings=self.embeddings,
            max_events=self.max_events,
        )


@torch.inference_mode()
def evaluate_classifier(
    model: torch.nn.Module,
    *,
    eval_data_dir: str,
    eval_split: str,
    tasks: list[str],
    store: EmbeddingStore,
    args,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    rows: list[tuple[np.ndarray, int, int]] = []
    tasks_sorted = sorted(tasks)
    if rank == 0:
        logger.info("Preparing %s split for evaluation across %d rank(s) ...", eval_split, world_size)
    for task in tqdm(
        tasks_sorted,
        desc=f"Loading {eval_split} parquet",
        disable=(rank != 0),
        dynamic_ncols=True,
    ):
        path = Path(eval_data_dir) / task / f"{eval_split}.parquet"
        df = pd.read_parquet(path, columns=["label", "event_ids"])
        for row_idx, row in enumerate(df.itertuples(index=False)):
            if row_idx % world_size != rank:
                continue
            rows.append((
                np.array(row.event_ids, dtype=np.int32),
                int(TASK_2_IDX[task]),
                parse_binary_label(row.label),
            ))
    local_num_rows = len(rows)
    if rank == 0:
        logger.info(
            "Evaluation data prepared: local shard has %d samples on rank 0 (global total will be gathered after forward).",
            local_num_rows,
        )

    all_logits = []
    all_labels = []
    all_task_idxs = []
    num_batches = math.ceil(local_num_rows / args.eval_batch_size) if local_num_rows > 0 else 0
    for start in tqdm(
        range(0, local_num_rows, args.eval_batch_size),
        desc=f"Evaluating {eval_split}",
        disable=(rank != 0),
        dynamic_ncols=True,
        total=num_batches,
    ):
        sub = rows[start : start + args.eval_batch_size]
        eids_list = [x[0] for x in sub]
        task_idxs = [x[1] for x in sub]
        labels = [x[2] for x in sub]
        batch = collate_latest_leftpad(
            eids_list=eids_list,
            task_idxs=task_idxs,
            labels=labels,
            embeddings=store.embeddings,
            max_events=args.max_events,
        )
        labels = batch["labels"]
        logits, _, _, _ = raw_model(
            batch["event_embs"].to(device),
            batch["event_mask"].to(device),
            batch["task_idxs"].to(device),
            return_aux_logits=True,
        )
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_task_idxs.append(batch["task_idxs"].cpu())

    if all_logits:
        local_logits = torch.cat(all_logits, dim=0)
        local_labels = torch.cat(all_labels, dim=0)
        local_task_idxs = torch.cat(all_task_idxs, dim=0)
    else:
        local_logits = torch.empty(0, dtype=torch.float32)
        local_labels = torch.empty(0, dtype=torch.long)
        local_task_idxs = torch.empty(0, dtype=torch.long)

    if world_size > 1:
        gathered: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            (
                local_logits.numpy(),
                local_labels.numpy(),
                local_task_idxs.numpy(),
            ),
        )
        logits_parts = [torch.from_numpy(x[0]) for x in gathered if x is not None and len(x[0]) > 0]
        labels_parts = [torch.from_numpy(x[1]) for x in gathered if x is not None and len(x[1]) > 0]
        task_idx_parts = [torch.from_numpy(x[2]) for x in gathered if x is not None and len(x[2]) > 0]
        logits = torch.cat(logits_parts, dim=0) if logits_parts else torch.empty(0, dtype=torch.float32)
        labels = torch.cat(labels_parts, dim=0) if labels_parts else torch.empty(0, dtype=torch.long)
        task_idxs = torch.cat(task_idx_parts, dim=0) if task_idx_parts else torch.empty(0, dtype=torch.long)
        if rank == 0:
            logger.info("Gathered evaluation outputs across ranks: %d total samples.", int(labels.numel()))
    else:
        logits = local_logits
        labels = local_labels
        task_idxs = local_task_idxs

    if labels.numel() == 0:
        raw_model.train()
        return (
            {
                "auc": float("nan"),
                "accuracy": float("nan"),
                "num_samples": 0.0,
                "num_pos": 0.0,
                "num_neg": 0.0,
                "pos_prob_mean": float("nan"),
                "neg_prob_mean": float("nan"),
                "macro_auc": float("nan"),
                "macro_accuracy": float("nan"),
            },
            {},
        )

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    overall = {
        "auc": _binary_roc_auc(probs, labels),
        "accuracy": (preds == labels).float().mean().item(),
        "num_samples": float(labels.numel()),
        "num_pos": float((labels == 1).sum().item()),
        "num_neg": float((labels == 0).sum().item()),
        "pos_prob_mean": probs[labels == 1].mean().item() if (labels == 1).any() else float("nan"),
        "neg_prob_mean": probs[labels == 0].mean().item() if (labels == 0).any() else float("nan"),
    }

    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    per_task: dict[str, dict[str, float]] = {}
    for t_idx in sorted(set(task_idxs.tolist())):
        mask = task_idxs == t_idx
        task_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
        task_probs = probs[mask]
        task_labels = labels[mask]
        task_preds = preds[mask]
        per_task[task_name] = {
            "auc": _binary_roc_auc(task_probs, task_labels),
            "accuracy": (task_preds == task_labels).float().mean().item(),
            "num_samples": float(task_labels.numel()),
            "num_pos": float((task_labels == 1).sum().item()),
            "num_neg": float((task_labels == 0).sum().item()),
            "pos_prob_mean": task_probs[task_labels == 1].mean().item() if (task_labels == 1).any() else float("nan"),
            "neg_prob_mean": task_probs[task_labels == 0].mean().item() if (task_labels == 0).any() else float("nan"),
        }

    if per_task:
        overall["macro_auc"] = float(np.mean([stats["auc"] for stats in per_task.values()]))
        overall["macro_accuracy"] = float(np.mean([stats["accuracy"] for stats in per_task.values()]))
    else:
        overall["macro_auc"] = float("nan")
        overall["macro_accuracy"] = float("nan")

    raw_model.train()
    return overall, per_task


def parse_args():
    p = argparse.ArgumentParser(
        description="Soft-token classifier on extracted task data using external event embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data_dir", default="extract_task_data/output")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--eval_split", default="val", choices=["train", "val", "test"])
    p.add_argument("--event_embedding_dir", required=True, help="Directory containing embeddings.npy, e.g. encode_events_result/bert")
    p.add_argument("--disease_model_name", default="michiyasunaga/BioLinkBERT-base")

    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=None)
    p.add_argument("--head_layers", type=int, default=1)
    p.add_argument("--max_positions", type=int, default=None)
    p.add_argument("--position_type", choices=["learned", "rotary"], default="learned")
    p.add_argument("--attention_type", choices=["bidirectional", "causal"], default="bidirectional")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--output_dir", default="output/disease-soft-token-classifier-extracted")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--max_events", type=int, default=1000, help="Keep only the latest M events; left-pad shorter sequences")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--aux_loss_weight", type=float, default=0.0)
    p.add_argument("--align_loss_weight", type=float, default=0.0)

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

    embeddings_path = str(Path(args.event_embedding_dir) / "embeddings.npy")
    store = EmbeddingStore(embeddings_path)
    event_dim = int(store.embeddings.shape[1])
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    task_text_embs = build_task_text_embs(args, device, rank, is_ddp)
    disease_dim = int(task_text_embs.shape[1])
    if rank == 0:
        logger.info(
            "Embedding dims: event_dim=%d from %s, disease_dim=%d from task text encoder",
            event_dim,
            embeddings_path,
            disease_dim,
        )

    if args.checkpoint:
        model = DiseaseEventSoftTokenClassifier.load_checkpoint(
            Path(args.checkpoint),
            task_text_embs=task_text_embs,
            device=device,
            dtype=dtype,
        )
    else:
        model = DiseaseEventSoftTokenClassifier(
            event_dim=event_dim,
            task_text_embs=task_text_embs,
            disease_dim=disease_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            head_layers=args.head_layers,
            max_positions=args.max_positions or (args.max_events + 1),
            position_type=args.position_type,
            attention_type=args.attention_type,
            dropout=args.dropout,
            dtype=dtype,
        ).to(device)

    if is_ddp and not args.eval_only:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, static_graph=True)

    if args.eval_only:
        overall, per_task = evaluate_classifier(
            model,
            eval_data_dir=args.data_dir,
            eval_split=args.eval_split,
            tasks=args.tasks,
            store=store,
            args=args,
            device=device,
        )
        if rank == 0:
            _log_eval("", overall, per_task)
        if is_ddp:
            dist.destroy_process_group()
        return

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )

    train_lengths = []
    for epoch_idx in range(args.epochs):
        ds = ExtractedTaskDataset(
            args.data_dir,
            "train",
            args.tasks,
            store,
            args.batch_size,
            epoch_idx,
            args.seed,
            world_size,
            rank,
            args.max_events,
        )
        train_lengths.append(len(ds))
    total_steps = sum(math.ceil(n / args.grad_accum) for n in train_lengths)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    pos_weight = torch.tensor(args.pos_weight, device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    global_step = 0
    best_auc = float("-inf")

    for epoch_idx in range(args.epochs):
        if rank == 0:
            logger.info("Epoch %d/%d ...", epoch_idx + 1, args.epochs)
        epoch_ds = ExtractedTaskDataset(
            args.data_dir,
            "train",
            args.tasks,
            store,
            args.batch_size,
            epoch_idx,
            args.seed,
            world_size,
            rank,
            args.max_events,
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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}/{args.epochs}", disable=(rank != 0), dynamic_ncols=True, total=len(epoch_ds))

        for batch_idx, batch in enumerate(pbar):
            labels = batch["labels"].to(device).float()
            logits, aux_logits, disease_hidden, event_pooled = model(
                batch["event_embs"].to(device),
                batch["event_mask"].to(device),
                batch["task_idxs"].to(device),
                return_aux_logits=True,
            )
            main_loss = loss_fn(logits, labels)
            aux_loss = loss_fn(aux_logits, labels)
            align_loss = F.mse_loss(
                F.normalize(event_pooled.float(), dim=-1),
                F.normalize(disease_hidden.float(), dim=-1),
            )
            loss = (main_loss + args.aux_loss_weight * aux_loss + args.align_loss_weight * align_loss) / args.grad_accum
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
                    lmain=f"{main_loss.item():.4f}",
                    laux=f"{aux_loss.item():.4f}",
                    lalign=f"{align_loss.item():.4f}",
                    posp=f"{probs[labels == 1].mean().item():.3f}" if (labels == 1).any() else "nan",
                    negp=f"{probs[labels == 0].mean().item():.3f}" if (labels == 0).any() else "nan",
                    gnorm=f"{float(grad_norm):.3f}" if not isinstance(grad_norm, float) or not math.isnan(grad_norm) else "nan",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if rank == 0:
            logger.info("Epoch %d/%d avg_loss=%.4f", epoch_idx + 1, args.epochs, epoch_loss / max(len(epoch_ds), 1))

        overall, per_task = evaluate_classifier(
            model,
            eval_data_dir=args.data_dir,
            eval_split=args.eval_split,
            tasks=args.tasks,
            store=store,
            args=args,
            device=device,
        )
        if rank == 0:
            _log_eval("  ", overall, per_task)
            raw_model = model.module if isinstance(model, DDP) else model
            epoch_dir = run_dir / f"epoch_{epoch_idx + 1}"
            raw_model.save_checkpoint(epoch_dir)
            if overall["auc"] > best_auc:
                best_auc = overall["auc"]
                raw_model.save_checkpoint(run_dir / "best")
            if use_wandb:
                import wandb
                log_dict = {
                    "epoch": epoch_idx + 1,
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
