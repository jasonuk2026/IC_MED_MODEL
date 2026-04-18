#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_embedding_disease_cond_v2 import (
    TASK_2_DISEASE_NAME,
    EmbeddingStore,
    _binary_roc_auc,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
    emb_list: list[np.ndarray],
    labels: list[int] | None,
    max_events: int,
) -> dict[str, torch.Tensor]:
    if not emb_list:
        raise ValueError("Cannot collate an empty batch.")
    event_dim = emb_list[0].shape[1] if emb_list[0].ndim == 2 else 0
    batch_size = len(emb_list)
    padded = np.zeros((batch_size, max_events, event_dim), dtype=np.float32)
    mask = np.zeros((batch_size, max_events), dtype=np.int64)

    for i, embs in enumerate(emb_list):
        if embs.shape[0] > max_events:
            embs = embs[-max_events:]
        if embs.shape[0] == 0:
            continue
        n = embs.shape[0]
        padded[i, max_events - n :] = embs
        mask[i, max_events - n :] = 1

    out = {
        "event_embs": torch.from_numpy(padded),
        "event_mask": torch.from_numpy(mask),
    }
    if labels is not None:
        out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out


def _load_task_rows(*, data_dir: str, task: str, split: str) -> list[tuple[np.ndarray, int]]:
    path = Path(data_dir) / task / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing extracted task parquet: {path}")

    df = pd.read_parquet(path, columns=["label", "event_ids"])
    rows: list[tuple[np.ndarray, int]] = []
    for row in df.itertuples(index=False):
        rows.append((
            np.array(row.event_ids, dtype=np.int32),
            parse_binary_label(row.label),
        ))
    return rows


class EventBatchCollator:
    def __init__(self, max_events: int):
        self.max_events = max_events

    def __call__(self, batch: list[tuple[np.ndarray, int]]) -> dict[str, torch.Tensor]:
        emb_list = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return collate_latest_leftpad(
            emb_list=emb_list,
            labels=labels,
            max_events=self.max_events,
        )


class SingleTaskExtractedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        task: str,
        split: str,
        embeddings: np.ndarray,
        max_events: int,
        training_epoch: int,
        seed: int,
    ):
        self.embeddings = embeddings
        self.max_events = max_events
        rows = _load_task_rows(data_dir=data_dir, task=task, split=split)
        rng = random.Random(seed + training_epoch * 1337)
        rng.shuffle(rows)
        self.samples = rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        eids, label = self.samples[idx]
        if len(eids) > self.max_events:
            eids = eids[-self.max_events:]
        if len(eids) == 0:
            event_dim = int(self.embeddings.shape[1])
            return np.zeros((0, event_dim), dtype=np.float32), label
        embs = self.embeddings[eids]  # already float32, no copy needed
        return embs, label


class EventMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        hidden_size: int = 768,
        head_layers: int = 1,
        intermediate_size: int | None = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.event_dim = event_dim
        self.hidden_size = hidden_size
        self.head_layers = head_layers
        self.intermediate_size = intermediate_size or (hidden_size * 4)
        self.dropout_p = dropout
        self.dtype = dtype

        self.event_norm = nn.RMSNorm(event_dim).to(dtype)
        self.event_proj = nn.Linear(event_dim, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.event_proj.weight)

        layers: list[nn.Module] = []
        in_dim = hidden_size
        for _ in range(head_layers):
            layers.extend([
                nn.RMSNorm(in_dim).to(dtype),
                nn.Linear(in_dim, self.intermediate_size, bias=False).to(dtype),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.intermediate_size, hidden_size, bias=False).to(dtype),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_size
        self.head = nn.Sequential(*layers)
        self.out_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.out_proj = nn.Linear(hidden_size, 1, bias=False).to(dtype)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, event_embs: torch.Tensor, event_mask: torch.Tensor, return_features: bool = False):
        event_tokens = self.event_proj(self.event_norm(event_embs.to(self.dtype)))
        mask = event_mask.to(event_tokens.dtype).unsqueeze(-1)
        pooled = (event_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        hidden = pooled
        if len(self.head) > 0:
            hidden = self.head(hidden)
        logits = self.out_proj(self.out_norm(hidden)).squeeze(-1).float()
        if return_features:
            return logits, pooled.float(), hidden.float()
        return logits

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "event_dim": self.event_dim,
                    "hidden_size": self.hidden_size,
                    "head_layers": self.head_layers,
                    "intermediate_size": self.intermediate_size,
                    "dropout": self.dropout_p,
                    "dtype": str(self.dtype),
                },
            },
            save_dir / "model.pt",
        )
        logger.info("  Saved checkpoint -> %s", save_dir)

    @classmethod
    def load_checkpoint(cls, save_dir: Path, *, device: torch.device, dtype: torch.dtype) -> "EventMLPClassifier":
        payload = torch.load(save_dir / "model.pt", map_location="cpu")
        cfg = payload["config"]
        model = cls(
            event_dim=cfg["event_dim"],
            hidden_size=cfg["hidden_size"],
            head_layers=cfg["head_layers"],
            intermediate_size=cfg["intermediate_size"],
            dropout=cfg["dropout"],
            dtype=dtype,
        )
        model.load_state_dict(payload["state_dict"])
        return model.to(device)


@torch.inference_mode()
def evaluate_single_task(
    model: torch.nn.Module,
    *,
    eval_data_dir: str,
    task: str,
    eval_split: str,
    store: EmbeddingStore,
    args,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
    logger.info("Preparing %s split for task=%s ...", eval_split, task)

    eval_ds = SingleTaskExtractedDataset(
        eval_data_dir,
        task,
        eval_split,
        store.embeddings,
        args.inferred_max_events,
        training_epoch=0,
        seed=0,
    )
    logger.info("Evaluation data prepared: %d samples.", len(eval_ds))
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=EventBatchCollator(args.inferred_max_events),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=args.num_workers == 0 and torch.cuda.is_available(),
        persistent_workers=False,
    )
    all_logits = []
    all_labels = []
    for batch in tqdm(
        eval_loader,
        desc=f"Evaluating {task}/{eval_split}",
        dynamic_ncols=True,
    ):
        logits = model(
            batch["event_embs"].to(device),
            batch["event_mask"].to(device),
        )
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())

    if all_logits:
        local_logits = torch.cat(all_logits, dim=0)
        local_labels = torch.cat(all_labels, dim=0)
    else:
        local_logits = torch.empty(0, dtype=torch.float32)
        local_labels = torch.empty(0, dtype=torch.long)

    logits = local_logits
    labels = local_labels

    if labels.numel() == 0:
        model.train()
        empty = {
            "auc": float("nan"),
            "auprc": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "num_samples": 0.0,
            "num_pos": 0.0,
            "num_neg": 0.0,
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "pos_prob_mean": float("nan"),
            "neg_prob_mean": float("nan"),
            "macro_auc": float("nan"),
            "macro_accuracy": float("nan"),
        }
        return empty, {task: empty.copy()}

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    confusion = _binary_confusion_metrics(preds, labels)
    metrics = {
        "auc": _binary_roc_auc(probs, labels),
        "auprc": _binary_average_precision(probs, labels),
        "accuracy": (preds == labels).float().mean().item(),
        "num_samples": float(labels.numel()),
        "num_pos": float((labels == 1).sum().item()),
        "num_neg": float((labels == 0).sum().item()),
        "pos_prob_mean": probs[labels == 1].mean().item() if (labels == 1).any() else float("nan"),
        "neg_prob_mean": probs[labels == 0].mean().item() if (labels == 0).any() else float("nan"),
    }
    metrics.update(confusion)
    metrics["macro_auc"] = metrics["auc"]
    metrics["macro_accuracy"] = metrics["accuracy"]

    model.train()
    return metrics, {task: metrics.copy()}


def infer_max_events_from_train_parquet(train_data_dir: str, task: str) -> int:
    train_path = Path(train_data_dir) / task / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training parquet for inferring max events: {train_path}")
    df = pd.read_parquet(train_path, columns=["event_ids"])
    if len(df) == 0:
        raise ValueError(f"Training parquet is empty: {train_path}")
    max_events = int(df["event_ids"].map(len).max())
    if max_events <= 0:
        raise ValueError(f"Inferred non-positive max_events={max_events} from {train_path}")
    return max_events


def get_split_label_stats(data_dir: str, task: str, split: str) -> dict[str, int]:
    path = Path(data_dir) / task / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet for stats: {path}")
    df = pd.read_parquet(path, columns=["label"])
    labels = [parse_binary_label(v) for v in df["label"].tolist()]
    num_samples = len(labels)
    num_pos = int(sum(labels))
    num_neg = int(num_samples - num_pos)
    return {
        "num_samples": num_samples,
        "num_pos": num_pos,
        "num_neg": num_neg,
    }


def _binary_average_precision(probs: torch.Tensor, labels: torch.Tensor) -> float:
    if probs.numel() == 0:
        return float("nan")
    labels = labels.long()
    n_pos = int((labels == 1).sum().item())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(probs, descending=True)
    sorted_labels = labels[order]
    tp_cum = (sorted_labels == 1).cumsum(0).float()
    precision_at_k = tp_cum / torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
    ap = (precision_at_k * (sorted_labels == 1).float()).sum() / float(n_pos)
    return float(ap.item())


def _binary_confusion_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = preds.long()
    labels = labels.long()
    tp = float(((preds == 1) & (labels == 1)).sum().item())
    tn = float(((preds == 0) & (labels == 0)).sum().item())
    fp = float(((preds == 1) & (labels == 0)).sum().item())
    fn = float(((preds == 0) & (labels == 1)).sum().item())
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    precision = tp / max(tp + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _log_eval(prefix: str, overall: dict[str, float], per_task: dict[str, dict[str, float]]):
    logger.info("%s val auc: %.4f", prefix, overall["auc"])
    logger.info("%s val auprc: %.4f", prefix, overall["auprc"])
    logger.info("%s val balanced_accuracy: %.4f", prefix, overall["balanced_accuracy"])
    logger.info("%s val f1: %.4f", prefix, overall["f1"])
    logger.info("%s val precision: %.4f", prefix, overall["precision"])
    logger.info("%s val recall: %.4f", prefix, overall["recall"])
    logger.info("%s val specificity: %.4f", prefix, overall["specificity"])
    logger.info("%s val accuracy: %.4f", prefix, overall["accuracy"])
    logger.info(
        "%s counts: n=%d  pos=%d  neg=%d  tp=%d  tn=%d  fp=%d  fn=%d",
        prefix,
        int(overall["num_samples"]),
        int(overall["num_pos"]),
        int(overall["num_neg"]),
        int(overall["tp"]),
        int(overall["tn"]),
        int(overall["fp"]),
        int(overall["fn"]),
    )
    logger.info(
        "%s mean prob: pos=%.4f  neg=%.4f",
        prefix,
        overall["pos_prob_mean"],
        overall["neg_prob_mean"],
    )
    for task, stats in per_task.items():
        logger.info(
            "    %s: auc=%.4f  auprc=%.4f  bal_acc=%.4f  f1=%.4f  prec=%.4f  rec=%.4f",
            task,
            stats["auc"],
            stats["auprc"],
            stats["balanced_accuracy"],
            stats["f1"],
            stats["precision"],
            stats["recall"],
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-task event-only mean-pooling MLP classifier on extracted task data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--train_data_dir", default="extract_task_data_oversampled/output")
    p.add_argument("--eval_data_dir", default="extract_task_data/output")
    p.add_argument("--task", required=True, choices=sorted(TASK_2_DISEASE_NAME.keys()))
    p.add_argument("--eval_split", default="val", choices=["train", "val", "test"])
    p.add_argument("--event_embedding_dir", required=True, help="Directory containing embeddings.npy, e.g. encode_events_result/bert")

    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--head_layers", type=int, default=1)
    p.add_argument("--intermediate_size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--output_dir", default="output/disease-soft-token-classifier-extracted-single")
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
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / args.task / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)
    logger.info("Single-task event-only mode: task=%s", args.task)
    args.inferred_max_events = infer_max_events_from_train_parquet(args.train_data_dir, args.task)
    logger.info(
        "Auto-inferred eval/truncation window from %s/%s/train.parquet: latest %d events",
        args.train_data_dir,
        args.task,
        args.inferred_max_events,
    )
    train_stats = get_split_label_stats(args.train_data_dir, args.task, "train")
    eval_stats = get_split_label_stats(args.eval_data_dir, args.task, args.eval_split)
    logger.info(
        "Train stats: n=%d pos=%d neg=%d pos/neg=%.4f",
        train_stats["num_samples"],
        train_stats["num_pos"],
        train_stats["num_neg"],
        train_stats["num_pos"] / max(train_stats["num_neg"], 1),
    )
    logger.info(
        "Eval stats (%s): n=%d pos=%d neg=%d pos/neg=%.4f",
        args.eval_split,
        eval_stats["num_samples"],
        eval_stats["num_pos"],
        eval_stats["num_neg"],
        eval_stats["num_pos"] / max(eval_stats["num_neg"], 1),
    )

    use_wandb = args.wandb_project is not None
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.task}-{timestamp}",
            tags=args.wandb_tags,
            config=vars(args),
            dir=str(run_dir),
            settings=wandb.Settings(console="wrap"),
        )

    embeddings_path = str(Path(args.event_embedding_dir) / "embeddings.npy")
    store = EmbeddingStore(embeddings_path)
    # Always materialise as float32 in RAM:
    #   - mmap fd is unsafe to share across forked workers after CUDA init
    #   - pre-converting dtype eliminates the per-sample .astype() copy in __getitem__
    store.embeddings = np.array(store.embeddings, dtype=np.float32)
    logger.info("Embeddings loaded into RAM as float32 %s.", store.embeddings.shape)
    event_dim = int(store.embeddings.shape[1])
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    logger.info("Embedding dim: event_dim=%d from %s", event_dim, embeddings_path)

    if args.checkpoint:
        model = EventMLPClassifier.load_checkpoint(
            Path(args.checkpoint),
            device=device,
            dtype=dtype,
        )
    else:
        model = EventMLPClassifier(
            event_dim=event_dim,
            hidden_size=args.hidden_size,
            head_layers=args.head_layers,
            intermediate_size=args.intermediate_size,
            dropout=args.dropout,
            dtype=dtype,
        ).to(device)

    logger.info(
        "Event-only MLP classifier: hidden=%d head_layers=%d intermediate=%s dropout=%.3f dtype=%s",
        args.hidden_size,
        args.head_layers,
        args.intermediate_size or (args.hidden_size * 4),
        args.dropout,
        dtype,
    )

    if args.eval_only:
        overall, per_task = evaluate_single_task(
            model,
            eval_data_dir=args.eval_data_dir,
            task=args.task,
            eval_split=args.eval_split,
            store=store,
            args=args,
            device=device,
        )
        _log_eval("", overall, per_task)
        return

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )

    # Build dataset and DataLoader once; reuse across all epochs.
    # Recreating the DataLoader per epoch kills and respawns workers at every epoch
    # boundary, which is the source of the "periodic freeze" between epochs.
    train_ds = SingleTaskExtractedDataset(
        args.train_data_dir,
        args.task,
        "train",
        store.embeddings,
        args.inferred_max_events,
        training_epoch=0,
        seed=args.seed,
    )
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = math.ceil(args.epochs * steps_per_epoch / args.grad_accum)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    pos_weight = torch.tensor(args.pos_weight, device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=EventBatchCollator(args.inferred_max_events),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=args.num_workers == 0 and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    global_step = 0
    best_auc = float("-inf")

    for epoch_idx in range(args.epochs):
        logger.info("Epoch %d/%d ...", epoch_idx + 1, args.epochs)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch_idx + 1}/{args.epochs}",
            dynamic_ncols=True,
            total=steps_per_epoch,
        )

        for batch_idx, batch in enumerate(pbar):
            labels = batch["labels"].to(device).float()
            logits, pooled, hidden = model(
                batch["event_embs"].to(device),
                batch["event_mask"].to(device),
                return_features=True,
            )
            loss = loss_fn(logits, labels) / args.grad_accum
            loss.backward()

            is_update_step = ((batch_idx + 1) % args.grad_accum == 0) or ((batch_idx + 1) == len(train_loader))
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
            probs = torch.sigmoid(logits.float())
            pbar.set_postfix(
                loss=f"{loss.item() * args.grad_accum:.4f}",
                posp=f"{probs[labels == 1].mean().item():.3f}" if (labels == 1).any() else "nan",
                negp=f"{probs[labels == 0].mean().item():.3f}" if (labels == 0).any() else "nan",
                pooled_norm=f"{pooled.norm(dim=-1).mean().item():.3f}",
                hidden_norm=f"{hidden.norm(dim=-1).mean().item():.3f}",
                gnorm=f"{float(grad_norm):.3f}" if not isinstance(grad_norm, float) or not math.isnan(grad_norm) else "nan",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        logger.info("Epoch %d/%d avg_loss=%.4f", epoch_idx + 1, args.epochs, epoch_loss / max(len(train_loader), 1))

        overall, per_task = evaluate_single_task(
            model,
            eval_data_dir=args.eval_data_dir,
            task=args.task,
            eval_split=args.eval_split,
            store=store,
            args=args,
            device=device,
        )
        _log_eval("  ", overall, per_task)
        epoch_dir = run_dir / f"epoch_{epoch_idx + 1}"
        model.save_checkpoint(epoch_dir)
        if overall["auc"] > best_auc:
            best_auc = overall["auc"]
            model.save_checkpoint(run_dir / "best")
        if use_wandb:
            import wandb
            wandb.log(
                {
                    "epoch": epoch_idx + 1,
                    "train/avg_loss": epoch_loss / max(len(train_loader), 1),
                    "eval/auc": overall["auc"],
                    "eval/auprc": overall["auprc"],
                    "eval/accuracy": overall["accuracy"],
                    "eval/balanced_accuracy": overall["balanced_accuracy"],
                    "eval/f1": overall["f1"],
                    "eval/precision": overall["precision"],
                    "eval/recall": overall["recall"],
                    "eval/specificity": overall["specificity"],
                    "eval/pos_prob_mean": overall["pos_prob_mean"],
                    "eval/neg_prob_mean": overall["neg_prob_mean"],
                }
            )

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    main()
