#!/usr/bin/env python3

import argparse
import copy
import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


TASKS = [
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
]


class LinearBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark multiple foundation event-embedding versions with one simple classifier per task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--embedding_dirs",
        nargs="+",
        required=True,
        help="Embedding directories under data/01_outputs/, each containing embeddings.npy.",
    )
    p.add_argument("--eval_data_dir", default="data/llm_eval_data_ixc")
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    p.add_argument("--train_split", default="val", choices=["val", "test"])
    p.add_argument("--test_split", default="test", choices=["val", "test"])
    p.add_argument("--max_events", type=int, default=1000)
    p.add_argument("--truncate_side", default="first", choices=["first", "last"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hidden_dim", type=int, default=0,
                   help="Keep 0 for pure logistic regression. Set >0 to use a tiny 2-layer MLP.")
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="output/foundation-simple-classifier-benchmark")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return int(float(text) != 0.0)


def _binary_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.float().cpu()
    labels = labels.long().cpu()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    auc = (cmp > 0).float().mean() + 0.5 * (cmp == 0).float().mean()
    return float(auc.item())


def _binary_average_precision(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = probs.float().cpu()
    labels = labels.long().cpu()
    n_pos = int((labels == 1).sum().item())
    if probs.numel() == 0 or n_pos == 0:
        return float("nan")
    order = torch.argsort(probs, descending=True)
    sorted_labels = labels[order]
    tp_cum = (sorted_labels == 1).cumsum(0).float()
    precision_at_k = tp_cum / torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
    ap = (precision_at_k * (sorted_labels == 1).float()).sum() / float(n_pos)
    return float(ap.item())


def _binary_confusion_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = preds.long().cpu()
    labels = labels.long().cpu()
    tp = float(((preds == 1) & (labels == 1)).sum().item())
    tn = float(((preds == 0) & (labels == 0)).sum().item())
    fp = float(((preds == 1) & (labels == 0)).sum().item())
    fn = float(((preds == 0) & (labels == 1)).sum().item())
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    precision = tp / max(tp + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    labels = labels.long().cpu()
    probs = torch.sigmoid(logits.float().cpu())
    preds = (probs >= 0.5).long()
    out = {
        "auc": _binary_roc_auc(probs, labels),
        "auprc": _binary_average_precision(probs, labels),
        "num_samples": float(labels.numel()),
        "num_pos": float((labels == 1).sum().item()),
        "num_neg": float((labels == 0).sum().item()),
        "pos_prob_mean": float(probs[labels == 1].mean().item()) if (labels == 1).any() else float("nan"),
        "neg_prob_mean": float(probs[labels == 0].mean().item()) if (labels == 0).any() else float("nan"),
    }
    out.update(_binary_confusion_metrics(preds, labels))
    return out


def make_model(input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if hidden_dim <= 0:
        return LinearBinaryClassifier(input_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    return logits


def read_eval_rows(eval_data_dir: str, task: str, split: str) -> List[Tuple[np.ndarray, int]]:
    path = Path(eval_data_dir) / task / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval parquet: {path}")
    df = pd.read_parquet(path, columns=["event_ids", "label"])
    rows = []
    for row in df.itertuples(index=False):
        rows.append((np.array(row.event_ids, dtype=np.int32), parse_binary_label(row.label)))
    return rows


def mean_pool_rows(
    rows: List[Tuple[np.ndarray, int]],
    embeddings: np.ndarray,
    max_events: Optional[int],
    truncate_side: str,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = int(embeddings.shape[1])
    feats = np.zeros((len(rows), dim), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.float32)
    for i, (event_ids, label) in enumerate(rows):
        if max_events is not None and len(event_ids) > max_events:
            if truncate_side == "last":
                event_ids = event_ids[-max_events:]
            else:
                event_ids = event_ids[:max_events]
        if len(event_ids) > 0:
            feats[i] = embeddings[event_ids].mean(axis=0, dtype=np.float32)
        labels[i] = float(label)
    return feats, labels


def standardize(
    train_x: np.ndarray,
    eval_arrays: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_x = (train_x - mean) / std
    out = [(arr - mean) / std for arr in eval_arrays]
    return train_x, out, mean, std


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logits_parts = []
    label_parts = []
    model.eval()
    for xb, yb in dl:
        xb = xb.to(device)
        logits_parts.append(forward_logits(model, xb).cpu())
        label_parts.append(yb.cpu())
    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    return compute_metrics(logits, labels)


def train_single_task(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    args,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model = make_model(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    pos_weight = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_train_auc = float("-inf")
    best_eval_auc = float("-inf")
    best_train_metrics = None  # type: Optional[Dict[str, float]]
    best_eval_metrics = None  # type: Optional[Dict[str, float]]
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(model, xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            batch_n = int(yb.numel())
            running_loss += float(loss.item()) * batch_n
            seen += batch_n

        train_metrics = evaluate_model(model, val_x, val_y, args.eval_batch_size, device)
        logger.info(
            "      epoch=%d train_loss=%.4f train_auc=%.4f train_auprc=%.4f train_bal_acc=%.4f",
            epoch,
            running_loss / max(seen, 1),
            train_metrics["auc"],
            train_metrics["auprc"],
            train_metrics["balanced_accuracy"],
        )

        if test_x is not None and test_y is not None:
            eval_metrics = evaluate_model(model, test_x, test_y, args.eval_batch_size, device)
            logger.info(
                "      epoch=%d test_auc=%.4f test_auprc=%.4f test_bal_acc=%.4f",
                epoch,
                eval_metrics["auc"],
                eval_metrics["auprc"],
                eval_metrics["balanced_accuracy"],
            )
            eval_auc = eval_metrics["auc"]
            if math.isnan(eval_auc):
                eval_auc = float("-inf")
        else:
            eval_metrics = train_metrics
            eval_auc = train_metrics["auc"]
            if math.isnan(eval_auc):
                eval_auc = float("-inf")

        train_auc = train_metrics["auc"]
        if math.isnan(train_auc):
            train_auc = float("-inf")
        if eval_auc > best_eval_auc:
            best_eval_auc = eval_auc
            best_epoch = epoch
            best_train_auc = train_auc
            best_train_metrics = train_metrics
            best_eval_metrics = eval_metrics
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break

    model.load_state_dict(best_state)
    final_train_metrics = best_train_metrics or evaluate_model(model, val_x, val_y, args.eval_batch_size, device)
    result = {
        "train": final_train_metrics,
        "best_epoch": {"epoch": float(best_epoch)},
    }
    if test_x is not None and test_y is not None:
        result["test"] = best_eval_metrics or evaluate_model(model, test_x, test_y, args.eval_batch_size, device)
    return result


def aggregate_task_metrics(task_results: Dict[str, Dict[str, Dict[str, float]]], split: str) -> Dict[str, float]:
    metric_names = [
        "auc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
    ]
    summary = {}  # type: Dict[str, float]
    for metric in metric_names:
        values = [task_results[task][split][metric] for task in task_results if split in task_results[task]]
        clean = [v for v in values if not math.isnan(v)]
        summary[f"macro_{metric}"] = float(sum(clean) / len(clean)) if clean else float("nan")
    return summary


def prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)
    logger.info("Device: %s", device)
    logger.info("Tasks: %s", ", ".join(args.tasks))
    logger.info("Train split: %s | Test split: %s", args.train_split, args.test_split)
    logger.info("Embedding dirs: %s", ", ".join(args.embedding_dirs))

    all_results = {}  # type: Dict[str, Dict[str, object]]
    summary_rows = []  # type: List[Dict[str, object]]

    for embedding_dir_name in args.embedding_dirs:
        embedding_dir = Path("data/01_outputs") / embedding_dir_name
        embeddings_path = embedding_dir / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

        logger.info("=" * 100)
        logger.info("Benchmarking embedding version: %s", embedding_dir_name)
        logger.info("=" * 100)
        embeddings = np.load(embeddings_path, mmap_mode="r")
        task_results = {}  # type: Dict[str, Dict[str, Dict[str, float]]]

        for task in args.tasks:
            logger.info("  Task: %s", task)
            train_rows = read_eval_rows(args.eval_data_dir, task, args.train_split)
            test_rows = read_eval_rows(args.eval_data_dir, task, args.test_split)

            train_x, train_y = mean_pool_rows(train_rows, embeddings, args.max_events, args.truncate_side)
            test_x, test_y = mean_pool_rows(test_rows, embeddings, args.max_events, args.truncate_side)
            train_x, [test_x], _, _ = standardize(train_x, [test_x])

            logger.info(
                "    pooled features: train=%s test=%s",
                tuple(train_x.shape),
                tuple(test_x.shape),
            )
            train_pos = int(train_y.sum())
            train_neg = int(len(train_y) - train_pos)
            test_pos = int(test_y.sum())
            test_neg = int(len(test_y) - test_pos)
            pos_weight = float(train_neg / max(train_pos, 1))
            logger.info(
                "    labels: train_pos=%d/%d train_neg=%d/%d test_pos=%d/%d test_neg=%d/%d pos_weight=%.4f",
                train_pos,
                len(train_y),
                train_neg,
                len(train_y),
                test_pos,
                len(test_y),
                test_neg,
                len(test_y),
                pos_weight,
            )

            task_result = train_single_task(
                train_x=train_x,
                train_y=train_y,
                val_x=train_x,
                val_y=train_y,
                test_x=test_x,
                test_y=test_y,
                args=args,
                device=device,
            )
            task_results[task] = task_result
            logger.info(
                "    best train: auc=%.4f auprc=%.4f bal_acc=%.4f f1=%.4f epoch=%d",
                task_result["train"]["auc"],
                task_result["train"]["auprc"],
                task_result["train"]["balanced_accuracy"],
                task_result["train"]["f1"],
                int(task_result["best_epoch"]["epoch"]),
            )
            logger.info(
                "    test: auc=%.4f auprc=%.4f bal_acc=%.4f f1=%.4f",
                task_result["test"]["auc"],
                task_result["test"]["auprc"],
                task_result["test"]["balanced_accuracy"],
                task_result["test"]["f1"],
            )

        aggregate = {
            "train": aggregate_task_metrics(task_results, "train"),
            "test": aggregate_task_metrics(task_results, "test"),
        }

        all_results[embedding_dir_name] = {
            "aggregate": aggregate,
            "tasks": task_results,
        }

        row = {
            "embedding_dir": embedding_dir_name,
            **prefix_metrics(aggregate["train"], "train"),
            **prefix_metrics(aggregate["test"], "test"),
        }
        summary_rows.append(row)

        logger.info(
            "Summary for %s: macro_train_auc=%.4f macro_train_auprc=%.4f macro_train_bal_acc=%.4f",
            embedding_dir_name,
            aggregate["train"]["macro_auc"],
            aggregate["train"]["macro_auprc"],
            aggregate["train"]["macro_balanced_accuracy"],
        )
        logger.info(
            "                     macro_test_auc=%.4f macro_test_auprc=%.4f macro_test_bal_acc=%.4f",
            aggregate["test"]["macro_auc"],
            aggregate["test"]["macro_auprc"],
            aggregate["test"]["macro_balanced_accuracy"],
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("test_macro_auc", ascending=False).reset_index(drop=True)

    summary_path = out_dir / "summary.csv"
    results_path = out_dir / "results.json"
    config_path = out_dir / "config.json"

    summary_df.to_csv(summary_path, index=False)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Saved summary -> %s", summary_path)
    logger.info("Saved detailed results -> %s", results_path)
    if not summary_df.empty:
        logger.info("Final ranking by macro test AUC:")
        for row in summary_df.itertuples(index=False):
            logger.info(
                "  %s: macro_test_auc=%.4f macro_test_auprc=%.4f macro_test_bal_acc=%.4f",
                row.embedding_dir,
                row.test_macro_auc,
                row.test_macro_auprc,
                row.test_macro_balanced_accuracy,
            )


if __name__ == "__main__":
    main()
