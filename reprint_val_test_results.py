#!/usr/bin/env python3
"""Load all serialized val/test results and print metrics for each task."""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score


def compute_metrics(task_dir: Path, split: str) -> dict | None:
    data_path = task_dir / f"{split}.npz"
    thresh_path = task_dir / "threshold.npy"

    if not data_path.exists():
        print(f"[SKIP] {task_dir.name} ({split}): no {split}.npz")
        return None
    if not thresh_path.exists():
        print(f"[SKIP] {task_dir.name} ({split}): no threshold.npy")
        return None

    data = np.load(data_path)
    scores = data["scores"]
    labels = data["labels"]
    threshold = float(np.load(thresh_path))
    preds = (scores > threshold).astype(int)

    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(labels, scores)
    except ValueError:
        auprc = float("nan")

    print(f"\n{'='*60}")
    print(f"Task      : {task_dir.name}  [{split}]")
    print(f"Threshold : {threshold:+.4f}")
    print(f"F1 Score  : {f1:.4f}   AUROC: {auroc:.4f}   AUPRC: {auprc:.4f}")
    print(f"N samples : {len(labels)}  (pos={labels.sum()}, neg={(labels==0).sum()})")
    print(classification_report(labels, preds, target_names=["Negative", "Positive"], zero_division=0))

    pos_pct = labels.mean()
    return {"task": task_dir.name, "f1": f1, "auroc": auroc, "auprc": auprc, "threshold": threshold, "pos_pct": pos_pct}


def print_summary(rows: list[dict], title: str) -> None:
    print(f"\n{'='*72}")
    print(title)
    print(f"{'Task':<25} {'F1':>8} {'AUROC':>8} {'AUPRC':>8} {'Pos':>8} {'Threshold':>10}")
    print("-" * 72)
    for r in rows:
        print(f"{r['task']:<25} {r['f1']:>8.4f} {r['auroc']:>8.4f} {r['auprc']:>8.4f} {r['pos_pct']:>8.4f} {r['threshold']:>+10.4f}")
    print("-" * 72)
    avg_f1 = np.mean([r["f1"] for r in rows])
    avg_auroc = np.mean([r["auroc"] for r in rows])
    avg_auprc = np.mean([r["auprc"] for r in rows])
    avg_pos_pct = np.mean([r["pos_pct"] for r in rows])
    print(f"{'Mean':<25} {avg_f1:>8.4f} {avg_auroc:>8.4f} {avg_auprc:>8.4f} {avg_pos_pct:>8.4f}")


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    task_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if not task_dirs:
        print(f"No task subdirectories found in {results_dir}")
        sys.exit(1)

    print(f"Loading results from: {results_dir.resolve()}")
    print(f"Tasks found: {[d.name for d in task_dirs]}")

    val_rows, test_rows = [], []
    for task_dir in task_dirs:
        val_result = compute_metrics(task_dir, "val")
        if val_result:
            val_rows.append(val_result)
        test_result = compute_metrics(task_dir, "test")
        if test_result:
            test_rows.append(test_result)

    if val_rows:
        print_summary(val_rows, "SUMMARY (val set)")
    if test_rows:
        print_summary(test_rows, "SUMMARY (test set)")


if __name__ == "__main__":
    main()
