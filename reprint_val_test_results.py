#!/usr/bin/env python3
"""Load all serialized test results and print metrics for each task."""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score


def print_task_results(task_dir: Path) -> dict | None:
    test_path = task_dir / "test.npz"
    thresh_path = task_dir / "threshold.npy"

    if not test_path.exists():
        print(f"[SKIP] {task_dir.name}: no test.npz")
        return None
    if not thresh_path.exists():
        print(f"[SKIP] {task_dir.name}: no threshold.npy")
        return None

    data = np.load(test_path)
    scores = data["scores"]
    labels = data["labels"]
    threshold = float(np.load(thresh_path))
    preds = (scores > threshold).astype(int)

    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float("nan")

    print(f"\n{'='*60}")
    print(f"Task      : {task_dir.name}")
    print(f"Threshold : {threshold:+.4f}")
    print(f"F1 Score  : {f1:.4f}   AUROC: {auroc:.4f}")
    print(f"N samples : {len(labels)}  (pos={labels.sum()}, neg={(labels==0).sum()})")
    print(classification_report(labels, preds, target_names=["Negative", "Positive"], zero_division=0))

    return {"task": task_dir.name, "f1": f1, "auroc": auroc, "threshold": threshold}


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

    rows = []
    for task_dir in task_dirs:
        result = print_task_results(task_dir)
        if result:
            rows.append(result)

    if not rows:
        return

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY (test set)")
    print(f"{'Task':<25} {'F1':>8} {'AUROC':>8} {'Threshold':>10}")
    print("-" * 55)
    for r in rows:
        print(f"{r['task']:<25} {r['f1']:>8.4f} {r['auroc']:>8.4f} {r['threshold']:>+10.4f}")
    print("-" * 55)
    avg_f1 = np.mean([r["f1"] for r in rows])
    avg_auroc = np.mean([r["auroc"] for r in rows])
    print(f"{'Mean':<25} {avg_f1:>8.4f} {avg_auroc:>8.4f}")


if __name__ == "__main__":
    main()
