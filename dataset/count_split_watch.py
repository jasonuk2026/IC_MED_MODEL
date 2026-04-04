"""
Interactive stats watcher: load EHR data once, re-evaluate determine_num_sample.py on demand.

Workflow:
  1. Run this script (EHR data loads once).
  2. Edit determine_num_sample.py freely.
  3. Press Enter to re-run stats with the latest policy — no reload of EHR data.
  4. Ctrl-C or type 'q' to quit.

Usage:
    python count_split_watch.py --task new_pancan --max_events 500
    python count_split_watch.py --all_tasks --max_events 500
"""

import argparse
import importlib
import os
import sys

import pandas as pd

import determine_num_sample  # will be reloaded on each iteration

EHRSHOT_ASSETS = "EHRSHOT_ASSETS"

VALID_TASKS = [
    "new_hypertension", "new_hyperlipidemia", "new_pancan",
    "new_celiac", "new_lupus", "new_acutemi",
]

SPLITS = ["train", "val", "test"]


def _is_positive(label_val) -> bool:
    try:
        return int(float(label_val)) != 0
    except (ValueError, TypeError):
        return False


def compute_stats(task: str, grouped: dict, df_labels_map: dict, max_events) -> None:
    df_labels = df_labels_map[task]

    counts = {s: {"pos": 0, "neg": 0} for s in SPLITS}
    raw_counts = {s: {"pos": 0, "neg": 0} for s in SPLITS}

    for row in df_labels.itertuples(index=False):
        split = row.split
        if split not in counts:
            continue
        is_pos = _is_positive(row.value)
        key = "pos" if is_pos else "neg"
        raw_counts[split][key] += 1

        patient_df = grouped.get(row.patient_id)
        M = int((patient_df["start"] < row.prediction_time).sum()) if patient_df is not None else 0

        if max_events is None or M <= max_events:
            n = 1
        else:
            n = determine_num_sample.get_sample_n_times(
                is_positive=is_pos,
                N_EVENTS_LIMIT=max_events,
                ACTUAL_NUM_EVENTS=M,
                task=task,
            )
        counts[split][key] += n

    print(f"\nTask: {task}  (max_events={max_events})")
    print(f"{'split':<8} {'raw_pos':>8} {'raw_neg':>8} {'cop_pos':>10} {'cop_neg':>10} {'ratio':>8}")
    print("-" * 58)
    for s in SPLITS:
        rp = raw_counts[s]["pos"]
        rn = raw_counts[s]["neg"]
        cp = counts[s]["pos"]
        cn = counts[s]["neg"]
        ratio = f"{cp/cn:.3f}" if cn > 0 else "N/A"
        print(f"{s:<8} {rp:>8,} {rn:>8,} {cp:>10,} {cn:>10,} {ratio:>8}")
    total_cp = sum(counts[s]["pos"] for s in SPLITS)
    total_cn = sum(counts[s]["neg"] for s in SPLITS)
    total_ratio = f"{total_cp/total_cn:.3f}" if total_cn > 0 else "N/A"
    print("-" * 58)
    print(f"{'total':<8} {sum(raw_counts[s]['pos'] for s in SPLITS):>8,} "
          f"{sum(raw_counts[s]['neg'] for s in SPLITS):>8,} "
          f"{total_cp:>10,} {total_cn:>10,} {total_ratio:>8}")


def run_all(tasks, grouped, df_labels_map, max_events):
    importlib.reload(determine_num_sample)
    for task in tasks:
        compute_stats(task, grouped, df_labels_map, max_events)
    print()


def main():
    parser = argparse.ArgumentParser(description="Interactive stats watcher (EHR loaded once)")
    parser.add_argument("--task", type=str, choices=VALID_TASKS, default=None)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=EHRSHOT_ASSETS)
    args = parser.parse_args()

    if not args.task and not args.all_tasks:
        parser.error("Provide --task <name> or --all_tasks")

    tasks = VALID_TASKS if args.all_tasks else [args.task]

    # ── Load EHR data once ────────────────────────────────────────────────────
    print("Loading raw EHR data (once)...")
    ehr_path = os.path.join(args.data_dir, "data", "ehrshot.csv")
    df_ehr = pd.read_csv(ehr_path, low_memory=False, usecols=["patient_id", "start"])
    df_ehr["start"] = pd.to_datetime(df_ehr["start"])
    grouped = dict(list(df_ehr.groupby("patient_id")))
    print(f"  {len(df_ehr):,} events across {len(grouped):,} patients")

    # ── Load labels + splits for each task once ───────────────────────────────
    splits_path = os.path.join(args.data_dir, "splits", "person_id_map.csv")
    df_splits = pd.read_csv(splits_path)
    pid_to_split = dict(zip(df_splits["omop_person_id"], df_splits["split"]))

    print("Loading labels...")
    df_labels_map = {}
    for task in tasks:
        labels_path = os.path.join(args.data_dir, "benchmark", task, "labeled_patients.csv")
        df = pd.read_csv(labels_path)
        df["prediction_time"] = pd.to_datetime(df["prediction_time"])
        df["split"] = df["patient_id"].map(pid_to_split)
        df = df.dropna(subset=["split"])
        df_labels_map[task] = df
        print(f"  {task}: {len(df)} samples")

    # ── Interactive loop ──────────────────────────────────────────────────────
    print("\nReady. Press Enter to (re)compute stats, or type 'q' + Enter to quit.")
    run_all(tasks, grouped, df_labels_map, args.max_events)

    while True:
        try:
            line = input("[ Edit determine_num_sample.py, then press Enter ] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if line.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break
        run_all(tasks, grouped, df_labels_map, args.max_events)


if __name__ == "__main__":
    main()
