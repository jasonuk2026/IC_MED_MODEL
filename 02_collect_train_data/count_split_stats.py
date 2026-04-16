"""
Quick stats script: count positive/negative copy-level samples per split.

Loads labels + splits + EHR event counts (no embeddings, no text building),
then applies the same get_sample_n_times policy as build_task_data.py.
Use this to rapidly iterate on determine_num_sample.py without a full build.

Usage:
    python count_split_stats.py --task new_pancan --max_events 500
    python count_split_stats.py --task new_pancan --max_events 500 --all_tasks
"""

import argparse
import os

import pandas as pd

from determine_num_sample import get_sample_n_times

EHRSHOT_ASSETS = "EHRSHOT_ASSETS"

VALID_TASKS = [
    "guo_los", "guo_readmission", "guo_icu",
    "new_hypertension", "new_hyperlipidemia", "new_pancan",
    "new_celiac", "new_lupus", "new_acutemi",
    "lab_thrombocytopenia", "lab_hyperkalemia", "lab_hypoglycemia",
    "lab_hyponatremia", "lab_anemia",
    "chexpert",
]

SPLITS = ["train", "val", "test"]


def _is_positive(label_val) -> bool:
    try:
        return int(float(label_val)) != 0
    except (ValueError, TypeError):
        return False


def compute_stats(task: str, grouped: dict, data_dir: str, max_events) -> None:
    labels_path = os.path.join(data_dir, "benchmark", task, "labeled_patients.csv")
    df_labels = pd.read_csv(labels_path)
    df_labels["prediction_time"] = pd.to_datetime(df_labels["prediction_time"])

    splits_path = os.path.join(data_dir, "splits", "person_id_map.csv")
    df_splits = pd.read_csv(splits_path)
    pid_to_split = dict(zip(df_splits["omop_person_id"], df_splits["split"]))
    df_labels["split"] = df_labels["patient_id"].map(pid_to_split)
    df_labels = df_labels.dropna(subset=["split"])

    # counts[split][pos/neg] = copy count
    counts = {s: {"pos": 0, "neg": 0} for s in SPLITS}
    raw_counts = {s: {"pos": 0, "neg": 0} for s in SPLITS}  # before oversampling

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
            n = get_sample_n_times(
                is_positive=is_pos,
                N_EVENTS_LIMIT=max_events,
                ACTUAL_NUM_EVENTS=M,
                task=task,
            )
        counts[split][key] += n

    # ── Print table ──────────────────────────────────────────────────────────────
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


def main():
    parser = argparse.ArgumentParser(description="Count pos/neg copy-level samples per split")
    parser.add_argument("--task", type=str, choices=VALID_TASKS, default=None)
    parser.add_argument("--all_tasks", action="store_true", help="Run for all tasks")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=EHRSHOT_ASSETS)
    args = parser.parse_args()

    if not args.task and not args.all_tasks:
        parser.error("Provide --task <name> or --all_tasks")

    tasks = VALID_TASKS if args.all_tasks else [args.task]

    print("Loading raw EHR data...")
    ehr_path = os.path.join(args.data_dir, "data", "ehrshot.csv")
    df_ehr = pd.read_csv(ehr_path, low_memory=False, usecols=["patient_id", "start"])
    df_ehr["start"] = pd.to_datetime(df_ehr["start"])
    grouped = dict(list(df_ehr.groupby("patient_id")))
    print(f"  {len(df_ehr):,} events across {len(grouped):,} patients")

    for task in tasks:
        compute_stats(task, grouped, args.data_dir, args.max_events)

    print()


if __name__ == "__main__":
    main()
