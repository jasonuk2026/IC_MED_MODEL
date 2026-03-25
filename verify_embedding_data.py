"""
Spot-check one sample from a generated embedding parquet against the original
labeled_patients.csv and ehrshot.csv.

Checks:
  1. Label consistency  — the sample's label matches labeled_patients.csv
  2. Event consistency  — each event's code/start exists in the patient's raw EHR
  3. Temporal filter    — all events have start < prediction_time
  4. Position sanity    — event_pos is within [0, history_len)

Usage:
  python verify_embedding_data.py --parquet <path> [--task <task>] [--index <i>]
"""

import argparse
import sys
import numpy as np
import pandas as pd


TASK_2_LABELS_PATH = {
    task: f"./EHRSHOT_ASSETS/benchmark/{task}/labeled_patients.csv"
    for task in [
        "new_hypertension", "new_hyperlipidemia", "new_pancan",
        "new_celiac", "new_lupus", "new_acutemi",
    ]
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True,
                        help="Path to the generated embedding parquet file.")
    parser.add_argument("--task", default=None,
                        help="Task name (inferred from filename if omitted).")
    parser.add_argument("--index", type=int, default=None,
                        help="Row index to check. Random if omitted.")
    parser.add_argument("--path_to_data_csv",
                        default="./EHRSHOT_ASSETS/data/ehrshot.csv")
    parser.add_argument("--path_to_labels_dir",
                        default="./EHRSHOT_ASSETS/benchmark")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load generated parquet ──────────────────────────────────────────────
    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df):,} samples, columns: {list(df.columns)}\n")

    # Pick a row
    idx = args.index if args.index is not None else np.random.randint(len(df))
    row = df.iloc[idx]
    print(f"Checking row index {idx}:")
    print(f"  patient_id      = {row['patient_id']}")
    print(f"  task            = {row['task']}")
    print(f"  split           = {row['split']}")
    print(f"  prediction_time = {row['prediction_time']}")
    print(f"  label           = {row['label']}")
    print(f"  history_len     = {row['history_len']}")
    print(f"  n_events        = {row['n_events']}")
    print(f"  pass_idx        = {row['pass_idx']}")
    print()

    task = args.task or row["task"]

    # ── Check 1: Label consistency ──────────────────────────────────────────
    path_to_labels = f"{args.path_to_labels_dir}/{task}/labeled_patients.csv"
    df_labels = pd.read_csv(path_to_labels)
    df_labels["prediction_time"] = pd.to_datetime(df_labels["prediction_time"])

    match = df_labels[
        (df_labels["patient_id"]      == row["patient_id"]) &
        (df_labels["prediction_time"] == row["prediction_time"])
    ]

    if match.empty:
        print("FAIL [label] — (patient_id, prediction_time) not found in labeled_patients.csv")
        sys.exit(1)

    orig_label = bool(match.iloc[0]["value"])
    gen_label  = bool(row["label"])
    if orig_label != gen_label:
        print(f"FAIL [label] — original={orig_label}, generated={gen_label}")
        sys.exit(1)
    print(f"PASS [label] — label={gen_label} matches labeled_patients.csv")

    # ── Check 2–4: Event consistency ────────────────────────────────────────
    print(f"\nLoading ehrshot.csv for patient {row['patient_id']} ...")
    df_ehr_full = pd.read_csv(args.path_to_data_csv, low_memory=False)
    df_ehr_full["start"] = pd.to_datetime(df_ehr_full["start"])

    df_patient = df_ehr_full[df_ehr_full["patient_id"] == row["patient_id"]].copy()
    df_history = (
        df_patient[df_patient["start"] < row["prediction_time"]]
        .sort_values("start")
        .reset_index(drop=True)
    )
    print(f"  Raw history before prediction_time: {len(df_history):,} events")

    # Check history_len matches
    if len(df_history) != row["history_len"]:
        print(f"FAIL [history_len] — raw={len(df_history)}, stored={row['history_len']}")
        sys.exit(1)
    print(f"PASS [history_len] — {row['history_len']} matches raw EHR")

    events = row["events"]
    print(f"\nChecking {len(events)} sampled events...")
    n_fail = 0
    for i, ev in enumerate(events):
        ev_code  = ev["code"]
        ev_start = pd.Timestamp(ev["start"])
        ev_pos   = ev["event_pos"]

        # Check 3: temporal filter
        if ev_start >= row["prediction_time"]:
            print(f"  FAIL [event {i}] start={ev_start} >= prediction_time={row['prediction_time']}")
            n_fail += 1
            continue

        # Check 4: position sanity
        if not (0 <= ev_pos < row["history_len"]):
            print(f"  FAIL [event {i}] event_pos={ev_pos} out of [0, {row['history_len']})")
            n_fail += 1
            continue

        # Check 2: event at event_pos in raw history matches code and start
        raw_event = df_history.iloc[ev_pos]
        raw_code  = str(raw_event["code"]) if pd.notna(raw_event["code"]) else "UNKNOWN"
        raw_start = raw_event["start"]

        if raw_code != ev_code or raw_start != ev_start:
            print(f"  FAIL [event {i}] pos={ev_pos}: "
                  f"stored=({ev_code}, {ev_start}), raw=({raw_code}, {raw_start})")
            n_fail += 1

    if n_fail == 0:
        print(f"PASS [events] — all {len(events)} events consistent with raw EHR")
    else:
        print(f"FAIL [events] — {n_fail}/{len(events)} events inconsistent")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
