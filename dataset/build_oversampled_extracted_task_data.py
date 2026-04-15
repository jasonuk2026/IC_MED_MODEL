#!/usr/bin/env python3
"""
Build oversampled parquet files from extracted task data.

Input:
  extract_task_data-style parquet with at least:
    patient_id, prediction_time, label, event_ids

Output:
  parquet with one row per copy-level sample. For rows with more than
  --max_events events, the number of copies is determined by
  determine_num_sample.get_sample_n_times(...). Each oversampled copy draws a
  deterministic random subset of positions, then re-sorts those positions so
  the resulting event_ids remain in chronological order:

    left  = oldest event
    right = newest event

Example:
  source ~/miniforge3/bin/activate torch
  python dataset/build_oversampled_extracted_task_data.py \
    --task new_pancan \
    --input_dir extract_task_data/output \
    --output_dir extract_task_data_oversampled/output \
    --max_events 1000 \
    --epoch_idx 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from determine_num_sample import get_sample_n_times


DEFAULT_TASKS = [
    "new_hypertension",
    "new_hyperlipidemia",
    "new_pancan",
    "new_celiac",
    "new_lupus",
    "new_acutemi",
]


OUTPUT_SCHEMA = pa.schema([
    pa.field("patient_id", pa.int64()),
    pa.field("prediction_time", pa.string()),
    pa.field("label", pa.string()),
    pa.field("label_type", pa.string()),
    pa.field("split", pa.string()),
    pa.field("task_name", pa.string()),
    pa.field("num_events", pa.int64()),
    pa.field("event_ids", pa.list_(pa.int32())),
    pa.field("sample_event_indices", pa.string()),
    pa.field("oversample_copy_idx", pa.float64()),
])


def parse_binary_label(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return float(value) != 0.0

    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False

    try:
        return float(text) != 0.0
    except ValueError as exc:
        raise ValueError(f"Unsupported binary label value: {value!r}") from exc


def deterministic_seed(patient_id, prediction_time, copy_idx: int, epoch_idx: int) -> int:
    return hash((int(patient_id), str(prediction_time), int(copy_idx), int(epoch_idx)))


def build_copy_rows(
    row,
    *,
    task: str,
    max_events: int,
    epoch_idx: int,
) -> list[dict]:
    event_ids = list(row.event_ids)
    num_events = len(event_ids)
    if hasattr(row, "num_events") and pd.notna(row.num_events):
        num_events = int(row.num_events)

    base_record = {
        "patient_id": int(row.patient_id),
        "prediction_time": str(row.prediction_time),
        "label": str(row.label),
        "label_type": str(row.label_type) if hasattr(row, "label_type") and pd.notna(row.label_type) else "",
        "split": str(row.split) if hasattr(row, "split") and pd.notna(row.split) else "",
        "task_name": str(row.task_name) if hasattr(row, "task_name") and pd.notna(row.task_name) else task,
        "num_events": int(num_events),
    }

    if num_events <= max_events:
        return [{
            **base_record,
            "event_ids": [int(x) for x in event_ids],
            "sample_event_indices": None,
            "oversample_copy_idx": None,
        }]

    n_copies = get_sample_n_times(
        is_positive=parse_binary_label(row.label),
        N_EVENTS_LIMIT=max_events,
        ACTUAL_NUM_EVENTS=num_events,
        task=task,
    )

    records: list[dict] = []
    for copy_idx in range(n_copies):
        rng = random.Random(deterministic_seed(row.patient_id, row.prediction_time, copy_idx, epoch_idx))
        sample_indices = sorted(rng.sample(range(num_events), max_events))
        sampled_event_ids = [int(event_ids[i]) for i in sample_indices]
        records.append({
            **base_record,
            "event_ids": sampled_event_ids,
            "sample_event_indices": json.dumps(sample_indices),
            "oversample_copy_idx": float(copy_idx),
        })
    return records


def write_records(records: list[dict], out_path: Path) -> None:
    table = pa.Table.from_arrays(
        [
            pa.array([r["patient_id"] for r in records], type=pa.int64()),
            pa.array([r["prediction_time"] for r in records], type=pa.string()),
            pa.array([r["label"] for r in records], type=pa.string()),
            pa.array([r["label_type"] for r in records], type=pa.string()),
            pa.array([r["split"] for r in records], type=pa.string()),
            pa.array([r["task_name"] for r in records], type=pa.string()),
            pa.array([r["num_events"] for r in records], type=pa.int64()),
            pa.array([r["event_ids"] for r in records], type=pa.list_(pa.int32())),
            pa.array([r["sample_event_indices"] for r in records], type=pa.string()),
            pa.array([r["oversample_copy_idx"] for r in records], type=pa.float64()),
        ],
        schema=OUTPUT_SCHEMA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def process_one_file(
    *,
    task: str,
    split: str,
    in_path: Path,
    out_path: Path,
    max_events: int,
    epoch_idx: int,
) -> None:
    df = pd.read_parquet(in_path)
    print(f"[{task}/{split}] Loaded {len(df):,} rows from {in_path}")

    out_records: list[dict] = []
    n_oversampled_rows = 0
    total_copies = 0
    orig_pos = 0
    orig_neg = 0
    copy_pos = 0
    copy_neg = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{task}/{split}", dynamic_ncols=True):
        is_pos = parse_binary_label(row.label)
        if is_pos:
            orig_pos += 1
        else:
            orig_neg += 1
        copies = build_copy_rows(
            row,
            task=task,
            max_events=max_events,
            epoch_idx=epoch_idx,
        )
        total_copies += len(copies)
        if len(copies) > 1:
            n_oversampled_rows += 1
        if is_pos:
            copy_pos += len(copies)
        else:
            copy_neg += len(copies)
        out_records.extend(copies)

    write_records(out_records, out_path)
    print(
        f"[{task}/{split}] Wrote {len(out_records):,} rows to {out_path} "
        f"(original={len(df):,}, oversampled_samples={n_oversampled_rows:,}, "
        f"avg_copies={total_copies / max(len(df), 1):.2f})"
    )
    print(
        f"[{task}/{split}] Label stats: "
        f"original pos={orig_pos:,}, neg={orig_neg:,}; "
        f"copy-level pos={copy_pos:,}, neg={copy_neg:,}"
    )
    if orig_pos > 0 and orig_neg > 0:
        print(
            f"[{task}/{split}] Class ratio: "
            f"original pos/neg={orig_pos / orig_neg:.4f}; "
            f"copy-level pos/neg={copy_pos / copy_neg:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build oversampled parquet files from extracted task data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS)
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    parser.add_argument("--input_dir", default="extract_task_data/output")
    parser.add_argument("--output_dir", default="extract_task_data_oversampled/output")
    parser.add_argument("--max_events", type=int, default=1000)
    parser.add_argument("--epoch_idx", type=int, default=0,
                        help="Used in deterministic sampling seed so different epochs can produce different copies")
    parser.add_argument("--output_suffix", default="",
                        help="Optional suffix inserted before .parquet, e.g. '_e0'")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Tasks: {args.task}")
    print(f"Splits: {args.splits}")
    print(f"max_events: {args.max_events}")
    print(f"epoch_idx: {args.epoch_idx}")

    for task in args.task:
        for split in args.splits:
            in_path = Path(args.input_dir) / task / f"{split}.parquet"
            if not in_path.exists():
                raise FileNotFoundError(f"Missing input parquet: {in_path}")
            out_name = f"{split}{args.output_suffix}.parquet"
            out_path = Path(args.output_dir) / task / out_name
            process_one_file(
                task=task,
                split=split,
                in_path=in_path,
                out_path=out_path,
                max_events=args.max_events,
                epoch_idx=args.epoch_idx,
            )


if __name__ == "__main__":
    main()
