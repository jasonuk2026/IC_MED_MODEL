#!/usr/bin/env python3
"""
Build evaluation parquet data aligned with the 01_gen_meta unique-event pipeline.

Compared with dataset/build_eval_task_data.py:
- event lookup uses (omop_table, code, value, unit)
- event ids come from event_index.parquet built from extract_unique_events.py
- output schema stays the same:
    patient_id (int64), task_idx (int16), label (int8), event_ids (list<int32>)

Usage:
    python 01_gen_meta/build_eval_task_data.py
    python 01_gen_meta/build_eval_task_data.py --tasks new_hypertension new_pancan --splits val test
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
if DATASET_DIR not in sys.path:
    sys.path.insert(0, DATASET_DIR)

from eval_sample_strategy import strategy as _strategy_fn


EHRSHOT_ASSETS = "data/EHRSHOT_ASSETS"
DEFAULT_EMBED_DIR = "data/01_outputs/qwen3_0.6b_embs"

VALID_TASKS = [
    "guo_los", "guo_readmission", "guo_icu",
    "new_hypertension", "new_hyperlipidemia", "new_pancan",
    "new_celiac", "new_lupus", "new_acutemi",
    "lab_thrombocytopenia", "lab_hyperkalemia", "lab_hypoglycemia",
    "lab_hyponatremia", "lab_anemia",
    "chexpert",
]

TASK_2_DISEASE_NAME = {
    "new_hypertension": "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan": "pancreatic cancer",
    "new_celiac": "celiac disease",
    "new_lupus": "systemic lupus erythematosus",
    "new_acutemi": "acute myocardial infarction",
}

TASK_2_IDX = {task: idx for idx, task in enumerate(sorted(TASK_2_DISEASE_NAME))}

PARQUET_SCHEMA = pa.schema([
    pa.field("patient_id", pa.int64()),
    pa.field("task_idx", pa.int16()),
    pa.field("label", pa.int8()),
    pa.field("event_ids", pa.list_(pa.int32())),
])


_GROUPED = {}
_EMBED_IDX = {}
_TASK_IDX = 0
_NUM_EVENTS = 1000
_INCLUDE_CONDITION_OCCURRENCE = False


def normalise_event_key(omop_table, code, value, unit):
    return (
        str(omop_table if omop_table is not None else "").strip(),
        str(code if code is not None else "").strip(),
        str(value if value is not None else "").strip(),
        str(unit if unit is not None else "").strip(),
    )


def _is_positive(label_val):
    if isinstance(label_val, bool):
        return label_val
    s = str(label_val).strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(float(s)) != 0
    except (ValueError, TypeError):
        return False


def load_embed_idx(embed_dir):
    idx_path = os.path.join(embed_dir, "event_index.parquet")
    df = pd.read_parquet(
        idx_path,
        columns=["event_id", "omop_table", "code", "value", "unit"],
    )
    result = {}
    for row in df.itertuples(index=False):
        key = normalise_event_key(row.omop_table, row.code, row.value, row.unit)
        result[key] = int(row.event_id)
    print("  Loaded {} embedding index entries from {}".format(len(result), idx_path))
    return result


def _process_eval_sample(args):
    pid, pred_time, label_val = args

    patient_events = _GROUPED.get(pid)
    if patient_events is None:
        return None

    events_df = patient_events.loc[patient_events["start"] < pred_time]
    if len(events_df) == 0:
        return None

    valid_event_eids = []
    for ev in events_df.itertuples(index=False):
        if (not _INCLUDE_CONDITION_OCCURRENCE) and ev.omop_table == "condition_occurrence":
            continue

        embed_val = ev.value if isinstance(ev.value, str) else ""
        embed_unit = ev.unit if isinstance(ev.unit, str) else ""
        norm_key = normalise_event_key(ev.omop_table, ev.code, embed_val, embed_unit)
        eid = _EMBED_IDX.get(norm_key)
        if eid is not None:
            valid_event_eids.append(eid)

    if not valid_event_eids:
        return None

    selected_positions = _strategy_fn(list(range(len(valid_event_eids))), _NUM_EVENTS)
    event_ids = [valid_event_eids[i] for i in selected_positions]

    return {
        "patient_id": int(pid),
        "task_idx": _TASK_IDX,
        "label": 1 if _is_positive(label_val) else 0,
        "event_ids": event_ids,
    }


def _flush(writer, records):
    table = pa.table(
        {
            "patient_id": pa.array([r["patient_id"] for r in records], type=pa.int64()),
            "task_idx": pa.array([r["task_idx"] for r in records], type=pa.int16()),
            "label": pa.array([r["label"] for r in records], type=pa.int8()),
            "event_ids": pa.array([r["event_ids"] for r in records], type=pa.list_(pa.int32())),
        },
        schema=PARQUET_SCHEMA,
    )
    writer.write_table(table)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build evaluation dataset using event_index.parquet from 01_gen_meta.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(sorted(TASK_2_DISEASE_NAME.keys())),
        choices=VALID_TASKS,
        help="Tasks to process",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to process",
    )
    parser.add_argument(
        "--num_events",
        type=int,
        default=1000,
        help="Max events to sample per patient via eval_sample_strategy",
    )
    parser.add_argument(
        "--output_dir",
        default="data/eval_data_latest",
    )
    parser.add_argument("--data_dir", default=EHRSHOT_ASSETS)
    parser.add_argument("--embed_dir", default=DEFAULT_EMBED_DIR)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--buffer_size", type=int, default=2000)
    parser.add_argument(
        "--include_condition_occurrence",
        action="store_true",
        help="Keep condition_occurrence events instead of matching the default preprocessing behavior.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = list(dict.fromkeys(args.tasks))
    data_dir = args.data_dir
    n_workers = args.num_workers or mp.cpu_count()

    invalid_tasks = [task for task in tasks if task not in TASK_2_IDX]
    if invalid_tasks:
        raise ValueError(
            "Tasks %r are not in TASK_2_IDX. Only disease-prediction tasks are supported." % invalid_tasks
        )

    print("Tasks: {}".format(tasks))
    print("Splits: {}".format(args.splits))
    print("num_events: {}  workers: {}".format(args.num_events, n_workers))

    print("Loading embedding index...")
    embed_idx = load_embed_idx(args.embed_dir)

    print("Loading raw EHR data...")
    ehr_path = os.path.join(data_dir, "data", "ehrshot.csv")
    df_ehr = pd.read_csv(
        ehr_path,
        low_memory=False,
        dtype={"value": str, "unit": str},
    )
    if df_ehr.columns[0].startswith("Unnamed"):
        df_ehr = df_ehr.drop(columns=[df_ehr.columns[0]])
    df_ehr["start"] = pd.to_datetime(df_ehr["start"])
    df_ehr = df_ehr.sort_values(["patient_id", "start"])
    print("  {} events across {} patients".format(len(df_ehr), df_ehr["patient_id"].nunique()))

    print("Indexing events by patient_id...")
    grouped = dict(list(df_ehr.groupby("patient_id")))

    global _GROUPED, _EMBED_IDX, _TASK_IDX, _NUM_EVENTS, _INCLUDE_CONDITION_OCCURRENCE
    _GROUPED = grouped
    _EMBED_IDX = embed_idx
    _NUM_EVENTS = args.num_events
    _INCLUDE_CONDITION_OCCURRENCE = args.include_condition_occurrence

    print("Loading patient splits...")
    splits_path = os.path.join(data_dir, "splits", "person_id_map.csv")
    df_splits = pd.read_csv(splits_path)
    pid_to_split = dict(zip(df_splits["omop_person_id"], df_splits["split"]))

    for task in tasks:
        print("Loading labels for task: {}".format(task))
        labels_path = os.path.join(data_dir, "benchmark", task, "labeled_patients.csv")
        df_labels = pd.read_csv(labels_path)
        df_labels["prediction_time"] = pd.to_datetime(df_labels["prediction_time"])
        df_labels["split"] = df_labels["patient_id"].map(pid_to_split)
        df_labels = df_labels.dropna(subset=["split"])
        df_labels = df_labels[df_labels["split"].isin(args.splits)]
        print("  {} labeled samples in splits {}".format(len(df_labels), args.splits))

        _TASK_IDX = TASK_2_IDX[task]
        imap_chunksize = max(1, len(df_labels) // (n_workers * 20)) if len(df_labels) > 0 else 1
        out_dir = os.path.join(args.output_dir, task)
        os.makedirs(out_dir, exist_ok=True)

        for split in args.splits:
            df_split = df_labels[df_labels["split"] == split]
            if len(df_split) == 0:
                print("  [{}:{}] No samples - skipping.".format(task, split))
                continue

            print("  [{}:{}] Processing {} samples ...".format(task, split, len(df_split)))
            worker_args = [
                (row.patient_id, row.prediction_time, row.value)
                for row in df_split.itertuples(index=False)
            ]

            out_path = os.path.join(out_dir, "{}.parquet".format(split))
            writer = pq.ParquetWriter(out_path, PARQUET_SCHEMA)
            buffer = []
            n_written = 0
            n_skipped = 0

            with mp.Pool(processes=n_workers) as pool:
                for result in tqdm(
                    pool.imap(_process_eval_sample, worker_args, chunksize=imap_chunksize),
                    total=len(worker_args),
                    desc="    {}:{}".format(task, split),
                ):
                    if result is None:
                        n_skipped += 1
                        continue
                    buffer.append(result)
                    if len(buffer) >= args.buffer_size:
                        _flush(writer, buffer)
                        n_written += len(buffer)
                        buffer.clear()

            if buffer:
                _flush(writer, buffer)
                n_written += len(buffer)
            writer.close()

            if n_written > 0:
                label_df = pd.read_parquet(out_path, columns=["label"])
                pos = int((label_df["label"] == 1).sum())
            else:
                pos = 0
            neg = n_written - pos
            print(
                "    -> {} rows ({} pos / {} neg), {} skipped -> {}".format(
                    n_written, pos, neg, n_skipped, out_path
                )
            )

    print("Done.")


if __name__ == "__main__":
    main()
