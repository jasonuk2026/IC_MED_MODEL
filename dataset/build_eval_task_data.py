"""
build_eval_task_data.py

Build evaluation dataset from EHRSHOT benchmark tasks.

Unlike build_task_data.py:
- No oversampling: exactly one output row per labeled sample.
- Event selection uses eval_sample_strategy.strategy() instead of random.sample.
- Output schema: patient_id (int64), task_idx (int16), label (int8),
                 event_ids (list<int32>)
- Outputs separate parquet files per split under --output_dir/{task}/{split}.parquet

Usage:
    python build_eval_task_data.py --task new_hypertension
    python build_eval_task_data.py --task new_hypertension --splits val test
    python build_eval_task_data.py --task new_hypertension --num_events 500
"""

import argparse
import multiprocessing as mp
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from eval_sample_strategy import strategy as _strategy_fn

# ── Constants ─────────────────────────────────────────────────────────────────

EHRSHOT_ASSETS = "EHRSHOT_ASSETS"

DEFAULT_EMBED_DIR = "data/biolinkbert_embeddings"

VALID_TASKS = [
    "guo_los", "guo_readmission", "guo_icu",
    "new_hypertension", "new_hyperlipidemia", "new_pancan",
    "new_celiac", "new_lupus", "new_acutemi",
    "lab_thrombocytopenia", "lab_hyperkalemia", "lab_hypoglycemia",
    "lab_hyponatremia", "lab_anemia",
    "chexpert",
]

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}

TASK_2_IDX: dict[str, int] = {t: i for i, t in enumerate(sorted(TASK_2_DISEASE_NAME))}

PARQUET_SCHEMA = pa.schema([
    pa.field("patient_id", pa.int64()),
    pa.field("task_idx",   pa.int16()),
    pa.field("label",      pa.int8()),
    pa.field("event_ids",  pa.list_(pa.int32())),
])

# ── Module-level globals (inherited by forked workers) ────────────────────────

_GROUPED:   dict = {}   # patient_id -> DataFrame of events
_EMBED_IDX: dict = {}   # (code, value, unit) -> embedding_idx (int)
_TASK_IDX:  int  = 0    # integer index for the current task
_NUM_EVENTS: int = 1000  # events to sample per patient


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_event_key(code, value, unit) -> tuple[str, str, str]:
    return (
        str(code  if code  is not None else "").strip(),
        str(value if value is not None else "").strip(),
        str(unit  if unit  is not None else "").strip(),
    )


def _is_positive(label_val) -> bool:
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


def load_embed_idx(embed_dir: str) -> dict:
    idx_path = os.path.join(embed_dir, "event_index.parquet")
    df = pd.read_parquet(idx_path, columns=["event_id", "code", "value", "unit"])
    result = {}
    for row in df.itertuples(index=False):
        key = normalise_event_key(row.code, row.value, row.unit)
        result[key] = int(row.event_id)
    print(f"  Loaded {len(result)} embedding index entries from {idx_path}")
    return result


# ── Worker ────────────────────────────────────────────────────────────────────

def _process_eval_sample(args: tuple) -> dict | None:
    """Process one labeled sample → one output row (no oversampling).

    Returns None if the patient has no valid embeddable events before
    prediction_time (these rows are silently dropped).
    """
    pid, pred_time, label_val = args

    patient_events = _GROUPED.get(pid)
    if patient_events is None:
        return None

    events_df = patient_events.loc[patient_events["start"] < pred_time]
    if len(events_df) == 0:
        return None

    # Collect embedding indices for all valid events (non-condition_occurrence,
    # has a BioLinkBERT embedding).  valid_event_eids[i] = embedding_idx of
    # the i-th valid event, so event_idxs = range(len(valid_event_eids)).
    valid_event_eids: list[int] = []
    for ev in events_df.itertuples(index=False):
        if ev.omop_table == "condition_occurrence":
            continue
        embed_val  = ev.value if isinstance(ev.value, str) else ""
        embed_unit = ev.unit  if isinstance(ev.unit,  str) else ""
        norm_key = normalise_event_key(ev.code, embed_val, embed_unit)
        eid = _EMBED_IDX.get(norm_key)
        if eid is not None:
            valid_event_eids.append(eid)

    if not valid_event_eids:
        return None

    # strategy() receives 0-based positions into valid_event_eids and returns
    # a sorted subset of those positions.
    selected_positions = _strategy_fn(list(range(len(valid_event_eids))), _NUM_EVENTS)
    event_ids = [valid_event_eids[i] for i in selected_positions]

    return {
        "patient_id": int(pid),
        "task_idx":   _TASK_IDX,
        "label":      1 if _is_positive(label_val) else 0,
        "event_ids":  event_ids,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build evaluation dataset (one row per labeled sample).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",        required=True, choices=VALID_TASKS)
    parser.add_argument("--splits",      nargs="+", default=["val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to process")
    parser.add_argument("--num_events",  type=int, default=1000,
                        help="Max events to sample per patient via eval_sample_strategy")
    parser.add_argument("--output_dir",  default=os.path.join(EHRSHOT_ASSETS, "llm_eval_data"))
    parser.add_argument("--data_dir",    default=EHRSHOT_ASSETS)
    parser.add_argument("--embed_dir",   default=DEFAULT_EMBED_DIR)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--buffer_size", type=int, default=2000)
    args = parser.parse_args()

    task      = args.task
    data_dir  = args.data_dir
    n_workers = args.num_workers or mp.cpu_count()

    if task not in TASK_2_IDX:
        raise ValueError(f"Task '{task}' is not in TASK_2_IDX. "
                         "Only disease-prediction tasks are supported for eval embedding.")

    print(f"Task: {task}  (task_idx={TASK_2_IDX[task]})")
    print(f"Splits: {args.splits}")
    print(f"num_events: {args.num_events}  workers: {n_workers}")

    # ── 1. Load embedding index ───────────────────────────────────────────────
    print("Loading BioLinkBERT embedding index...")
    embed_idx = load_embed_idx(args.embed_dir)

    # ── 2. Load labels ────────────────────────────────────────────────────────
    print(f"Loading labels for task: {task}")
    labels_path = os.path.join(data_dir, "benchmark", task, "labeled_patients.csv")
    df_labels = pd.read_csv(labels_path)
    df_labels["prediction_time"] = pd.to_datetime(df_labels["prediction_time"])

    # ── 3. Load splits ────────────────────────────────────────────────────────
    print("Loading patient splits...")
    splits_path = os.path.join(data_dir, "splits", "person_id_map.csv")
    df_splits   = pd.read_csv(splits_path)
    pid_to_split = dict(zip(df_splits["omop_person_id"], df_splits["split"]))
    df_labels["split"] = df_labels["patient_id"].map(pid_to_split)
    df_labels = df_labels.dropna(subset=["split"])

    # Filter to requested splits
    df_labels = df_labels[df_labels["split"].isin(args.splits)]
    print(f"  {len(df_labels)} labeled samples in splits {args.splits}")

    # ── 4. Load raw EHR data ──────────────────────────────────────────────────
    print("Loading raw EHR data...")
    ehr_path = os.path.join(data_dir, "data", "ehrshot.csv")
    df_ehr   = pd.read_csv(ehr_path, low_memory=False, dtype={"value": str, "unit": str})
    if df_ehr.columns[0].startswith("Unnamed"):
        df_ehr = df_ehr.drop(columns=[df_ehr.columns[0]])
    df_ehr["start"] = pd.to_datetime(df_ehr["start"])
    df_ehr = df_ehr.sort_values(["patient_id", "start"])
    print(f"  {len(df_ehr)} events across {df_ehr['patient_id'].nunique()} patients")

    print("Indexing events by patient_id...")
    grouped = dict(list(df_ehr.groupby("patient_id")))

    # ── 5. Set globals before forking ─────────────────────────────────────────
    global _GROUPED, _EMBED_IDX, _TASK_IDX, _NUM_EVENTS
    _GROUPED    = grouped
    _EMBED_IDX  = embed_idx
    _TASK_IDX   = TASK_2_IDX[task]
    _NUM_EVENTS = args.num_events

    # ── 6. Process each split independently ───────────────────────────────────
    imap_chunksize = max(1, len(df_labels) // (n_workers * 20))
    out_dir = os.path.join(args.output_dir, task)
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits:
        df_split = df_labels[df_labels["split"] == split]
        if len(df_split) == 0:
            print(f"  [{split}] No samples — skipping.")
            continue

        print(f"  [{split}] Processing {len(df_split)} samples …")
        worker_args = [
            (row.patient_id, row.prediction_time, row.value)
            for row in df_split.itertuples(index=False)
        ]

        out_path = os.path.join(out_dir, f"{split}.parquet")
        writer   = pq.ParquetWriter(out_path, PARQUET_SCHEMA)
        buffer: list[dict] = []
        n_written = n_skipped = 0

        with mp.Pool(processes=n_workers) as pool:
            for result in tqdm(
                pool.imap(_process_eval_sample, worker_args, chunksize=imap_chunksize),
                total=len(worker_args),
                desc=f"    {split}",
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

        pos = sum(1 for r in pd.read_parquet(out_path, columns=["label"])["label"] if r == 1)
        neg = n_written - pos
        print(f"    → {n_written} rows ({pos} pos / {neg} neg), "
              f"{n_skipped} skipped → {out_path}")

    print("Done.")


def _flush(writer: pq.ParquetWriter, records: list[dict]) -> None:
    table = pa.table(
        {
            "patient_id": pa.array([r["patient_id"] for r in records], type=pa.int64()),
            "task_idx":   pa.array([r["task_idx"]   for r in records], type=pa.int16()),
            "label":      pa.array([r["label"]       for r in records], type=pa.int8()),
            "event_ids":  pa.array(
                [r["event_ids"] for r in records],
                type=pa.list_(pa.int32()),
            ),
        },
        schema=PARQUET_SCHEMA,
    )
    writer.write_table(table)


if __name__ == "__main__":
    main()
