"""
Build LLM-ready dataset from EHRSHOT benchmark tasks (v6).

Differences from v5:
- Loads BioLinkBERT embedding index from --embed_dir.
  Each event JSON now contains an "embedding_idx" field (int or null) pointing
  to the corresponding row in embeddings.npy.
- Condition events (omop_table == "condition_occurrence") are excluded entirely.

Usage:
    python build_llm_dataset_v6.py --task guo_los
    python build_llm_dataset_v6.py --task guo_los --max_events 500 --num_workers 16
    python build_llm_dataset_v6.py --task guo_los --max_events 500 --buffer_size 1000 \\
        --embed_dir /scratch/u6dk/zduan.u6dk/codes/ehr/data/biolinkbert_embeddings
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from determine_num_sample import get_sample_n_times


# ── Constants ──────────────────────────────────────────────────────────────────

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

TASK_DESCRIPTIONS = {
    "guo_los": "Predict whether this patient's hospital stay will be >= 7 days (long length of stay). Prediction is made at 11:59 PM on the day of admission.",
    "guo_readmission": "Predict whether this patient will be readmitted within 30 days of discharge.",
    "guo_icu": "Predict whether this patient will be transferred to the ICU during this admission.",
    "new_hypertension": "Predict whether this patient will be newly diagnosed with essential hypertension within 1 year after discharge.",
    "new_hyperlipidemia": "Predict whether this patient will be newly diagnosed with hyperlipidemia within 1 year after discharge.",
    "new_pancan": "Predict whether this patient will be newly diagnosed with pancreatic cancer within 1 year after discharge.",
    "new_celiac": "Predict whether this patient will be newly diagnosed with celiac disease within 1 year after discharge.",
    "new_lupus": "Predict whether this patient will be newly diagnosed with lupus within 1 year after discharge.",
    "new_acutemi": "Predict whether this patient will be newly diagnosed with acute myocardial infarction within 1 year after discharge.",
    "lab_thrombocytopenia": "Predict the severity of thrombocytopenia (low platelet count): 0=normal, 1=mild (<150), 2=moderate (<100), 3=severe (<50).",
    "lab_hyperkalemia": "Predict the severity of hyperkalemia (high potassium): 0=normal, 1=mild (>5.5), 2=moderate (>6.0), 3=severe (>7.0).",
    "lab_hypoglycemia": "Predict the severity of hypoglycemia (low glucose): 0=normal, 1=mild (<3.9), 2=moderate (<3.5), 3=severe (<3.0).",
    "lab_hyponatremia": "Predict the severity of hyponatremia (low sodium): 0=normal, 1=mild (<=135), 2=moderate (<130), 3=severe (<125).",
    "lab_anemia": "Predict the severity of anemia (low hemoglobin): 0=normal, 1=mild (<120), 2=moderate (<110), 3=severe (<70).",
    "chexpert": "Predict the 14 CheXpert chest X-ray findings (multilabel binary). Labels: No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Lesion, Lung Opacity, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices.",
}

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
    "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

OMOP_TABLE_PREFIX = {
    "drug_exposure": "Drug",
    "condition_occurrence": "Condition",
    "measurement": "Measurement",
    "procedure_occurrence": "Procedure",
    "observation": "Observation",
    "visit_occurrence": "Visit",
    "visit_detail": "Visit Detail",
    "device_exposure": "Device",
    "death": "Death",
    "note": "Note",
    "person": "Demographics",
}


# ── Module-level globals (inherited by forked workers via copy-on-write) ───────

_GROUPED: dict = {}
_CODE_2_DESC: dict = {}
_EMBED_IDX: dict = {}   # (norm_code, norm_value, norm_unit) -> event_id (int)


# ── Embedding index helpers ────────────────────────────────────────────────────

def normalise_event_key(code, value, unit) -> tuple[str, str, str]:
    return (
        str(code  if code  is not None else "").strip(),
        str(value if value is not None else "").strip(),
        str(unit  if unit  is not None else "").strip(),
    )


def load_embed_idx(embed_dir: str) -> dict:
    idx_path = os.path.join(embed_dir, "event_index.parquet")
    df = pd.read_parquet(idx_path, columns=["event_id", "code", "value", "unit"])
    result = {}
    for row in df.itertuples(index=False):
        key = normalise_event_key(row.code, row.value, row.unit)
        result[key] = int(row.event_id)
    print(f"  Loaded {len(result)} embedding index entries from {idx_path}")
    return result


# ── Code → description mapping ─────────────────────────────────────────────────

def load_code_description_map(data_dir: str) -> dict[str, str]:
    t2c_path = os.path.join(data_dir, "models", "clmbr", "token_2_code.json")
    t2d_path = os.path.join(data_dir, "models", "clmbr", "token_2_description.json")
    with open(t2c_path) as f:
        token_2_code: dict[str, str] = json.load(f)
    with open(t2d_path) as f:
        token_2_desc: dict[str, str] = json.load(f)
    code_2_desc: dict[str, str] = {}
    for token_id, code in token_2_code.items():
        if token_id in token_2_desc:
            desc = token_2_desc[token_id]
            if desc and desc != code:
                code_2_desc[code] = desc
    print(f"  Loaded {len(code_2_desc)} code->description mappings")
    return code_2_desc


# ── Event formatting ───────────────────────────────────────────────────────────

def _build_timeline_from_df(events_df: pd.DataFrame) -> tuple[str, int, dict[str, int]]:
    n_hits = 0
    misses: dict[str, int] = {}
    lines: list[str] = []

    for _, ev in events_df.iterrows():
        # Skip condition events
        if ev["omop_table"] == "condition_occurrence":
            continue

        code = ev["code"]
        if code in _CODE_2_DESC:
            description = _CODE_2_DESC[code]
            n_hits += 1
        else:
            description = None
            misses[code] = misses.get(code, 0) + 1

        # value and unit are raw CSV strings (loaded with dtype=str) or NaN.
        # Use isinstance checks instead of isnan to handle both cases cleanly.
        raw_value = ev["value"]   # str | float(NaN)
        raw_unit  = ev["unit"]    # str | float(NaN)

        has_value = isinstance(raw_value, str) and raw_value.strip() != ""
        if has_value:
            numeric = pd.to_numeric(raw_value, errors="coerce")
            value = (
                None
                if (isinstance(numeric, float) and math.isnan(numeric))
                else float(numeric)
            )
        else:
            value = None

        unit = raw_unit.strip() if isinstance(raw_unit, str) and raw_unit.strip() != "" else None

        # Embedding index lookup using the raw CSV strings directly — must match
        # how the embedding script built its keys (also from raw CSV strings via
        # dtype=str).  Converting to float and back would change the string
        # representation (e.g. "62.597999572753906" → "62.59799957275391").
        embed_val  = raw_value if isinstance(raw_value, str) else ""
        embed_unit = raw_unit  if isinstance(raw_unit,  str) else ""
        norm_key = normalise_event_key(code, embed_val, embed_unit)
        if norm_key not in _EMBED_IDX:
            raise KeyError(
                f"No embedding found for event key {norm_key!r} "
                f"(code={code!r}, raw_value={raw_value!r}, raw_unit={raw_unit!r}). "
                "This is a bug — all events should have a corresponding embedding."
            )
        embedding_idx = _EMBED_IDX[norm_key]

        event = {
            "time": str(ev["start"])[:16],
            "type": OMOP_TABLE_PREFIX.get(ev["omop_table"], ev["omop_table"]),
            "table": ev["omop_table"],
            "code": code,
            "description": description,
            "value": value,
            "unit": unit,
            "embedding_idx": embedding_idx,
        }
        lines.append(json.dumps(event, ensure_ascii=False))

    return "\n".join(lines), n_hits, misses


def _label_str(label_val, is_chexpert: bool) -> str:
    if is_chexpert:
        label_bits = bin(int(label_val))[2:].zfill(14)
        findings = [CHEXPERT_LABELS[i] for i, b in enumerate(label_bits) if b == "1"]
        return "; ".join(findings) if findings else "None"
    return str(label_val)


def _is_positive(label_val) -> bool:
    try:
        return int(float(label_val)) != 0
    except (ValueError, TypeError):
        return False


# ── Worker: one copy per call ──────────────────────────────────────────────────

def _process_copy(args: tuple) -> tuple[dict, int, dict[str, int]]:
    """Process a single copy of a labeled sample.

    copy_idx=None  → short timeline, v4 path (all events, no sampling).
    copy_idx=int   → oversampled copy, random sample of max_events events.

    Args:
        args: (pid, pred_time, label_val, label_type, split,
               max_events, task_desc, task, is_chexpert, copy_idx, M)
    """
    (pid, pred_time, label_val, label_type, split,
     max_events, task_desc, task, is_chexpert,
     copy_idx, M) = args

    label = _label_str(label_val, is_chexpert)

    patient_events = _GROUPED.get(pid)
    if patient_events is None or M == 0:
        return {
            "patient_id": pid,
            "prediction_time": str(pred_time),
            "label": label,
            "label_type": label_type,
            "split": split,
            "timeline": "",
            "task_name": task,
            "task_description": task_desc,
            "num_events": M,
            "sample_event_indices": None,
            "oversample_copy_idx": copy_idx,
        }, 0, {}

    events_df = patient_events.loc[patient_events["start"] < pred_time]

    if copy_idx is None:
        timeline_text, n_hits, misses = _build_timeline_from_df(events_df)
        sample_indices_str = None
    else:
        rng = random.Random(hash((int(pid), str(pred_time), copy_idx)))
        sample_indices = sorted(rng.sample(range(M), max_events))
        sampled_df = events_df.iloc[sample_indices]
        timeline_text, n_hits, misses = _build_timeline_from_df(sampled_df)
        sample_indices_str = json.dumps(sample_indices)

    return {
        "patient_id": pid,
        "prediction_time": str(pred_time),
        "label": label,
        "label_type": label_type,
        "split": split,
        "timeline": timeline_text,
        "task_name": task,
        "task_description": task_desc,
        "num_events": M,
        "sample_event_indices": sample_indices_str,
        "oversample_copy_idx": copy_idx,
    }, n_hits, misses


# ── Coverage report ────────────────────────────────────────────────────────────

def print_coverage_report(n_hits: int, misses: dict[str, int], top_n: int = 20) -> None:
    n_misses = sum(misses.values())
    total = n_hits + n_misses
    if total == 0:
        print("  No events processed.")
        return
    print()
    print("=" * 60)
    print("Code→Description Coverage Report")
    print("=" * 60)
    print(f"  Total events in timelines : {total:>10,}")
    print(f"  Mapped (description found): {n_hits:>10,}  ({100. * n_hits / total:.1f}%)")
    print(f"  Unmapped (raw code kept)  : {n_misses:>10,}  ({100. * n_misses / total:.1f}%)")
    print(f"  Unique unmapped codes     : {len(misses):>10,}")
    if misses:
        print()
        print(f"  Top {top_n} unmapped codes by event frequency:")
        for code, count in sorted(misses.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            print(f"    {code:<40s}  {count:>8,} events  ({100. * count / total:.2f}%)")
    print("=" * 60)
    print()


# ── Copy-args helpers ──────────────────────────────────────────────────────────

def _compute_M(pid, pred_time, grouped: dict) -> int:
    patient_df = grouped.get(pid)
    if patient_df is None:
        return 0
    return int((patient_df["start"] < pred_time).sum())


def _count_copies(df_labels: pd.DataFrame, grouped: dict, max_events, task: str = "") -> tuple[int, int]:
    """Return (total_copy_count, n_oversampled_samples)."""
    total = 0
    n_oversampled = 0
    for row in df_labels.itertuples(index=False):
        M = _compute_M(row.patient_id, row.prediction_time, grouped)
        if max_events is None or M <= max_events:
            total += 1
        else:
            n = get_sample_n_times(
                is_positive=_is_positive(row.value),
                N_EVENTS_LIMIT=max_events,
                ACTUAL_NUM_EVENTS=M,
                task=task,
            )
            total += n
            n_oversampled += 1
    return total, n_oversampled


def _iter_copy_args(
    df_labels: pd.DataFrame,
    grouped: dict,
    max_events,
    task_desc: str,
    task: str,
    is_chexpert: bool,
) -> Iterator[tuple]:
    """Lazily yield one args-tuple per copy. Never materialises the full list."""
    for row in df_labels.itertuples(index=False):
        pid = row.patient_id
        pred_time = row.prediction_time
        label_val = row.value
        label_type = row.label_type
        split = row.split
        M = _compute_M(pid, pred_time, grouped)

        base = (pid, pred_time, label_val, label_type, split,
                max_events, task_desc, task, is_chexpert)

        if max_events is None or M <= max_events:
            yield base + (None, M)
        else:
            n_copies = get_sample_n_times(
                is_positive=_is_positive(label_val),
                N_EVENTS_LIMIT=max_events,
                ACTUAL_NUM_EVENTS=M,
                task=task,
            )
            for copy_idx in range(n_copies):
                yield base + (copy_idx, M)


# ── Parquet schema (explicit to avoid null-type inference on all-None batches) ──

PARQUET_SCHEMA = pa.schema([
    pa.field("patient_id", pa.int64()),
    pa.field("prediction_time", pa.string()),
    pa.field("label", pa.string()),
    pa.field("label_type", pa.string()),
    pa.field("split", pa.string()),
    pa.field("timeline", pa.string()),
    pa.field("task_name", pa.string()),
    pa.field("task_description", pa.string()),
    pa.field("num_events", pa.int64()),
    pa.field("sample_event_indices", pa.string()),   # JSON list string, or null
    pa.field("oversample_copy_idx", pa.float64()),   # int or null → float64 in pandas
])


# ── Streaming parquet writer ───────────────────────────────────────────────────

class SplitParquetWriter:
    """Writes records split-by-split to parquet files without holding all data in RAM."""

    SPLITS = ["train", "val", "test"]

    def __init__(self, out_paths: dict[str, str]):
        self._out_paths = out_paths
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._row_counts: dict[str, int] = {s: 0 for s in self.SPLITS}

    def write_batch(self, records: list[dict]) -> None:
        if not records:
            return
        df = pd.DataFrame(records)
        for split in self.SPLITS:
            df_s = df[df["split"] == split].reset_index(drop=True)
            if len(df_s) == 0:
                continue
            table = pa.Table.from_pandas(df_s, schema=PARQUET_SCHEMA, preserve_index=False)
            if split not in self._writers:
                self._writers[split] = pq.ParquetWriter(
                    self._out_paths[split], PARQUET_SCHEMA
                )
            self._writers[split].write_table(table)
            self._row_counts[split] += len(df_s)

    def close(self) -> dict[str, int]:
        for w in self._writers.values():
            w.close()
        return self._row_counts


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build LLM-ready dataset from EHRSHOT tasks (v6, with embedding indices, no condition events)"
    )
    parser.add_argument("--task", type=str, required=True, choices=VALID_TASKS)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(EHRSHOT_ASSETS, "llm_data_v6"))
    parser.add_argument("--max_events", type=int, default=None,
                        help="N_EVENTS threshold. Samples with more events are oversampled.")
    parser.add_argument("--buffer_size", type=int, default=2000,
                        help="Number of records to buffer in RAM before flushing to disk (default: 2000)")
    parser.add_argument("--data_dir", type=str, default=EHRSHOT_ASSETS)
    parser.add_argument("--embed_dir", type=str, default=DEFAULT_EMBED_DIR,
                        help="Directory containing event_index.parquet from BioLinkBERT extraction")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count)")
    args = parser.parse_args()

    task = args.task
    data_dir = args.data_dir
    n_workers = args.num_workers or mp.cpu_count()
    output_dir = os.path.join(args.output_dir, task)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using {n_workers} worker processes, buffer_size={args.buffer_size}")
    if args.max_events is not None:
        print(f"Oversampling config: N_EVENTS={args.max_events} "
              f"(policy in determine_num_samples.py)")
    else:
        print("max_events not set: all events included, no oversampling")

    # ── 1. Load embedding index ───────────────────────────────────────────────
    print("Loading BioLinkBERT embedding index...")
    embed_idx = load_embed_idx(args.embed_dir)

    # ── 2. Load vocab mapping ─────────────────────────────────────────────────
    print("Loading code->description mapping...")
    code_2_desc = load_code_description_map(data_dir)

    # ── 3. Load labels ────────────────────────────────────────────────────────
    print(f"Loading labels for task: {task}")
    labels_path = os.path.join(data_dir, "benchmark", task, "labeled_patients.csv")
    df_labels = pd.read_csv(labels_path)
    df_labels["prediction_time"] = pd.to_datetime(df_labels["prediction_time"])
    print(f"  Found {len(df_labels)} labeled samples across {df_labels['patient_id'].nunique()} patients")

    # ── 4. Load splits ────────────────────────────────────────────────────────
    print("Loading patient splits...")
    splits_path = os.path.join(data_dir, "splits", "person_id_map.csv")
    df_splits = pd.read_csv(splits_path)
    pid_to_split = dict(zip(df_splits["omop_person_id"], df_splits["split"]))
    df_labels["split"] = df_labels["patient_id"].map(pid_to_split)
    n_before = len(df_labels)
    df_labels = df_labels.dropna(subset=["split"])
    if len(df_labels) < n_before:
        print(f"  Warning: dropped {n_before - len(df_labels)} samples with no split assignment")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {(df_labels['split'] == split).sum()} samples")

    # ── 5. Load raw EHR data ──────────────────────────────────────────────────
    print("Loading raw EHR data (this may take a moment)...")
    ehr_path = os.path.join(data_dir, "data", "ehrshot.csv")
    # Read value/unit as raw strings to preserve the original CSV representation.
    # This is critical for embedding lookup: the embedding script also reads with
    # dtype=str, so converting to float and back would cause string mismatches
    # (e.g. "62.597999572753906" -> float -> str -> "62.59799957275391").
    df_ehr = pd.read_csv(ehr_path, low_memory=False, dtype={"value": str, "unit": str})
    if df_ehr.columns[0] == "" or df_ehr.columns[0].startswith("Unnamed"):
        df_ehr = df_ehr.drop(columns=[df_ehr.columns[0]])
    df_ehr["start"] = pd.to_datetime(df_ehr["start"])
    df_ehr = df_ehr.sort_values(["patient_id", "start"])
    print(f"  Loaded {len(df_ehr)} events across {df_ehr['patient_id'].nunique()} patients")

    print("Indexing events by patient_id...")
    grouped = dict(list(df_ehr.groupby("patient_id")))

    # ── 6. Set globals before forking ─────────────────────────────────────────
    global _GROUPED, _CODE_2_DESC, _EMBED_IDX
    _GROUPED = grouped
    _CODE_2_DESC = code_2_desc
    _EMBED_IDX = embed_idx

    # ── 7. Count total copies (fast pass, no materialisation) ─────────────────
    task_desc = TASK_DESCRIPTIONS[task]
    is_chexpert = task == "chexpert"

    print("Counting total copy-level tasks...")
    n_total_copies, n_oversampled = _count_copies(df_labels, grouped, args.max_events, task=task)
    print(f"  {len(df_labels)} original samples → {n_total_copies:,} copy-level tasks "
          f"({n_oversampled} samples oversampled)")

    # ── 8. Dispatch with streaming writes ─────────────────────────────────────
    imap_chunksize = max(1, n_total_copies // (n_workers * 20))
    out_paths = {
        split: os.path.join(output_dir, f"{split}.parquet")
        for split in ["train", "val", "test"]
    }
    writer = SplitParquetWriter(out_paths)

    total_hits = 0
    total_misses: dict[str, int] = {}
    buffer: list[dict] = []

    print(f"Building LLM-ready dataset (chunksize={imap_chunksize}, "
          f"flush every {args.buffer_size} records)...")

    copy_gen = _iter_copy_args(df_labels, grouped, args.max_events,
                               task_desc, task, is_chexpert)

    with mp.Pool(processes=n_workers) as pool:
        for record, n_hits, misses in tqdm(
            pool.imap(_process_copy, copy_gen, chunksize=imap_chunksize),
            total=n_total_copies,
            desc="Processing copies",
        ):
            buffer.append(record)
            total_hits += n_hits
            for code, count in misses.items():
                total_misses[code] = total_misses.get(code, 0) + count

            if len(buffer) >= args.buffer_size:
                writer.write_batch(buffer)
                buffer.clear()

    # flush remaining records
    writer.write_batch(buffer)
    row_counts = writer.close()

    print_coverage_report(total_hits, total_misses)

    total_rows = sum(row_counts.values())
    print(f"Total output rows: {total_rows:,} (from {len(df_labels)} original samples)")
    for split in ["train", "val", "test"]:
        if row_counts[split] == 0:
            print(f"  Warning: no samples for split '{split}'")
            continue
        print(f"  Saved {split}: {row_counts[split]:,} rows -> {out_paths[split]}")

    print("Done!")


if __name__ == "__main__":
    main()
