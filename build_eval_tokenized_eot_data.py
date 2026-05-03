#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


OMOP_TABLE_PREFIX = {
    "condition_occurrence": "Condition",
    "procedure_occurrence": "Procedure",
    "drug_exposure": "Drug",
    "measurement": "Measurement",
    "observation": "Observation",
    "visit_occurrence": "Visit",
    "device_exposure": "Device",
    "death": "Death",
    "note": "Note",
    "person": "Demographics",
}

TASK_2_DISEASE_NAME = {
    "new_hypertension": "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan": "pancreatic cancer",
    "new_celiac": "celiac disease",
    "new_lupus": "systemic lupus erythematosus",
    "new_acutemi": "acute myocardial infarction",
}
TASK_2_IDX = {task: idx for idx, task in enumerate(sorted(TASK_2_DISEASE_NAME))}

OUTPUT_SCHEMA = pa.schema([
    pa.field("patient_id", pa.int64()),
    pa.field("task_idx", pa.int16()),
    pa.field("label", pa.int8()),
    pa.field("split", pa.string()),
    pa.field("num_valid_tokens", pa.int32()),
    pa.field("num_unique_events_in_chunk", pa.int32()),
    pa.field("input_ids", pa.list_(pa.int32())),
    pa.field("attention_mask", pa.list_(pa.int8())),
    pa.field("event_ids", pa.list_(pa.int32())),
])


HERE = Path(__file__).resolve().parent
GEN_META_DIR = HERE / "01_gen_meta"
if str(GEN_META_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_META_DIR))

from build_next_event_train_parquet import format_event_row, load_event_template


_GROUPED = {}
_EVENT_TOKEN_MAP = {}
_MAX_TOKENS = 2048
_TRUNCATE_SIDE = "last"
_INCLUDE_CONDITION_OCCURRENCE = False
_TASK_IDX = 0
_SPLIT_NAME = "val"


def normalize_optional_str(x) -> str | None:
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    if pd.isna(x):
        return None
    return str(x).strip() or None


def unique_event_key(omop_table: object, code: object, value: object, unit: object) -> tuple[str, str, str, str]:
    return (
        normalize_optional_str(omop_table) or "",
        normalize_optional_str(code) or "",
        normalize_optional_str(value) or "",
        normalize_optional_str(unit) or "",
    )


def load_code_description_map(concept_csv: str) -> dict[str, str]:
    logger.info("Loading concept map from %s", concept_csv)
    concept_df = pd.read_csv(
        concept_csv,
        usecols=["concept_name", "vocabulary_id", "concept_code"],
        low_memory=False,
        dtype=str,
    ).fillna("")
    concept_df["code"] = concept_df["vocabulary_id"] + "/" + concept_df["concept_code"]
    filtered = concept_df[concept_df["code"] != concept_df["concept_name"]]
    code2desc = dict(zip(filtered["code"], filtered["concept_name"]))
    logger.info("Loaded %s code->description mappings", f"{len(code2desc):,}")
    return code2desc


def build_event_token_cache(
    df_ehr: pd.DataFrame,
    tokenizer,
    event_template,
    include_condition_occurrence: bool,
    append_eos_token_id: int,
    tokenize_batch_size: int,
) -> Dict[tuple[str, str, str, str], List[int]]:
    key_df = df_ehr[["omop_table", "event_type", "code", "description", "value", "unit"]].drop_duplicates(
        subset=["omop_table", "code", "value", "unit"],
        keep="first",
    ).reset_index(drop=True)
    if not include_condition_occurrence:
        key_df = key_df[key_df["omop_table"] != "condition_occurrence"].copy()
    logger.info("Unique events to tokenize in-memory: %s", f"{len(key_df):,}")

    unique_texts: List[str] = []
    unique_keys: List[tuple[str, str, str, str]] = []
    dropped = 0
    for _, row in tqdm(key_df.iterrows(), total=len(key_df), desc="render unique events", dynamic_ncols=True):
        text = format_event_row(row, include_condition_occurrence, event_template)
        if not text:
            dropped += 1
            continue
        unique_keys.append(unique_event_key(row["omop_table"], row["code"], row["value"], row["unit"]))
        unique_texts.append(text)
    logger.info("Renderable unique events: %s (dropped=%s)", f"{len(unique_keys):,}", f"{dropped:,}")

    event_token_map: Dict[tuple[str, str, str, str], List[int]] = {}
    for i in tqdm(range(0, len(unique_texts), tokenize_batch_size), desc="tokenize unique events", dynamic_ncols=True):
        batch_texts = unique_texts[i : i + tokenize_batch_size]
        batch_keys = unique_keys[i : i + tokenize_batch_size]
        enc = tokenizer(batch_texts, add_special_tokens=False, return_attention_mask=False)
        for key, token_ids in zip(batch_keys, enc["input_ids"]):
            ids = [int(x) for x in token_ids]
            ids.append(int(append_eos_token_id))
            event_token_map[key] = ids
    logger.info("Tokenized unique event cache size: %s", f"{len(event_token_map):,}")
    return event_token_map


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


def _truncate_stream(token_stream: List[int], event_stream: List[int]) -> tuple[List[int], List[int]]:
    if len(token_stream) <= _MAX_TOKENS:
        return token_stream, event_stream
    if _TRUNCATE_SIDE == "last":
        return token_stream[-_MAX_TOKENS:], event_stream[-_MAX_TOKENS:]
    return token_stream[:_MAX_TOKENS], event_stream[:_MAX_TOKENS]


def _process_eval_sample(args):
    pid, pred_time, label_val = args
    patient_events = _GROUPED.get(pid)
    if patient_events is None:
        return None

    events_df = patient_events.loc[patient_events["start"] < pred_time]
    if len(events_df) == 0:
        return None

    token_stream: List[int] = []
    event_stream: List[int] = []
    event_idx = 0
    for row in events_df.itertuples(index=False):
        if (not _INCLUDE_CONDITION_OCCURRENCE) and row.omop_table == "condition_occurrence":
            continue
        key = unique_event_key(row.omop_table, row.code, row.value, row.unit)
        token_ids = _EVENT_TOKEN_MAP.get(key)
        if token_ids is None:
            continue
        token_stream.extend(token_ids)
        event_stream.extend([event_idx] * len(token_ids))
        event_idx += 1

    if not token_stream:
        return None

    token_stream, event_stream = _truncate_stream(token_stream, event_stream)
    n = len(token_stream)
    input_ids = list(token_stream)
    attention_mask = [1] * n
    event_ids = list(event_stream)

    return {
        "patient_id": int(pid),
        "task_idx": int(_TASK_IDX),
        "label": int(1 if _is_positive(label_val) else 0),
        "split": _SPLIT_NAME,
        "num_valid_tokens": int(n),
        "num_unique_events_in_chunk": int(len(set(event_stream))),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "event_ids": event_ids,
    }


def _flush(writer, rows: List[dict]):
    table = pa.table(
        {
            "patient_id": [r["patient_id"] for r in rows],
            "task_idx": [r["task_idx"] for r in rows],
            "label": [r["label"] for r in rows],
            "split": [r["split"] for r in rows],
            "num_valid_tokens": [r["num_valid_tokens"] for r in rows],
            "num_unique_events_in_chunk": [r["num_unique_events_in_chunk"] for r in rows],
            "input_ids": [r["input_ids"] for r in rows],
            "attention_mask": [r["attention_mask"] for r in rows],
            "event_ids": [r["event_ids"] for r in rows],
        },
        schema=OUTPUT_SCHEMA,
    )
    writer.write_table(table)


def print_example_row(row: dict, tokenizer, preview_tokens: int):
    preview_n = min(preview_tokens, int(row["num_valid_tokens"]))
    logger.info("Example sample")
    logger.info(
        "  patient_id=%s task_idx=%s label=%s split=%s num_valid_tokens=%s num_unique_events_in_chunk=%s",
        row["patient_id"], row["task_idx"], row["label"], row["split"], row["num_valid_tokens"], row["num_unique_events_in_chunk"],
    )
    logger.info("  first_%d_input_ids=%s", preview_n, row["input_ids"][:preview_n])
    logger.info("  first_%d_attention_mask=%s", preview_n, row["attention_mask"][:preview_n])
    logger.info("  first_%d_event_ids=%s", preview_n, row["event_ids"][:preview_n])
    logger.info("  decoded_preview=%s", json.dumps(tokenizer.decode(row["input_ids"][:preview_n], skip_special_tokens=False)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Build tokenized patient-level evaluation parquet directly from raw EHRSHOT tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data_dir", default="data/EHRSHOT_ASSETS")
    p.add_argument("--ehrshot_csv", default=None)
    p.add_argument("--concept_csv", default=None)
    p.add_argument("--template_path", default="01_gen_meta/templates/biolinkbert_event.j2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=sorted(TASK_2_DISEASE_NAME.keys()), choices=sorted(TASK_2_DISEASE_NAME.keys()))
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--include_condition_occurrence", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--truncate_side", default="last", choices=["first", "last"])
    p.add_argument("--tokenize_batch_size", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--buffer_size", type=int, default=2048)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--preview_tokens", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    ehrshot_csv = Path(args.ehrshot_csv) if args.ehrshot_csv else data_dir / "data" / "ehrshot.csv"
    concept_csv = Path(args.concept_csv) if args.concept_csv else data_dir / "femr" / "logs" / "omop_dir" / "concept.csv"

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(f"Tokenizer {args.model_name} has no pad_token_id; needed as end-of-event token.")
    logger.info("Using end-of-event token: %r (id=%d)", tokenizer.pad_token, pad_token_id)

    logger.info("Loading and sorting raw EHR CSV: %s", ehrshot_csv)
    df_ehr = pd.read_csv(
        ehrshot_csv,
        usecols=["patient_id", "start", "omop_table", "code", "value", "unit"],
        low_memory=False,
        dtype={"value": str, "unit": str},
    )
    if df_ehr.columns[0] == "" or str(df_ehr.columns[0]).startswith("Unnamed"):
        df_ehr = df_ehr.drop(columns=[df_ehr.columns[0]])
    df_ehr["start"] = pd.to_datetime(df_ehr["start"])
    df_ehr = df_ehr.sort_values(["patient_id", "start"], ascending=[True, True]).reset_index(drop=True)
    logger.info("Loaded %s events across %s patients", f"{len(df_ehr):,}", f"{df_ehr['patient_id'].nunique():,}")

    if args.max_patients is not None:
        keep_ids = df_ehr["patient_id"].drop_duplicates().tolist()[: args.max_patients]
        df_ehr = df_ehr[df_ehr["patient_id"].isin(keep_ids)].copy()
        logger.info("Restricted raw EHR to first %s patients", f"{len(keep_ids):,}")

    code2desc = load_code_description_map(str(concept_csv))
    for col in ["omop_table", "code", "value", "unit"]:
        df_ehr[col] = df_ehr[col].fillna("").astype(str).str.strip()
    df_ehr["description"] = df_ehr["code"].map(lambda code: normalize_optional_str(code2desc.get(code, "")) or "")
    df_ehr["event_type"] = df_ehr["omop_table"].map(lambda x: OMOP_TABLE_PREFIX.get(x, x))

    event_template = load_event_template(args.template_path)
    event_token_map = build_event_token_cache(
        df_ehr=df_ehr,
        tokenizer=tokenizer,
        event_template=event_template,
        include_condition_occurrence=args.include_condition_occurrence,
        append_eos_token_id=pad_token_id,
        tokenize_batch_size=args.tokenize_batch_size,
    )

    logger.info("Indexing raw EHR by patient_id")
    grouped = dict(list(df_ehr.groupby("patient_id", sort=False)))

    logger.info("Loading patient splits")
    splits_path = data_dir / "splits" / "person_id_map.csv"
    df_splits = pd.read_csv(splits_path)
    pid_to_split = dict(zip(df_splits["omop_person_id"], df_splits["split"]))

    global _GROUPED, _EVENT_TOKEN_MAP, _MAX_TOKENS, _TRUNCATE_SIDE, _INCLUDE_CONDITION_OCCURRENCE, _TASK_IDX, _SPLIT_NAME
    _GROUPED = grouped
    _EVENT_TOKEN_MAP = event_token_map
    _MAX_TOKENS = int(args.max_tokens)
    _TRUNCATE_SIDE = args.truncate_side
    _INCLUDE_CONDITION_OCCURRENCE = args.include_condition_occurrence

    metadata = {
        "model_name": args.model_name,
        "ehrshot_csv": str(ehrshot_csv),
        "concept_csv": str(concept_csv),
        "template_path": args.template_path,
        "tasks": args.tasks,
        "splits": args.splits,
        "include_condition_occurrence": bool(args.include_condition_occurrence),
        "max_tokens": int(args.max_tokens),
        "truncate_side": args.truncate_side,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(pad_token_id),
    }

    example_row = None
    for task in args.tasks:
        logger.info("Loading labels for task: %s", task)
        labels_path = data_dir / "benchmark" / task / "labeled_patients.csv"
        df_labels = pd.read_csv(labels_path)
        df_labels["prediction_time"] = pd.to_datetime(df_labels["prediction_time"])
        df_labels["split"] = df_labels["patient_id"].map(pid_to_split)
        df_labels = df_labels.dropna(subset=["split"])
        df_labels = df_labels[df_labels["split"].isin(args.splits)]
        logger.info("  %s labeled samples in splits %s", f"{len(df_labels):,}", args.splits)

        task_out_dir = output_dir / task
        task_out_dir.mkdir(parents=True, exist_ok=True)
        _TASK_IDX = TASK_2_IDX[task]

        for split in args.splits:
            df_split = df_labels[df_labels["split"] == split]
            if len(df_split) == 0:
                logger.info("  [%s:%s] No samples - skipping", task, split)
                continue

            logger.info("  [%s:%s] Processing %s samples", task, split, f"{len(df_split):,}")
            _SPLIT_NAME = split
            worker_args = [
                (int(row.patient_id), row.prediction_time, row.value)
                for row in df_split.itertuples(index=False)
            ]
            out_path = task_out_dir / f"{split}.parquet"
            writer = pq.ParquetWriter(out_path, OUTPUT_SCHEMA)
            buffer = []
            n_written = 0
            n_skipped = 0

            chunksize = max(1, len(worker_args) // max(args.num_workers * 20, 1)) if len(worker_args) > 0 else 1
            with mp.Pool(processes=args.num_workers) as pool:
                for result in tqdm(
                    pool.imap(_process_eval_sample, worker_args, chunksize=chunksize),
                    total=len(worker_args),
                    desc=f"{task}:{split}",
                    dynamic_ncols=True,
                ):
                    if result is None:
                        n_skipped += 1
                        continue
                    if example_row is None:
                        example_row = result
                    buffer.append(result)
                    if len(buffer) >= args.buffer_size:
                        _flush(writer, buffer)
                        n_written += len(buffer)
                        buffer.clear()

            if buffer:
                _flush(writer, buffer)
                n_written += len(buffer)
                buffer.clear()
            writer.close()

            if n_written > 0:
                label_df = pd.read_parquet(out_path, columns=["label"])
                pos = int((label_df["label"] == 1).sum())
            else:
                pos = 0
            neg = n_written - pos
            logger.info(
                "    -> %s rows (%s pos / %s neg), %s skipped -> %s",
                f"{n_written:,}", f"{pos:,}", f"{neg:,}", f"{n_skipped:,}", out_path,
            )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    logger.info("Metadata -> %s", metadata_path)
    if example_row is not None:
        print_example_row(example_row, tokenizer, args.preview_tokens)
    logger.info("Done.")


if __name__ == "__main__":
    main()
