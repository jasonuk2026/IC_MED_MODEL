#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment, StrictUndefined
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


OUTPUT_SCHEMA = pa.schema([
    pa.field("patient_id", pa.int64()),
    pa.field("chunk_idx", pa.int32()),
    pa.field("num_valid_tokens", pa.int32()),
    pa.field("num_unique_events_in_chunk", pa.int32()),
    pa.field("input_ids", pa.list_(pa.int32())),
    pa.field("attention_mask", pa.list_(pa.int8())),
    pa.field("event_ids", pa.list_(pa.int32())),
    pa.field("labels", pa.list_(pa.int32())),
])


_DF_EHR: pd.DataFrame | None = None
_EVENT_TOKEN_MAP: Dict[tuple[str, str, str, str], List[int]] = {}
_INCLUDE_CONDITION_OCCURRENCE = False
_SEQ_LEN = 2048
_PAD_TOKEN_ID = 0


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


def load_event_template(template_path: str):
    template_text = Path(template_path).read_text()
    env = Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return env.from_string(template_text)


def format_event_row(ev: pd.Series, include_condition_occurrence: bool, event_template) -> str | None:
    if (not include_condition_occurrence) and ev["omop_table"] == "condition_occurrence":
        return None

    code = str(ev["code"]).strip()
    event_type = normalize_optional_str(ev.get("event_type")) or OMOP_TABLE_PREFIX.get(ev["omop_table"], ev["omop_table"])
    description = normalize_optional_str(ev.get("description")) or ""
    value = normalize_optional_str(ev["value"])
    unit = normalize_optional_str(ev["unit"])
    rendered = event_template.render(
        event_type=event_type or "",
        description=description,
        code=code,
        value=value or "",
        unit=unit or "",
    ).strip()
    return rendered or None


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
    logger.info("Unique events to tokenize: %s", f"{len(key_df):,}")

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


def build_patient_ranges(patient_ids: Sequence[int]) -> List[tuple[int, int, int]]:
    ranges: List[tuple[int, int, int]] = []
    if len(patient_ids) == 0:
        return ranges
    start = 0
    current = int(patient_ids[0])
    for idx in range(1, len(patient_ids)):
        pid = int(patient_ids[idx])
        if pid != current:
            ranges.append((current, start, idx))
            current = pid
            start = idx
    ranges.append((current, start, len(patient_ids)))
    return ranges


def iter_blocks(xs: Sequence[tuple[int, int, int]], block_size: int) -> Iterable[List[tuple[int, int, int]]]:
    for i in range(0, len(xs), block_size):
        yield list(xs[i : i + block_size])


def init_thread_globals(
    *,
    df_ehr: pd.DataFrame,
    event_token_map: Dict[tuple[str, str, str, str], List[int]],
    include_condition_occurrence: bool,
    seq_len: int,
    pad_token_id: int,
):
    global _DF_EHR, _EVENT_TOKEN_MAP, _INCLUDE_CONDITION_OCCURRENCE, _SEQ_LEN, _PAD_TOKEN_ID
    _DF_EHR = df_ehr
    _EVENT_TOKEN_MAP = event_token_map
    _INCLUDE_CONDITION_OCCURRENCE = include_condition_occurrence
    _SEQ_LEN = seq_len
    _PAD_TOKEN_ID = int(pad_token_id)


def process_patient_block(block: List[tuple[int, int, int]]) -> dict:
    assert _DF_EHR is not None
    rows: List[dict] = []
    patient_count = 0
    event_count = 0

    for patient_id, start_idx, end_idx in block:
        patient_count += 1
        pdf = _DF_EHR.iloc[start_idx:end_idx]
        token_stream: List[int] = []
        event_stream: List[int] = []
        event_idx = 0

        for row in pdf.itertuples(index=False):
            if (not _INCLUDE_CONDITION_OCCURRENCE) and row.omop_table == "condition_occurrence":
                continue
            key = unique_event_key(row.omop_table, row.code, row.value, row.unit)
            token_ids = _EVENT_TOKEN_MAP.get(key)
            if token_ids is None:
                continue
            token_stream.extend(token_ids)
            event_stream.extend([event_idx] * len(token_ids))
            event_idx += 1
            event_count += 1

        if not token_stream:
            continue

        for chunk_idx, token_start in enumerate(range(0, len(token_stream), _SEQ_LEN)):
            chunk_ids = token_stream[token_start : token_start + _SEQ_LEN]
            chunk_events = event_stream[token_start : token_start + _SEQ_LEN]
            n = len(chunk_ids)

            input_ids = [_PAD_TOKEN_ID] * _SEQ_LEN
            attention_mask = [0] * _SEQ_LEN
            event_ids = [-1] * _SEQ_LEN
            labels = [-100] * _SEQ_LEN

            input_ids[:n] = chunk_ids
            attention_mask[:n] = [1] * n
            event_ids[:n] = chunk_events
            labels[:n] = chunk_ids

            rows.append(
                {
                    "patient_id": int(patient_id),
                    "chunk_idx": int(chunk_idx),
                    "num_valid_tokens": int(n),
                    "num_unique_events_in_chunk": int(len(set(chunk_events))) if chunk_events else 0,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "event_ids": event_ids,
                    "labels": labels,
                }
            )

    return {
        "rows": rows,
        "patients": patient_count,
        "events": event_count,
    }


def rows_to_table(rows: List[dict]) -> pa.Table:
    return pa.table(
        {
            "patient_id": [r["patient_id"] for r in rows],
            "chunk_idx": [r["chunk_idx"] for r in rows],
            "num_valid_tokens": [r["num_valid_tokens"] for r in rows],
            "num_unique_events_in_chunk": [r["num_unique_events_in_chunk"] for r in rows],
            "input_ids": [r["input_ids"] for r in rows],
            "attention_mask": [r["attention_mask"] for r in rows],
            "event_ids": [r["event_ids"] for r in rows],
            "labels": [r["labels"] for r in rows],
        },
        schema=OUTPUT_SCHEMA,
    )


def print_example_row(example_row: dict, tokenizer, preview_tokens: int):
    n = int(example_row["num_valid_tokens"])
    preview_n = min(preview_tokens, n)
    preview_ids = example_row["input_ids"][:preview_n]
    preview_mask = example_row["attention_mask"][:preview_n]
    preview_event_ids = example_row["event_ids"][:preview_n]
    preview_labels = example_row["labels"][:preview_n]
    decoded_preview = tokenizer.decode(preview_ids, skip_special_tokens=False)

    logger.info("Example sample")
    logger.info(
        "  patient_id=%s chunk_idx=%s num_valid_tokens=%s num_unique_events_in_chunk=%s",
        example_row["patient_id"],
        example_row["chunk_idx"],
        example_row["num_valid_tokens"],
        example_row["num_unique_events_in_chunk"],
    )
    logger.info("  first_%d_input_ids=%s", preview_n, preview_ids)
    logger.info("  first_%d_attention_mask=%s", preview_n, preview_mask)
    logger.info("  first_%d_event_ids=%s", preview_n, preview_event_ids)
    logger.info("  first_%d_labels=%s", preview_n, preview_labels)
    logger.info("  decoded_preview=%s", json.dumps(decoded_preview))


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline builder for event-EOT CPT training parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data_dir", default="data/EHRSHOT_ASSETS")
    p.add_argument("--ehrshot_csv", default=None)
    p.add_argument("--concept_csv", default=None)
    p.add_argument("--template_path", default="01_gen_meta/templates/biolinkbert_event.j2")
    p.add_argument("--output_path", required=True)
    p.add_argument("--metadata_path", default=None)
    p.add_argument("--include_condition_occurrence", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--tokenize_batch_size", type=int, default=4096)
    p.add_argument("--num_threads", type=int, default=16)
    p.add_argument("--patients_per_task", type=int, default=64)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--preview_tokens", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

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
        logger.info("Restricted to first %s patients", f"{len(keep_ids):,}")

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

    patient_ranges = build_patient_ranges(df_ehr["patient_id"].tolist())
    logger.info("Patient ranges prepared: %s", f"{len(patient_ranges):,}")
    init_thread_globals(
        df_ehr=df_ehr,
        event_token_map=event_token_map,
        include_condition_occurrence=args.include_condition_occurrence,
        seq_len=args.seq_len,
        pad_token_id=pad_token_id,
    )

    sample_count = 0
    patient_count = 0
    event_count = 0
    example_row = None
    blocks = list(iter_blocks(patient_ranges, args.patients_per_task))
    logger.info(
        "Building offline chunks with %d thread(s), %d patient block(s), seq_len=%d",
        args.num_threads,
        len(blocks),
        args.seq_len,
    )

    with pq.ParquetWriter(output_path, OUTPUT_SCHEMA) as writer:
        with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
            futures = [ex.submit(process_patient_block, block) for block in blocks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="patient blocks", dynamic_ncols=True):
                result = fut.result()
                rows = result["rows"]
                patient_count += int(result["patients"])
                event_count += int(result["events"])
                if not rows:
                    continue
                if example_row is None:
                    example_row = rows[0]
                writer.write_table(rows_to_table(rows))
                sample_count += len(rows)

    metadata = {
        "model_name": args.model_name,
        "ehrshot_csv": str(ehrshot_csv),
        "concept_csv": str(concept_csv),
        "template_path": args.template_path,
        "include_condition_occurrence": bool(args.include_condition_occurrence),
        "seq_len": int(args.seq_len),
        "tokenize_batch_size": int(args.tokenize_batch_size),
        "num_threads": int(args.num_threads),
        "patients_per_task": int(args.patients_per_task),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(pad_token_id),
        "samples": int(sample_count),
        "patients": int(patient_count),
        "events": int(event_count),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    logger.info("Writing parquet complete: %s", output_path)
    logger.info(
        "Summary: samples=%s patients=%s events=%s",
        f"{sample_count:,}",
        f"{patient_count:,}",
        f"{event_count:,}",
    )
    logger.info("Metadata -> %s", metadata_path)
    if example_row is not None:
        print_example_row(example_row, tokenizer, args.preview_tokens)


if __name__ == "__main__":
    main()
