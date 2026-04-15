#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment, StrictUndefined
from tqdm import tqdm
from transformers import AutoTokenizer

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
    pa.field("block_idx", pa.int32()),
    pa.field("num_tokens", pa.int32()),
    pa.field("input_ids", pa.list_(pa.int32())),
])


_PATIENT_GROUPS: dict[int, pd.DataFrame] = {}
_INCLUDE_CONDITION_OCCURRENCE = False
_EVENT_TOKEN_MAP: dict[tuple[str, str, str, str], list[int]] = {}
_EOS_TOKEN_ID = None
_BLOCK_SIZE = None
_EVENT_TEMPLATE = None


def normalize_optional_str(x) -> str | None:
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    if pd.isna(x):
        return None
    return str(x).strip() or None


def load_event_template(template_path: str):
    template_text = Path(template_path).read_text()
    env = Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return env.from_string(template_text)


def unique_event_key(omop_table: object, code: object, value: object, unit: object) -> tuple[str, str, str, str]:
    return (
        normalize_optional_str(omop_table) or "",
        normalize_optional_str(code) or "",
        normalize_optional_str(value) or "",
        normalize_optional_str(unit) or "",
    )


def format_event_row(ev: pd.Series, include_condition_occurrence: bool) -> str | None:
    if (not include_condition_occurrence) and ev["omop_table"] == "condition_occurrence":
        return None

    code = str(ev["code"]).strip()
    event_type = normalize_optional_str(ev.get("event_type")) or OMOP_TABLE_PREFIX.get(ev["omop_table"], ev["omop_table"])
    description = normalize_optional_str(ev.get("description")) or ""
    value = normalize_optional_str(ev["value"])
    unit = normalize_optional_str(ev["unit"])
    rendered = _EVENT_TEMPLATE.render(
        event_type=event_type or "",
        description=description,
        code=code,
        value=value or "",
        unit=unit or "",
    ).strip()
    return rendered or None


def chunk_list(xs: list[int], block_size: int) -> list[list[int]]:
    n_full = len(xs) // block_size
    return [xs[i * block_size : (i + 1) * block_size] for i in range(n_full)]


def _init_worker(
    patient_groups: dict[int, pd.DataFrame],
    event_token_map: dict[tuple[str, str, str, str], list[int]],
    include_condition_occurrence: bool,
    eos_token_id: int,
    block_size: int,
):
    global _PATIENT_GROUPS, _EVENT_TOKEN_MAP, _INCLUDE_CONDITION_OCCURRENCE, _EOS_TOKEN_ID, _BLOCK_SIZE
    _PATIENT_GROUPS = patient_groups
    _EVENT_TOKEN_MAP = event_token_map
    _INCLUDE_CONDITION_OCCURRENCE = include_condition_occurrence
    _EOS_TOKEN_ID = eos_token_id
    _BLOCK_SIZE = block_size


def _process_patient(patient_id: int) -> tuple[list[dict], int, int]:
    events_df = _PATIENT_GROUPS[int(patient_id)]
    patient_tokens: list[int] = []
    valid_events = 0
    for _, ev in events_df.iterrows():
        if (not _INCLUDE_CONDITION_OCCURRENCE) and ev["omop_table"] == "condition_occurrence":
            continue

        key = unique_event_key(ev["omop_table"], ev["code"], ev["value"], ev["unit"])
        event_ids = _EVENT_TOKEN_MAP.get(key)
        if event_ids is None:
            raise KeyError(
                f"Missing tokenized event for key={key!r} "
                f"(patient_id={patient_id}, omop_table={ev['omop_table']!r}, code={ev['code']!r}, "
                f"value={ev['value']!r}, unit={ev['unit']!r})."
            )
        patient_tokens.extend(int(x) for x in event_ids)
        patient_tokens.append(int(_EOS_TOKEN_ID))
        valid_events += 1

    if not patient_tokens:
        return [], valid_events, 0

    blocks = chunk_list(patient_tokens, _BLOCK_SIZE)
    rows = [
        {
            "patient_id": int(patient_id),
            "block_idx": int(block_idx),
            "num_tokens": _BLOCK_SIZE,
            "input_ids": [int(x) for x in block],
        }
        for block_idx, block in enumerate(blocks)
    ]
    return rows, valid_events, len(patient_tokens)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build fixed-length token blocks for CPT from raw EHR CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tokenizer_name", default="Qwen/Qwen3-0.6B-Base", help="Tokenizer name or local path, e.g. Qwen/Qwen3-0.6B-Base")
    p.add_argument("--data_dir", default="EHRSHOT_ASSETS", help="EHRSHOT asset root containing data/ehrshot.csv and models/clmbr/*")
    p.add_argument("--ehr_csv", default=None, help="Override raw EHR CSV path; defaults to <data_dir>/data/ehrshot.csv")
    p.add_argument("--unique_event_parquet", default="EHRSHOT_ASSETS/features/unique_event_rows_plus_cond_fast.parquet", help="Unique-event parquet used to build the token cache")
    p.add_argument("--output_path", default="EHRSHOT_ASSETS/cpt_blocks/qwen3_0.6b_block2048.parquet", help="Output parquet path")
    p.add_argument("--block_size", type=int, default=2048, help="Fixed token length N for each row/block")
    p.add_argument("--template_path", default="encode_events/event_to_text.j2", help="Jinja2 template used to render each event")
    p.add_argument("--include_condition_occurrence", action="store_true", help="Keep condition_occurrence rows")
    p.add_argument("--max_patients", type=int, default=None, help="Optional cap for quick experiments")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--num_workers", type=int, default=None, help="Patient-level worker processes (default: cpu_count)")
    p.add_argument("--tokenize_batch_size", type=int, default=4096, help="Batch size used when tokenizing unique events")
    return p.parse_args()


def main():
    global _EVENT_TEMPLATE
    args = parse_args()
    data_dir = Path(args.data_dir)
    ehr_csv = Path(args.ehr_csv) if args.ehr_csv else data_dir / "data" / "ehrshot.csv"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, local_files_only=args.local_files_only)
    eos_token = tokenizer.pad_token
    eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        eos_token = tokenizer.pad_token
        eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has neither eos_token_id nor pad_token_id; cannot append end-of-event token.")
    print(f"Using end-of-event token: {eos_token!r} (id={eos_token_id})")

    event_template = load_event_template(args.template_path)
    _EVENT_TEMPLATE = event_template

    print(f"Reading raw EHR CSV: {ehr_csv}")
    chunks = []
    for chunk in tqdm(
        pd.read_csv(ehr_csv, low_memory=False, dtype={"value": str, "unit": str}, chunksize=500_000),
        desc="reading CSV",
        unit="chunk",
        dynamic_ncols=True,
    ):
        if chunk.columns[0] == "" or str(chunk.columns[0]).startswith("Unnamed"):
            chunk = chunk.drop(columns=[chunk.columns[0]])
        chunks.append(chunk)
    df_ehr = pd.concat(chunks, ignore_index=True)
    del chunks
    df_ehr["start"] = pd.to_datetime(df_ehr["start"])
    df_ehr = df_ehr.sort_values(["patient_id", "start"], ascending=[True, False]).reset_index(drop=True)
    print(f"Loaded {len(df_ehr):,} raw events across {df_ehr['patient_id'].nunique():,} patients")

    patient_ids = df_ehr["patient_id"].drop_duplicates().tolist()
    if args.max_patients is not None:
        patient_ids = patient_ids[: args.max_patients]
        df_ehr = df_ehr[df_ehr["patient_id"].isin(patient_ids)]
        print(f"Restricted to first {len(patient_ids):,} patients for this run")

    patient_groups = {int(pid): pdf for pid, pdf in df_ehr.groupby("patient_id", sort=False)}

    print(f"Loading unique event parquet: {args.unique_event_parquet}")
    key_df = pd.read_parquet(args.unique_event_parquet)
    for col in ["omop_table", "code", "value", "unit", "event_type", "description"]:
        if col in key_df.columns:
            key_df[col] = key_df[col].fillna("").astype(str).str.strip()
    if not args.include_condition_occurrence:
        key_df = key_df[key_df["omop_table"] != "condition_occurrence"].copy()
    print(f"Unique events to tokenize: {len(key_df):,}")

    sample_texts = []
    for _, row in key_df.head(5).iterrows():
        t = format_event_row(row, args.include_condition_occurrence)
        if t:
            sample_texts.append(t)
    print("\n--- Sample unique-event strings ---")
    for t in sample_texts:
        print(" ", t)
    print("---\n")

    unique_texts: list[str] = []
    unique_keys: list[tuple[str, str, str, str]] = []
    for _, row in tqdm(key_df.iterrows(), total=len(key_df), desc="render unique events", dynamic_ncols=True):
        text = format_event_row(row, args.include_condition_occurrence)
        if not text:
            continue
        unique_keys.append(unique_event_key(row["omop_table"], row["code"], row["value"], row["unit"]))
        unique_texts.append(text)

    print("Tokenizing unique events once ...")
    event_token_map: dict[tuple[str, str, str, str], list[int]] = {}
    for i in tqdm(range(0, len(unique_texts), args.tokenize_batch_size), desc="tokenize unique events", dynamic_ncols=True):
        batch_texts = unique_texts[i : i + args.tokenize_batch_size]
        batch_keys = unique_keys[i : i + args.tokenize_batch_size]
        enc = tokenizer(batch_texts, add_special_tokens=False)
        for key, token_ids in zip(batch_keys, enc["input_ids"]):
            event_token_map[key] = [int(x) for x in token_ids]
    print(f"Tokenized unique event cache size: {len(event_token_map):,}")

    rows: list[dict] = []
    total_event_count = 0
    total_token_count = 0
    num_workers = args.num_workers or mp.cpu_count()
    print(f"Processing {len(patient_groups):,} patients with {num_workers} worker(s)")
    chunksize = max(1, len(patient_groups) // max(num_workers * 8, 1))
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(
            patient_groups,
            event_token_map,
            args.include_condition_occurrence,
            int(eos_token_id),
            args.block_size,
        ),
    ) as pool:
        for patient_rows, event_count, token_count in tqdm(
            pool.imap(_process_patient, patient_groups.keys(), chunksize=chunksize),
            total=len(patient_groups),
            desc="patients",
            dynamic_ncols=True,
        ):
            if patient_rows:
                rows.extend(patient_rows)
            total_event_count += event_count
            total_token_count += token_count

    if not rows:
        raise ValueError("No token blocks were created. Check input paths and filtering options.")

    print(f"Formatted {total_event_count:,} events into {total_token_count:,} tokens")
    print(f"Writing {len(rows):,} blocks to {output_path}")
    table = pa.table(
        {
            "patient_id": pa.array([r["patient_id"] for r in rows], type=pa.int64()),
            "block_idx": pa.array([r["block_idx"] for r in rows], type=pa.int32()),
            "num_tokens": pa.array([r["num_tokens"] for r in rows], type=pa.int32()),
            "input_ids": pa.array([r["input_ids"] for r in rows], type=pa.list_(pa.int32())),
        },
        schema=OUTPUT_SCHEMA,
    )
    pq.write_table(table, output_path)

    meta_path = output_path.with_suffix(".json")
    meta = {
        "tokenizer_name": args.tokenizer_name,
        "template_path": args.template_path,
        "unique_event_parquet": args.unique_event_parquet,
        "tokenize_batch_size": args.tokenize_batch_size,
        "eos_token": eos_token,
        "eos_token_id": int(eos_token_id),
        "block_size": args.block_size,
        "include_condition_occurrence": bool(args.include_condition_occurrence),
        "num_unique_events": len(event_token_map),
        "num_blocks": len(rows),
        "num_patients": len({r["patient_id"] for r in rows}),
        "num_events": total_event_count,
        "num_tokens": total_token_count,
        "avg_blocks_per_patient": len(rows) / max(len({r["patient_id"] for r in rows}), 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
