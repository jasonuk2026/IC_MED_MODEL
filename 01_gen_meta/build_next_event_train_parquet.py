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
    pa.field("chunk_idx", pa.int32()),
    pa.field("num_events", pa.int32()),
    pa.field("event_token_ids", pa.list_(pa.list_(pa.int32()))),
])

_PATIENT_GROUPS: dict[int, pd.DataFrame] = {}
_INCLUDE_CONDITION_OCCURRENCE = False
_EVENT_TOKEN_MAP: dict[tuple[str, str, str, str], list[int]] = {}
_EOS_TOKEN_ID = None
_EVENTS_PER_ROW = None


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


def _init_worker(
    patient_groups: dict[int, pd.DataFrame],
    event_token_map: dict[tuple[str, str, str, str], list[int]],
    include_condition_occurrence: bool,
    eos_token_id: int,
    events_per_row: int,
):
    global _PATIENT_GROUPS, _EVENT_TOKEN_MAP, _INCLUDE_CONDITION_OCCURRENCE, _EOS_TOKEN_ID, _EVENTS_PER_ROW
    _PATIENT_GROUPS = patient_groups
    _EVENT_TOKEN_MAP = event_token_map
    _INCLUDE_CONDITION_OCCURRENCE = include_condition_occurrence
    _EOS_TOKEN_ID = eos_token_id
    _EVENTS_PER_ROW = events_per_row


def chunk_event_lists(xs: list[list[int]], chunk_size: int) -> list[list[list[int]]]:
    n_full = len(xs) // chunk_size
    return [xs[i * chunk_size : (i + 1) * chunk_size] for i in range(n_full)]


def _process_patient(patient_id: int) -> list[dict]:
    events_df = _PATIENT_GROUPS[int(patient_id)]
    event_token_ids: list[list[int]] = []
    for _, ev in events_df.iterrows():
        if (not _INCLUDE_CONDITION_OCCURRENCE) and ev["omop_table"] == "condition_occurrence":
            continue
        key = unique_event_key(ev["omop_table"], ev["code"], ev["value"], ev["unit"])
        token_ids = _EVENT_TOKEN_MAP.get(key)
        if token_ids is None:
            raise KeyError(
                f"Missing tokenized event for key={key!r} "
                f"(patient_id={patient_id}, omop_table={ev['omop_table']!r}, code={ev['code']!r}, "
                f"value={ev['value']!r}, unit={ev['unit']!r})."
            )
        ids = [int(x) for x in token_ids] + [int(_EOS_TOKEN_ID)]
        event_token_ids.append(ids)

    if len(event_token_ids) < _EVENTS_PER_ROW:
        return []

    chunks = chunk_event_lists(event_token_ids, _EVENTS_PER_ROW)
    return [
        {
            "patient_id": int(patient_id),
            "chunk_idx": int(chunk_idx),
            "num_events": int(len(chunk)),
            "event_token_ids": chunk,
        }
        for chunk_idx, chunk in enumerate(chunks)
    ]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build fixed-event-count patient chunks for next-event prediction training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tokenizer_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data_dir", default="data/EHRSHOT_ASSETS")
    p.add_argument("--ehr_csv", default=None)
    p.add_argument("--unique_event_parquet", default="data/01_outputs/unique_events.parquet")
    p.add_argument("--output_path", default="data/next_event_train/qwen3_0.6b_patient_events.parquet")
    p.add_argument("--template_path", default="encode_events/event_to_text.j2")
    p.add_argument("--include_condition_occurrence", action="store_true")
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--events_per_row", type=int, default=1024)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--tokenize_batch_size", type=int, default=4096)
    return p.parse_args()


def main():
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
        raise ValueError("Tokenizer has no pad_token_id; cannot append end-of-event token.")
    print(f"Using end-of-event token: {eos_token!r} (id={eos_token_id})")

    event_template = load_event_template(args.template_path)

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
    df_ehr = df_ehr.sort_values(["patient_id", "start"], ascending=[True, True]).reset_index(drop=True)
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

    unique_texts: list[str] = []
    unique_keys: list[tuple[str, str, str, str]] = []
    for _, row in tqdm(key_df.iterrows(), total=len(key_df), desc="render unique events", dynamic_ncols=True):
        text = format_event_row(row, args.include_condition_occurrence, event_template)
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
            args.events_per_row,
        ),
    ) as pool:
        for patient_rows in tqdm(
            pool.imap(_process_patient, patient_groups.keys(), chunksize=chunksize),
            total=len(patient_groups),
            desc="patients",
            dynamic_ncols=True,
        ):
            if patient_rows:
                rows.extend(patient_rows)

    if not rows:
        raise ValueError("No patient rows were created. Check input paths and filtering options.")

    print(f"Writing {len(rows):,} chunk rows to {output_path}")
    table = pa.table(
        {
            "patient_id": pa.array([r["patient_id"] for r in rows], type=pa.int64()),
            "chunk_idx": pa.array([r["chunk_idx"] for r in rows], type=pa.int32()),
            "num_events": pa.array([r["num_events"] for r in rows], type=pa.int32()),
            "event_token_ids": pa.array([r["event_token_ids"] for r in rows], type=pa.list_(pa.list_(pa.int32()))),
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
        "include_condition_occurrence": bool(args.include_condition_occurrence),
        "events_per_row": args.events_per_row,
        "num_unique_events": len(event_token_map),
        "num_rows": len(rows),
        "num_patients": len({r["patient_id"] for r in rows}),
        "avg_events_per_row": sum(r["num_events"] for r in rows) / max(len(rows), 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
