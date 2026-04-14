#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
_TOKENIZER = None
_CODE_2_DESC: dict[str, str] = {}
_INCLUDE_CONDITION_OCCURRENCE = False
_EOS_TOKEN_ID = None
_BLOCK_SIZE = None


def load_code_description_map(data_dir: Path) -> dict[str, str]:
    t2c_path = data_dir / "models" / "clmbr" / "token_2_code.json"
    t2d_path = data_dir / "models" / "clmbr" / "token_2_description.json"
    if not t2c_path.exists() or not t2d_path.exists():
        print(f"[WARN] Code description files not found under {data_dir}; raw codes will be used.")
        return {}
    with open(t2c_path) as f:
        token_2_code = json.load(f)
    with open(t2d_path) as f:
        token_2_desc = json.load(f)
    code_2_desc: dict[str, str] = {}
    for token_id, code in token_2_code.items():
        desc = token_2_desc.get(token_id)
        if desc and desc != code:
            code_2_desc[str(code)] = str(desc)
    print(f"Loaded {len(code_2_desc)} code->description mappings")
    return code_2_desc


def normalize_optional_str(x) -> str | None:
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    if pd.isna(x):
        return None
    return str(x).strip() or None


def normalize_optional_float(x) -> float | None:
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def format_event_row(ev: pd.Series, code_2_desc: dict[str, str], include_condition_occurrence: bool) -> str | None:
    if (not include_condition_occurrence) and ev["omop_table"] == "condition_occurrence":
        return None

    code = str(ev["code"])
    event_type = OMOP_TABLE_PREFIX.get(ev["omop_table"], ev["omop_table"])
    time = str(ev["start"])[:16]
    description = code_2_desc.get(code)
    value = normalize_optional_float(ev["value"])
    unit = normalize_optional_str(ev["unit"])

    label = f"{code} {description}" if description else code
    parts = [f"[{event_type}]", time, "|", label]
    if value is not None:
        val_str = f"{value:g}"
        if unit:
            val_str += f" {unit}"
        parts.append(val_str)

    return " ".join(parts)


def chunk_list(xs: list[int], block_size: int) -> list[list[int]]:
    n_full = len(xs) // block_size
    return [xs[i * block_size : (i + 1) * block_size] for i in range(n_full)]


def _init_worker(
    patient_groups: dict[int, pd.DataFrame],
    tokenizer_name: str,
    local_files_only: bool,
    code_2_desc: dict[str, str],
    include_condition_occurrence: bool,
    eos_token_id: int,
    block_size: int,
):
    global _PATIENT_GROUPS, _TOKENIZER, _CODE_2_DESC, _INCLUDE_CONDITION_OCCURRENCE, _EOS_TOKEN_ID, _BLOCK_SIZE
    _PATIENT_GROUPS = patient_groups
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
    _CODE_2_DESC = code_2_desc
    _INCLUDE_CONDITION_OCCURRENCE = include_condition_occurrence
    _EOS_TOKEN_ID = eos_token_id
    _BLOCK_SIZE = block_size


def _process_patient(patient_id: int) -> tuple[list[dict], int, int]:
    events_df = _PATIENT_GROUPS[int(patient_id)]
    event_texts: list[str] = []
    valid_events = 0
    for _, ev in events_df.iterrows():
        event_text = format_event_row(ev, _CODE_2_DESC, _INCLUDE_CONDITION_OCCURRENCE)
        if event_text is None:
            continue
        event_texts.append(event_text)
        valid_events += 1

    if not event_texts:
        return [], 0, 0

    enc = _TOKENIZER(event_texts, add_special_tokens=False)
    patient_tokens: list[int] = []
    for event_ids in enc["input_ids"]:
        if not event_ids:
            continue
        patient_tokens.extend(int(x) for x in event_ids)
        patient_tokens.append(int(_EOS_TOKEN_ID))

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
    p.add_argument("--tokenizer_name", required=True, help="Tokenizer name or local path, e.g. Qwen/Qwen3-0.6B-Base")
    p.add_argument("--data_dir", required=True, help="EHRSHOT asset root containing data/ehrshot.csv and models/clmbr/*")
    p.add_argument("--ehr_csv", default=None, help="Override raw EHR CSV path; defaults to <data_dir>/data/ehrshot.csv")
    p.add_argument("--output_path", required=True, help="Output parquet path")
    p.add_argument("--block_size", type=int, required=True, help="Fixed token length N for each row/block")
    p.add_argument("--include_condition_occurrence", action="store_true", help="Keep condition_occurrence rows")
    p.add_argument("--max_patients", type=int, default=None, help="Optional cap for quick experiments")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--num_workers", type=int, default=None, help="Patient-level worker processes (default: cpu_count)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    ehr_csv = Path(args.ehr_csv) if args.ehr_csv else data_dir / "data" / "ehrshot.csv"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, local_files_only=args.local_files_only)
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token = tokenizer.pad_token
        eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has neither eos_token_id nor pad_token_id; cannot append end-of-event token.")
    print(f"Using end-of-event token: {eos_token!r} (id={eos_token_id})")

    code_2_desc = load_code_description_map(data_dir)

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

    # Preview: show the first few formatted event strings for the first patient
    first_pid = next(iter(patient_groups))
    sample_texts = []
    for _, ev in patient_groups[first_pid].iterrows():
        t = format_event_row(ev, code_2_desc, args.include_condition_occurrence)
        if t:
            sample_texts.append(t)
        if len(sample_texts) >= 5:
            break
    print(f"\n--- Sample event strings (patient {first_pid}) ---")
    for t in sample_texts:
        print(" ", t)
    print("---\n")

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
            args.tokenizer_name,
            args.local_files_only,
            code_2_desc,
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
        "eos_token": eos_token,
        "eos_token_id": int(eos_token_id),
        "block_size": args.block_size,
        "include_condition_occurrence": bool(args.include_condition_occurrence),
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
