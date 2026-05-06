#!/usr/bin/env python3
"""
Build event-EOT CPT training parquet from MIMIC-IV MEDS parquet data.

Replicates the logic of build_ehr_event_eot_cpt_parquet.py but reads from the
MIMIC-IV MEDS parquet files (mimic-2.2-meds/data/{split}/*.parquet) instead of
EHRSHOT CSV.

Each MEDS event is decoded into (omop_table, code, description, value, unit) and
rendered via the same Jinja2 template.  The output parquet is fully compatible with
train_ehr_event_eot_cpt.py (--train_parquet flag).

Code → (omop_table, code, description, value, unit) mapping
============================================================
MEDS code prefix           omop_table           code             value       unit
----------------------------------------------------------------------------------
LAB//{itemid}//{unit}      measurement          LAB/{itemid}     num/txt     {unit}
DIAGNOSIS//ICD//{v}//{c}   condition_occurrence ICD{v}CM/{c}     –           –
PROCEDURE//ICD//{v}//{c}   procedure_occurrence ICD{v}Proc/{c}   –           –
MEDICATION//{drug}//{txt}  drug_exposure        {drug}           {txt}       –
HOSPITAL_ADMISSION//…      visit_occurrence     HOSPITAL_ADMISSION  parts    –
HOSPITAL_DISCHARGE//…      visit_occurrence     HOSPITAL_DISCHARGE  parts    –
ICU_ADMISSION//…           visit_occurrence     ICU_ADMISSION    {unit}      –
ICU_DISCHARGE//…           visit_occurrence     ICU_DISCHARGE    {unit}      –
TRANSFER_TO//…             observation          TRANSFER_TO      rest        –
ED_REGISTRATION            visit_occurrence     ED_REGISTRATION  –           –
ED_OUT                     visit_occurrence     ED_OUT           –           –
DRG//{type}//{num}//{desc} observation          DRG/{type}/{num} {desc}      –
HCPCS//{desc}              procedure_occurrence {desc}           –           –
GENDER//{g}                person               GENDER           {g}         –
MEDS_BIRTH                 person               MEDS_BIRTH       –           –
MEDS_DEATH                 death                MEDS_DEATH       –           –
<OMR vitals>               measurement          {code}           {txt}       –

Description sources
===================
LAB items        : d_labitems.csv.gz         → label (+ fluid)
ICD-9/10 dx      : d_icd_diagnoses.csv.gz    → long_title
ICD-9/10 proc    : d_icd_procedures.csv.gz   → long_title
All others       : code itself (already semantic) or empty
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import polars as pl
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

# ── thread-local globals (same pattern as original builder) ──────────────────
_DF_EHR: pd.DataFrame | None = None
_EVENT_TOKEN_MAP: Dict[Tuple[str, str, str, str], List[int]] = {}
_SEQ_LEN = 2048
_PAD_TOKEN_ID = 0


# ── helpers ──────────────────────────────────────────────────────────────────

def normalize_optional_str(x) -> str | None:
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    if pd.isna(x):
        return None
    return str(x).strip() or None


def unique_event_key(omop_table: str, code: str, value: str, unit: str) -> Tuple[str, str, str, str]:
    return (
        normalize_optional_str(omop_table) or "",
        normalize_optional_str(code) or "",
        normalize_optional_str(value) or "",
        normalize_optional_str(unit) or "",
    )


def load_event_template(template_path: str):
    env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True,
                      undefined=StrictUndefined)
    return env.from_string(Path(template_path).read_text())


def format_event_text(omop_table: str, code: str, description: str,
                      value: str, unit: str, event_template) -> str | None:
    rendered = event_template.render(
        event_type=omop_table or "",
        description=description or "",
        code=code or "",
        value=value or "",
        unit=unit or "",
    ).strip()
    return rendered or None


# ── MIMIC description lookup tables ─────────────────────────────────────────

def build_mimic_description_maps(mimic_raw_dir: Path) -> dict:
    """
    Returns a dict with three sub-maps keyed by the normalized MEDS code prefix:
      lab_map  : itemid (str) → label string
      diag_map : (icd_version_str, icd_code_str) → long_title
      proc_map : (icd_version_str, icd_code_str) → long_title
    """
    hosp = mimic_raw_dir / "hosp"

    # Lab items: itemid → "label (fluid)"
    lab_df = pd.read_csv(hosp / "d_labitems.csv.gz",
                         usecols=["itemid", "label", "fluid"],
                         dtype=str).fillna("")
    lab_map = {
        str(row["itemid"]): f"{row['label']} ({row['fluid']})" if row["fluid"] else row["label"]
        for _, row in lab_df.iterrows()
    }
    logger.info("Lab description map: %s entries", f"{len(lab_map):,}")

    # ICD diagnoses: (version, icd_code) → long_title
    diag_df = pd.read_csv(hosp / "d_icd_diagnoses.csv.gz",
                          usecols=["icd_version", "icd_code", "long_title"],
                          dtype=str).fillna("")
    diag_map = {
        (str(r["icd_version"]), str(r["icd_code"])): r["long_title"]
        for _, r in diag_df.iterrows()
    }
    logger.info("ICD diagnosis description map: %s entries", f"{len(diag_map):,}")

    # ICD procedures: (version, icd_code) → long_title
    proc_df = pd.read_csv(hosp / "d_icd_procedures.csv.gz",
                          usecols=["icd_version", "icd_code", "long_title"],
                          dtype=str).fillna("")
    proc_map = {
        (str(r["icd_version"]), str(r["icd_code"])): r["long_title"]
        for _, r in proc_df.iterrows()
    }
    logger.info("ICD procedure description map: %s entries", f"{len(proc_map):,}")

    return {"lab": lab_map, "diag": diag_map, "proc": proc_map}


# ── MEDS code → EHR row ──────────────────────────────────────────────────────

# OMR-style vital sign codes (the code itself is already the semantic description)
_OMR_PREFIXES = (
    "Blood Pressure", "Weight", "BMI", "Height", "eGFR",
)


def parse_meds_event(
    code: str,
    numeric_value: float | None,
    text_value: str | None,
    desc_maps: dict,
) -> Tuple[str, str, str, str, str] | None:
    """
    Parse one MEDS event row into (omop_table, norm_code, description, value, unit).
    Returns None if the event should be skipped.

    Conventions:
      omop_table   : string fed to the template as event_type
      norm_code    : canonical code id (for deduplication and display)
      description  : human-readable label (empty if same as code or unknown)
      value        : string representation of the numeric or categorical value
      unit         : measurement unit string
    """
    parts = code.split("//")
    prefix = parts[0]

    # Helper: pick best value representation
    def _val() -> str:
        if numeric_value is not None and not (isinstance(numeric_value, float) and pd.isna(numeric_value)):
            # Use 6 significant figures to avoid float32 precision artifacts (e.g. 52.099998 → 52.1)
            v = f"{float(numeric_value):.6g}"
            return v if v not in ("nan", "") else ""
        if text_value and str(text_value).strip() not in ("", "___", "nan"):
            return str(text_value).strip()
        return ""

    # ── Lab events ────────────────────────────────────────────────────────────
    if prefix == "LAB" and len(parts) >= 2:
        itemid = parts[1]
        unit = parts[2] if len(parts) > 2 and parts[2] not in ("UNK", "") else ""
        norm_code = f"LAB/{itemid}"
        desc = desc_maps["lab"].get(itemid, "")
        return ("Measurement", norm_code, desc, _val(), unit)

    # ── ICD diagnoses ─────────────────────────────────────────────────────────
    if prefix == "DIAGNOSIS" and len(parts) == 4:
        # parts: ["DIAGNOSIS", "ICD", version, icd_code]
        ver = parts[2]          # "9" or "10"
        icd_code = parts[3]
        norm_code = f"ICD{ver}CM/{icd_code}"
        desc = desc_maps["diag"].get((ver, icd_code), "")
        return ("Condition", norm_code, desc, "", "")

    # ── ICD procedures ────────────────────────────────────────────────────────
    if prefix == "PROCEDURE" and len(parts) == 4:
        ver = parts[2]
        icd_code = parts[3]
        norm_code = f"ICD{ver}Proc/{icd_code}"
        desc = desc_maps["proc"].get((ver, icd_code), "")
        return ("Procedure", norm_code, desc, "", "")

    # ── Medications ───────────────────────────────────────────────────────────
    if prefix == "MEDICATION" and len(parts) >= 2:
        drug = parts[1].strip()
        event_txt = parts[2].strip() if len(parts) > 2 else ""
        # drug name is already descriptive; description = drug name
        return ("Drug", drug, drug, event_txt, "")

    # ── Hospital admissions / discharges ─────────────────────────────────────
    if prefix == "HOSPITAL_ADMISSION":
        details = " | ".join(p for p in parts[1:] if p.strip()) if len(parts) > 1 else ""
        return ("Visit", "HOSPITAL_ADMISSION", "Hospital Admission", details, "")

    if prefix == "HOSPITAL_DISCHARGE":
        location = parts[1].strip() if len(parts) > 1 else ""
        return ("Visit", "HOSPITAL_DISCHARGE", "Hospital Discharge", location, "")

    # ── ICU admissions / discharges ───────────────────────────────────────────
    if prefix == "ICU_ADMISSION":
        unit_name = parts[1].strip() if len(parts) > 1 else ""
        return ("Visit", "ICU_ADMISSION", "ICU Admission", unit_name, "")

    if prefix == "ICU_DISCHARGE":
        unit_name = parts[1].strip() if len(parts) > 1 else ""
        return ("Visit", "ICU_DISCHARGE", "ICU Discharge", unit_name, "")

    # ── ED events ─────────────────────────────────────────────────────────────
    if prefix == "ED_REGISTRATION":
        return ("Visit", "ED_REGISTRATION", "ED Registration", "", "")

    if prefix == "ED_OUT":
        return ("Visit", "ED_OUT", "ED Discharge", "", "")

    # ── Transfers ─────────────────────────────────────────────────────────────
    if prefix == "TRANSFER_TO":
        rest = " | ".join(p for p in parts[1:] if p.strip()) if len(parts) > 1 else ""
        return ("Observation", "TRANSFER_TO", "Transfer", rest, "")

    # ── DRG codes ─────────────────────────────────────────────────────────────
    if prefix == "DRG" and len(parts) >= 3:
        # parts: ["DRG", drg_type, drg_code, description...]
        drg_type = parts[1]
        drg_num = parts[2]
        drg_desc = " ".join(parts[3:]).strip() if len(parts) > 3 else ""
        norm_code = f"DRG/{drg_type}/{drg_num}"
        return ("Observation", norm_code, drg_desc, "", "")

    # ── HCPCS codes ───────────────────────────────────────────────────────────
    if prefix == "HCPCS" and len(parts) >= 2:
        desc = parts[1].strip()
        return ("Procedure", desc, desc, "", "")

    # ── Demographics ──────────────────────────────────────────────────────────
    if prefix == "GENDER" and len(parts) >= 2:
        gender_val = "Male" if parts[1].upper() == "M" else "Female"
        return ("Demographics", "GENDER", "Gender", gender_val, "")

    if prefix == "MEDS_BIRTH":
        return ("Demographics", "MEDS_BIRTH", "Date of Birth", "", "")

    if prefix == "MEDS_DEATH":
        return ("Death", "MEDS_DEATH", "Death", "", "")

    # ── OMR vital signs (Blood Pressure, Weight, BMI, Height, eGFR, …) ───────
    if any(code.startswith(p) for p in _OMR_PREFIXES):
        # code itself is the description (e.g., "Blood Pressure", "Weight (Lbs)")
        return ("Measurement", code, code, _val(), "")

    # Unknown / unsupported prefix – skip
    return None


# ── MEDS parquet → flat pandas DataFrame ────────────────────────────────────

def load_meds_split(meds_dir: Path, split: str, desc_maps: dict) -> pd.DataFrame:
    """
    Load one MEDS split (train/test) and flatten into a pandas DataFrame with
    columns: patient_id, start, omop_table, code, description, value, unit
    Sorted by (patient_id, start).
    """
    logger.info("Loading MEDS split: %s", split)
    lf = pl.scan_parquet(meds_dir / split / "*.parquet").select(
        "subject_id", "time", "code", "numeric_value", "text_value"
    )
    df_pl = lf.collect()
    logger.info("  Raw events in %s: %s", split, f"{len(df_pl):,}")

    rows = []
    for row in tqdm(df_pl.iter_rows(named=True), total=len(df_pl),
                    desc=f"parse {split}", dynamic_ncols=True):
        parsed = parse_meds_event(
            code=row["code"],
            numeric_value=row["numeric_value"],
            text_value=row["text_value"],
            desc_maps=desc_maps,
        )
        if parsed is None:
            continue
        omop_table, norm_code, desc, value, unit = parsed
        # Skip events with null time (e.g., GENDER has null time in MEDS)
        t = row["time"]
        if t is None:
            t = pd.Timestamp("1970-01-01")  # put static events at epoch so they sort first
        rows.append({
            "patient_id": int(row["subject_id"]),
            "start": pd.Timestamp(t),
            "omop_table": omop_table,
            "code": norm_code,
            "description": desc,
            "value": value,
            "unit": unit,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["patient_id", "start"], ascending=True).reset_index(drop=True)
    logger.info("  Parsed events in %s: %s across %s patients",
                split, f"{len(df):,}", f"{df['patient_id'].nunique():,}")
    return df


# ── event token cache (same as original) ─────────────────────────────────────

def build_event_token_cache(
    df_ehr: pd.DataFrame,
    tokenizer,
    event_template,
    append_eos_token_id: int,
    tokenize_batch_size: int,
) -> Dict[Tuple[str, str, str, str], List[int]]:
    key_df = df_ehr[["omop_table", "code", "description", "value", "unit"]].drop_duplicates(
        subset=["omop_table", "code", "value", "unit"], keep="first"
    ).reset_index(drop=True)
    logger.info("Unique events to tokenize: %s", f"{len(key_df):,}")

    unique_texts: List[str] = []
    unique_keys: List[Tuple[str, str, str, str]] = []
    dropped = 0
    for _, row in tqdm(key_df.iterrows(), total=len(key_df), desc="render events", dynamic_ncols=True):
        text = format_event_text(
            omop_table=row["omop_table"],
            code=row["code"],
            description=row["description"],
            value=row["value"],
            unit=row["unit"],
            event_template=event_template,
        )
        if not text:
            dropped += 1
            continue
        unique_keys.append(unique_event_key(row["omop_table"], row["code"], row["value"], row["unit"]))
        unique_texts.append(text)
    logger.info("Renderable unique events: %s (dropped=%s)", f"{len(unique_keys):,}", f"{dropped:,}")

    event_token_map: Dict[Tuple[str, str, str, str], List[int]] = {}
    for i in tqdm(range(0, len(unique_texts), tokenize_batch_size), desc="tokenize", dynamic_ncols=True):
        batch_texts = unique_texts[i: i + tokenize_batch_size]
        batch_keys = unique_keys[i: i + tokenize_batch_size]
        enc = tokenizer(batch_texts, add_special_tokens=False, return_attention_mask=False)
        for key, token_ids in zip(batch_keys, enc["input_ids"]):
            ids = [int(x) for x in token_ids]
            ids.append(int(append_eos_token_id))
            event_token_map[key] = ids
    logger.info("Token cache size: %s", f"{len(event_token_map):,}")
    return event_token_map


# ── patient block processing (same as original) ──────────────────────────────

def build_patient_ranges(patient_ids: Sequence[int]) -> List[Tuple[int, int, int]]:
    ranges: List[Tuple[int, int, int]] = []
    if not patient_ids:
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


def iter_blocks(xs, block_size: int):
    for i in range(0, len(xs), block_size):
        yield list(xs[i: i + block_size])


def init_thread_globals(*, df_ehr, event_token_map, seq_len, pad_token_id):
    global _DF_EHR, _EVENT_TOKEN_MAP, _SEQ_LEN, _PAD_TOKEN_ID
    _DF_EHR = df_ehr
    _EVENT_TOKEN_MAP = event_token_map
    _SEQ_LEN = seq_len
    _PAD_TOKEN_ID = int(pad_token_id)


def process_patient_block(block: List[Tuple[int, int, int]]) -> dict:
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
            chunk_ids = token_stream[token_start: token_start + _SEQ_LEN]
            chunk_events = event_stream[token_start: token_start + _SEQ_LEN]
            n = len(chunk_ids)

            input_ids = [_PAD_TOKEN_ID] * _SEQ_LEN
            attention_mask = [0] * _SEQ_LEN
            event_ids = [-1] * _SEQ_LEN
            labels = [-100] * _SEQ_LEN

            input_ids[:n] = chunk_ids
            attention_mask[:n] = [1] * n
            event_ids[:n] = chunk_events
            labels[:n] = chunk_ids

            rows.append({
                "patient_id": int(patient_id),
                "chunk_idx": int(chunk_idx),
                "num_valid_tokens": int(n),
                "num_unique_events_in_chunk": int(len(set(chunk_events))),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "event_ids": event_ids,
                "labels": labels,
            })

    return {"rows": rows, "patients": patient_count, "events": event_count}


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


def print_example(rows: List[dict], tokenizer, preview_tokens: int):
    if not rows:
        return
    r = rows[0]
    n = r["num_valid_tokens"]
    pn = min(preview_tokens, n)
    decoded = tokenizer.decode(r["input_ids"][:pn], skip_special_tokens=False)
    logger.info("Example sample: patient_id=%s chunk_idx=%s valid_tokens=%s",
                r["patient_id"], r["chunk_idx"], n)
    logger.info("  decoded_preview=%s", json.dumps(decoded))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build event-EOT CPT training parquet from MIMIC-IV MEDS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--meds_dir", default="/gpfs/home/zduan/codes/ethos-ares/mimic-2.2-meds/data",
                   help="Root of MEDS data (contains train/ and test/ subdirectories)")
    p.add_argument("--mimic_raw_dir", default="/gpfs/home/zduan/codes/ethos-ares/mimic-iv-2.2",
                   help="Root of raw MIMIC-IV 2.2 (contains hosp/ subdirectory)")
    p.add_argument("--split", default="all", choices=["train", "test", "all"],
                   help="Which MEDS split(s) to process. 'all' concatenates train+test.")
    p.add_argument("--template_path", default="01_gen_meta/templates/biolinkbert_event.j2")
    p.add_argument("--output_path", required=True)
    p.add_argument("--metadata_path", default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--tokenize_batch_size", type=int, default=4096)
    p.add_argument("--num_threads", type=int, default=16)
    p.add_argument("--patients_per_task", type=int, default=64)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--preview_tokens", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = (Path(args.metadata_path) if args.metadata_path
                     else output_path.with_suffix(".json"))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    meds_dir = Path(args.meds_dir)
    mimic_raw_dir = Path(args.mimic_raw_dir)

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(f"Tokenizer {args.model_name!r} has no pad_token_id.")
    logger.info("End-of-event token: %r (id=%d)", tokenizer.pad_token, pad_token_id)

    # Build description lookup tables
    desc_maps = build_mimic_description_maps(mimic_raw_dir)

    # Load MEDS data
    splits = ["train", "test"] if args.split == "all" else [args.split]
    dfs = [load_meds_split(meds_dir, s, desc_maps) for s in splits]
    df_ehr = pd.concat(dfs, ignore_index=True).sort_values(
        ["patient_id", "start"], ascending=True
    ).reset_index(drop=True)
    logger.info("Total events: %s across %s patients",
                f"{len(df_ehr):,}", f"{df_ehr['patient_id'].nunique():,}")

    if args.max_patients is not None:
        keep_ids = df_ehr["patient_id"].drop_duplicates().tolist()[: args.max_patients]
        df_ehr = df_ehr[df_ehr["patient_id"].isin(keep_ids)].copy()
        logger.info("Restricted to %s patients", f"{len(keep_ids):,}")

    # Build event token cache
    event_template = load_event_template(args.template_path)
    event_token_map = build_event_token_cache(
        df_ehr=df_ehr,
        tokenizer=tokenizer,
        event_template=event_template,
        append_eos_token_id=pad_token_id,
        tokenize_batch_size=args.tokenize_batch_size,
    )

    # Process patients in parallel
    patient_ranges = build_patient_ranges(df_ehr["patient_id"].tolist())
    logger.info("Patients: %s, blocks of %s", f"{len(patient_ranges):,}", args.patients_per_task)
    init_thread_globals(
        df_ehr=df_ehr,
        event_token_map=event_token_map,
        seq_len=args.seq_len,
        pad_token_id=pad_token_id,
    )

    sample_count = 0
    patient_count = 0
    event_count = 0
    example_rows: List[dict] = []
    blocks = list(iter_blocks(patient_ranges, args.patients_per_task))
    logger.info("Building parquet with %d worker(s), %d block(s), seq_len=%d",
                args.num_threads, len(blocks), args.seq_len)

    # Use multiprocessing.Pool (fork-based on Linux) for true CPU parallelism.
    # Globals (_DF_EHR, _EVENT_TOKEN_MAP, etc.) are inherited by child processes
    # via copy-on-write fork — no serialisation overhead for large objects.
    with pq.ParquetWriter(output_path, OUTPUT_SCHEMA) as writer:
        with mp.Pool(processes=args.num_threads) as pool:
            for result in tqdm(
                pool.imap_unordered(process_patient_block, blocks),
                total=len(blocks),
                desc="patient blocks",
                dynamic_ncols=True,
            ):
                rows = result["rows"]
                patient_count += result["patients"]
                event_count += result["events"]
                if not rows:
                    continue
                if not example_rows:
                    example_rows = rows[:1]
                writer.write_table(rows_to_table(rows))
                sample_count += len(rows)

    metadata = {
        "model_name": args.model_name,
        "meds_dir": str(meds_dir),
        "mimic_raw_dir": str(mimic_raw_dir),
        "split": args.split,
        "template_path": args.template_path,
        "seq_len": args.seq_len,
        "tokenize_batch_size": args.tokenize_batch_size,
        "num_threads": args.num_threads,
        "patients_per_task": args.patients_per_task,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(pad_token_id),
        "samples": sample_count,
        "patients": patient_count,
        "events": event_count,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    logger.info("Done. samples=%s patients=%s events=%s",
                f"{sample_count:,}", f"{patient_count:,}", f"{event_count:,}")
    logger.info("Parquet -> %s", output_path)
    logger.info("Metadata -> %s", metadata_path)
    if example_rows:
        print_example(example_rows, tokenizer, args.preview_tokens)


if __name__ == "__main__":
    main()
