#!/usr/bin/env python3
"""
Build tokenized evaluation parquet for MIMIC downstream tasks.

Analogous to build_eval_tokenized_eot_data.py but reading from:
  - MEDS parquet (mimic-2.2-meds/data/{split}/*.parquet)
  - task_labels/*.parquet  (from extract_task_labels.py)
  - MIMIC raw directory    (for d_labitems / d_icd_* description lookup)

For each labeled patient-admission pair:
  1. Load all events before anchor_time  (temporal cutoff)
  2. Convert MEDS codes → event text via Jinja2 template
  3. Tokenize with the CPT model's tokenizer
  4. Truncate from the TAIL to max_tokens (keep most recent events)
  5. Write (subject_id, task, label, split, input_ids, attention_mask, event_ids)

No padding is added here — the fine-tuning collator pads at batch time.

Usage:
  python build_mimic_eval_parquet.py \
      --model_name Qwen/Qwen3-0.6B \
      --meds_dir /gpfs/home/zduan/codes/ethos-ares/mimic-2.2-meds/data \
      --mimic_raw_dir /gpfs/home/zduan/codes/ethos-ares/mimic-iv-2.2 \
      --task_labels_dir /gpfs/home/zduan/codes/ethos-ares/task_labels \
      --tasks icu_mortality hospital_readmission_30d \
      --output_dir /gpfs/home/zduan/codes/ehr/ordered_data/mimic_eval \
      --max_tokens 8192 \
      --num_workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

# ── import shared utilities from build_mimic_cpt_parquet.py ─────────────────
_EHR_DIR = Path("/gpfs/home/zduan/codes/ehr")
if str(_EHR_DIR) not in sys.path:
    sys.path.insert(0, str(_EHR_DIR))

from build_mimic_cpt_parquet import (
    build_mimic_description_maps,
    build_event_token_cache,
    load_event_template,
    parse_meds_event,
    normalize_optional_str,
    unique_event_key,
    format_event_text,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

SUPPORTED_TASKS = [
    "icu_mortality",
    "icu_mortality_sepsis",
    "icu_mortality_after_24h",
    "icu_mortality_after_24h_sepsis",
    "hospital_readmission_30d",
    "icu_readmission",
    "icu_readmission_ich",
    "hospital_mortality",
]

OUTPUT_SCHEMA = pa.schema([
    pa.field("subject_id",            pa.int64()),
    pa.field("task",                  pa.string()),
    pa.field("label",                 pa.int8()),
    pa.field("split",                 pa.string()),
    pa.field("num_valid_tokens",      pa.int32()),
    pa.field("num_unique_events",     pa.int32()),
    pa.field("input_ids",             pa.list_(pa.int32())),
    pa.field("attention_mask",        pa.list_(pa.int8())),
    pa.field("event_ids",             pa.list_(pa.int32())),
])

# ── process globals (shared across multiprocessing workers) ──────────────────
_GROUPED: Dict[int, pd.DataFrame] = {}
_EVENT_TOKEN_MAP: Dict[tuple, List[int]] = {}
_MAX_TOKENS: int = 2048
_TASK_NAME: str = ""
_SPLIT_NAME: str = ""


def _load_meds_split_flat(meds_dir: Path, split: str, desc_maps: dict) -> pd.DataFrame:
    """Load MEDS parquet → flat DataFrame (patient_id, start, omop_table, code, description, value, unit)."""
    logger.info("Loading MEDS split '%s' from %s", split, meds_dir / split)
    df_pl = pl.scan_parquet(meds_dir / split / "*.parquet").select(
        "subject_id", "time", "code", "numeric_value", "text_value"
    ).collect()
    logger.info("  Raw events: %s", f"{len(df_pl):,}")

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
        t = row["time"]
        if t is None:
            t = pd.Timestamp("1970-01-01")
        rows.append({
            "patient_id": int(row["subject_id"]),
            "start":      pd.Timestamp(t),
            "omop_table": omop_table,
            "code":       norm_code,
            "description": desc,
            "value":      value,
            "unit":       unit,
        })

    df = pd.DataFrame(rows).sort_values(["patient_id", "start"]).reset_index(drop=True)
    logger.info("  Parsed events: %s across %s patients",
                f"{len(df):,}", f"{df['patient_id'].nunique():,}")
    return df


def _process_one_sample(args_tuple):
    """Worker: build token stream for one (subject_id, anchor_time, label) sample."""
    subject_id, anchor_time, label = args_tuple
    patient_df = _GROUPED.get(int(subject_id))
    if patient_df is None:
        return None

    # Temporal cutoff: only events strictly before anchor_time
    events_before = patient_df[patient_df["start"] < anchor_time]
    if len(events_before) == 0:
        return None

    token_stream: List[int] = []
    event_stream: List[int] = []
    event_idx = 0
    for row in events_before.itertuples(index=False):
        key = unique_event_key(row.omop_table, row.code, row.value, row.unit)
        token_ids = _EVENT_TOKEN_MAP.get(key)
        if token_ids is None:
            continue
        token_stream.extend(token_ids)
        event_stream.extend([event_idx] * len(token_ids))
        event_idx += 1

    if not token_stream:
        return None

    # Truncate from the TAIL → keep the most recent events
    if len(token_stream) > _MAX_TOKENS:
        token_stream = token_stream[-_MAX_TOKENS:]
        event_stream = event_stream[-_MAX_TOKENS:]
        # Re-index event_ids to start from 0
        offset = event_stream[0]
        event_stream = [e - offset for e in event_stream]

    n = len(token_stream)
    return {
        "subject_id":        int(subject_id),
        "task":              _TASK_NAME,
        "label":             int(1 if label else 0),
        "split":             _SPLIT_NAME,
        "num_valid_tokens":  n,
        "num_unique_events": len(set(event_stream)),
        "input_ids":         token_stream,
        "attention_mask":    [1] * n,
        "event_ids":         event_stream,
    }


def _flush(writer: pq.ParquetWriter, rows: List[dict]):
    table = pa.table({
        "subject_id":        [r["subject_id"]        for r in rows],
        "task":              [r["task"]               for r in rows],
        "label":             [r["label"]              for r in rows],
        "split":             [r["split"]              for r in rows],
        "num_valid_tokens":  [r["num_valid_tokens"]   for r in rows],
        "num_unique_events": [r["num_unique_events"]  for r in rows],
        "input_ids":         [r["input_ids"]          for r in rows],
        "attention_mask":    [r["attention_mask"]     for r in rows],
        "event_ids":         [r["event_ids"]          for r in rows],
    }, schema=OUTPUT_SCHEMA)
    writer.write_table(table)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_name",      default="Qwen/Qwen3-0.6B")
    p.add_argument("--meds_dir",        required=True,
                   help="Path to mimic-2.2-meds/data")
    p.add_argument("--mimic_raw_dir",   required=True,
                   help="Path to mimic-iv-2.2 (contains hosp/)")
    p.add_argument("--task_labels_dir", required=True,
                   help="Directory with task parquets from extract_task_labels.py")
    p.add_argument("--template_path",
                   default=str(_EHR_DIR / "01_gen_meta/templates/biolinkbert_event.j2"))
    p.add_argument("--tasks",    nargs="+", default=["icu_mortality", "hospital_readmission_30d"],
                   choices=SUPPORTED_TASKS)
    p.add_argument("--splits",   nargs="+", default=["train", "test"],
                   choices=["train", "test"])
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--max_tokens",       type=int, default=8192)
    p.add_argument("--tokenize_batch_size", type=int, default=4096)
    p.add_argument("--num_workers",      type=int, default=16)
    p.add_argument("--buffer_size",      type=int, default=1024)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--preview_tokens",   type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── tokenizer ──
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id; needed as EOT token.")
    logger.info("EOT token: %r (id=%d)", tokenizer.pad_token, pad_token_id)

    # ── MIMIC description maps ──
    desc_maps = build_mimic_description_maps(Path(args.mimic_raw_dir))
    event_template = load_event_template(args.template_path)

    # ── load each MEDS split once; process all tasks for that split ──
    meds_dir = Path(args.meds_dir)
    task_labels_dir = Path(args.task_labels_dir)

    global _GROUPED, _EVENT_TOKEN_MAP, _MAX_TOKENS, _TASK_NAME, _SPLIT_NAME

    _MAX_TOKENS = args.max_tokens

    metadata: dict = {
        "model_name": args.model_name,
        "tasks": args.tasks,
        "splits": args.splits,
        "max_tokens": args.max_tokens,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(pad_token_id),
        "truncate_side": "last",
    }

    for split in args.splits:
        logger.info("=" * 60)
        logger.info("Processing MEDS split: %s", split)

        # Load flat EHR for this split
        df_ehr = _load_meds_split_flat(meds_dir, split, desc_maps)

        # Build event token cache (deduplicated)
        event_token_map = build_event_token_cache(
            df_ehr=df_ehr,
            tokenizer=tokenizer,
            event_template=event_template,
            append_eos_token_id=pad_token_id,
            tokenize_batch_size=args.tokenize_batch_size,
        )

        # Index patients
        grouped = {int(pid): pdf for pid, pdf in df_ehr.groupby("patient_id", sort=False)}
        logger.info("Indexed %s patients for split '%s'", f"{len(grouped):,}", split)

        # Share with workers
        _GROUPED = grouped
        _EVENT_TOKEN_MAP = event_token_map

        for task in args.tasks:
            label_path = task_labels_dir / f"{task}.parquet"
            if not label_path.exists():
                logger.warning("Task label file not found: %s — skipping", label_path)
                continue

            df_labels = pl.read_parquet(label_path).filter(
                pl.col("split") == split
            ).to_pandas()

            if len(df_labels) == 0:
                logger.info("[%s:%s] No samples — skipping", task, split)
                continue

            # anchor_time is the prediction cutoff
            df_labels["anchor_time"] = pd.to_datetime(df_labels["anchor_time"])

            logger.info("[%s:%s] %s labeled samples", task, split, f"{len(df_labels):,}")
            _TASK_NAME  = task
            _SPLIT_NAME = split

            worker_args = [
                (int(row.subject_id), row.anchor_time, bool(row.label))
                for row in df_labels.itertuples(index=False)
            ]

            task_dir = output_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            out_path = task_dir / f"{split}.parquet"
            writer = pq.ParquetWriter(out_path, OUTPUT_SCHEMA)
            buffer: List[dict] = []
            n_written = n_skipped = 0
            example_row = None

            chunksize = max(1, len(worker_args) // max(args.num_workers * 10, 1))
            with mp.Pool(processes=args.num_workers) as pool:
                for result in tqdm(
                    pool.imap(_process_one_sample, worker_args, chunksize=chunksize),
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

            writer.close()
            pos = sum(1 for r in [example_row] if r and r["label"] == 1)  # placeholder
            if n_written > 0:
                _tmp = pq.read_table(out_path, columns=["label"])
                pos = int((_tmp["label"].to_pylist().count(1)))
            neg = n_written - pos
            logger.info(
                "  [%s:%s] %s rows  pos=%s (%.1f%%)  neg=%s  skipped=%s  -> %s",
                task, split,
                f"{n_written:,}", f"{pos:,}", 100 * pos / max(n_written, 1),
                f"{neg:,}", f"{n_skipped:,}", out_path,
            )

            if example_row is not None:
                preview = min(args.preview_tokens, example_row["num_valid_tokens"])
                decoded = tokenizer.decode(example_row["input_ids"][:preview],
                                           skip_special_tokens=False)
                logger.info("  Example  subject_id=%s  label=%s  tokens=%s  preview=%s",
                            example_row["subject_id"], example_row["label"],
                            example_row["num_valid_tokens"], json.dumps(decoded))

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Metadata -> %s", metadata_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
