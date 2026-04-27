#!/usr/bin/env python3
"""
Extract unique semantic events into a parquet for later text templating.

Each output parquet row corresponds to one unique semantic event, deduplicated by
the fields that matter for downstream text rendering:

    (omop_table, code, value, unit)

This matches the spirit of `extract_bio_emb.py` while preserving structured
columns for later Jinja templating. Deduplication is done with pandas
`drop_duplicates(keep="first")` on the unique-key columns.

By default, `condition_occurrence` rows are excluded to match the existing
timeline / CPT preprocessing behavior in this repo.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
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
    pa.field("event_id", pa.int64()),
    pa.field("omop_table", pa.string()),
    pa.field("event_type", pa.string()),
    pa.field("code", pa.string()),
    pa.field("description", pa.string()),
    pa.field("value", pa.string()),
    pa.field("unit", pa.string()),
])


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ehrshot_csv", default="EHRSHOT_ASSETS/data/ehrshot.csv")
    p.add_argument("--concept_csv", default="EHRSHOT_ASSETS/femr/logs/omop_dir/concept.csv")
    p.add_argument("--output_path", default="hx1/unique_events.parquet")
    p.add_argument(
        "--include_condition_occurrence",
        action="store_true",
        help="Keep condition_occurrence rows. Default matches existing preprocessing and drops them.",
    )
    return p.parse_args()


def normalize_str(x) -> str:
    if isinstance(x, str):
        return x.strip()
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_code_description_map(concept_csv: str) -> dict[str, str]:
    logger.info("Loading concept.csv ...")
    concept_df = pd.read_csv(
        concept_csv,
        usecols=["concept_name", "vocabulary_id", "concept_code"],
        low_memory=False,
        dtype=str,
    ).fillna("")
    concept_df["code"] = concept_df["vocabulary_id"] + "/" + concept_df["concept_code"]
    filtered = concept_df[concept_df["code"] != concept_df["concept_name"]]
    code2desc = dict(zip(filtered["code"], filtered["concept_name"]))
    logger.info("Loaded %s code→description mappings.", f"{len(code2desc):,}")
    return code2desc


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    code2desc = load_code_description_map(args.concept_csv)
    logger.info("Loading key columns from %s ...", args.ehrshot_csv)
    df = pd.read_csv(
        args.ehrshot_csv,
        usecols=["omop_table", "code", "value", "unit"],
        low_memory=False,
        dtype=str,
        keep_default_na=False,
    )

    if df.columns[0] == "" or str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    total_rows = len(df)
    logger.info("Loaded %s rows.", f"{total_rows:,}")

    if not args.include_condition_occurrence:
        before = len(df)
        df = df[df["omop_table"] != "condition_occurrence"].copy()
        logger.info(
            "Filtered condition_occurrence rows: %s -> %s",
            f"{before:,}",
            f"{len(df):,}",
        )

    for col in ["omop_table", "code", "value", "unit"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    logger.info("Dropping duplicates across %s rows ...", f"{len(df):,}")
    df = df.drop_duplicates(subset=["omop_table", "code", "value", "unit"], keep="first").reset_index(drop=True)

    logger.info("Building output rows for %s unique events ...", f"{len(df):,}")
    df.insert(0, "event_id", range(len(df)))
    df["description"] = df["code"].map(lambda code: normalize_str(code2desc.get(code, "")))
    df["event_type"] = df["omop_table"].map(lambda x: OMOP_TABLE_PREFIX.get(x, x))
    df = df[["event_id", "omop_table", "event_type", "code", "description", "value", "unit"]]

    logger.info("Writing %s unique events to %s", f"{len(df):,}", output_path)
    table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
    pq.write_table(table, output_path)

    logger.info("Done.")
    logger.info("Output parquet: %s", output_path)
    logger.info("Unique rows written: %s", f"{len(df):,}")
    logger.info("Rows seen: %s", f"{total_rows:,}")
    logger.info("Condition rows kept: %s", "yes" if args.include_condition_occurrence else "no")


if __name__ == "__main__":
    main()
