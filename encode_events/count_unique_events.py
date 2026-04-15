#!/usr/bin/env python3
"""
Count unique semantic events by loading the full key table and using pandas
`drop_duplicates` on the unique-key columns.

This is an alternative to the hash-set based approach in
`encode_events/count_unique_events.py`. It follows the deduplication rule used
by `encode_events/extract_event_parquet.py`:

    (omop_table, code, value, unit)

Algorithm:
1. Read only the key columns from ehrshot.csv
2. Optionally drop condition_occurrence rows
3. Normalize key fields to stripped strings
4. Use `drop_duplicates(subset=KEY_COLUMNS)`
5. Report the number of remaining rows
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


KEY_COLUMNS = ["omop_table", "code", "value", "unit"]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ehrshot_csv", default="EHRSHOT_ASSETS/data/ehrshot.csv")
    p.add_argument(
        "--include_condition_occurrence",
        action="store_true",
        help="Keep condition_occurrence rows. Default matches extract_event_parquet.py and drops them.",
    )
    return p.parse_args()


def normalize_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def load_key_frame(csv_path: str, include_condition_occurrence: bool) -> pd.DataFrame:
    logger.info("Loading key columns from %s ...", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=KEY_COLUMNS,
        low_memory=False,
        dtype=str,
        keep_default_na=False,
    )

    if df.columns[0] == "" or str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    if not include_condition_occurrence:
        before = len(df)
        df = df[df["omop_table"] != "condition_occurrence"].copy()
        logger.info(
            "Filtered condition_occurrence rows: %s -> %s",
            f"{before:,}",
            f"{len(df):,}",
        )

    for col in KEY_COLUMNS:
        df[col] = normalize_series(df[col])

    return df.reset_index(drop=True)


def count_unique_sorted(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    logger.info("Dropping duplicates across %s rows ...", f"{len(df):,}")
    unique_df = df.drop_duplicates(subset=KEY_COLUMNS, keep="first")
    return int(len(unique_df))


def main():
    args = parse_args()
    df = load_key_frame(args.ehrshot_csv, args.include_condition_occurrence)
    unique_count = count_unique_sorted(df)

    logger.info("Done.")
    logger.info("Rows considered: %s", f"{len(df):,}")
    logger.info("Condition rows kept: %s", "yes" if args.include_condition_occurrence else "no")
    logger.info("Unique events: %s", f"{unique_count:,}")


if __name__ == "__main__":
    main()
