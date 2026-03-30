#!/usr/bin/env python3
"""
check_event_lookup.py

For a given input parquet (same format as training data), check what fraction
of (code, value, unit) event keys are present in the BioLinkBERT event index.

Usage:
    python check_event_lookup.py \
        --data_paths data/embedding_inputs/sharded_m500/train_shard_*.parquet \
        --bert_index data/biolinkbert_embeddings/event_index.parquet

The script reports:
  • total event slots across all records
  • number / fraction of misses (key not in index)
  • top-N missing keys by occurrence count
"""

import argparse
import glob
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from extract_biolinkbert_embeddings import normalise_event_key

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_paths", nargs="+", required=True,
                   help="Input parquet file(s). Supports glob patterns.")
    p.add_argument("--bert_index", default="data/biolinkbert_embeddings/event_index.parquet",
                   help="event_index.parquet produced by extract_biolinkbert_embeddings.py.")
    p.add_argument("--top_missing", type=int, default=20,
                   help="Show this many most-frequent missing keys.")
    return p.parse_args()


def main():
    args = parse_args()

    # Expand globs
    paths: list[str] = []
    for pat in args.data_paths:
        expanded = sorted(glob.glob(pat))
        if not expanded:
            logger.warning(f"Pattern matched no files: {pat}")
        paths.extend(expanded)
    if not paths:
        raise ValueError("No input files found.")
    logger.info(f"Input files: {len(paths)}")

    # Build key set from index
    logger.info(f"Loading event index from {args.bert_index} …")
    index_df = pd.read_parquet(args.bert_index)
    key_set: set[tuple[str, str, str]] = {
        normalise_event_key(c, v, u)
        for c, v, u in zip(index_df["code"], index_df["value"], index_df["unit"])
    }
    logger.info(f"  {len(key_set):,} unique keys in index.")

    total_events = 0
    miss_counter: Counter = Counter()

    for path in paths:
        df = pd.read_parquet(path)
        for events in df["events"]:
            for e in events:
                total_events += 1
                key = normalise_event_key(e.get("code"), e.get("value"), e.get("unit"))
                if key not in key_set:
                    miss_counter[key] += 1

    n_miss = sum(miss_counter.values())
    hit_rate = 100.0 * (total_events - n_miss) / max(total_events, 1)

    logger.info("─" * 60)
    logger.info(f"Total event slots : {total_events:,}")
    logger.info(f"Hits              : {total_events - n_miss:,}  ({hit_rate:.2f}%)")
    logger.info(f"Misses            : {n_miss:,}  ({100-hit_rate:.2f}%)")
    logger.info(f"Unique missing keys: {len(miss_counter):,}")

    if miss_counter:
        logger.info(f"\nTop-{args.top_missing} missing keys (code, value, unit)  [count]:")
        for key, cnt in miss_counter.most_common(args.top_missing):
            logger.info(f"  {cnt:>8,}  {key}")
    else:
        logger.info("All event keys found in index — 100% hit rate.")


if __name__ == "__main__":
    main()
