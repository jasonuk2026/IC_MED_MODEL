#!/usr/bin/env python3
"""
compare_embeddings.py

Randomly sample N events that appear in both embedding stores and compare
their vectors numerically.

Usage:
    python compare_embeddings.py \
        --dir_a data/biolinkbert_embeddings_correct \
        --dir_b data/biolinkbert_embeddings \
        --n 20 --seed 42
"""

import argparse
import random
import numpy as np
import pandas as pd


def load_store(directory: str):
    index = pd.read_parquet(f"{directory}/event_index.parquet")
    embs  = np.load(f"{directory}/embeddings.npy", mmap_mode="r")
    # key: (code, value, unit) → row index
    key2id = {
        (str(r.code), str(r.value), str(r.unit)): int(r.event_id)
        for r in index.itertuples()
    }
    return key2id, embs, index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir_a", default="data/biolinkbert_embeddings_correct")
    p.add_argument("--dir_b", default="data/biolinkbert_embeddings")
    p.add_argument("--n",     type=int, default=20, help="number of events to sample")
    p.add_argument("--seed",  type=int, default=42)
    args = p.parse_args()

    print(f"Loading {args.dir_a} ...")
    key2id_a, embs_a, index_a = load_store(args.dir_a)
    print(f"  {len(key2id_a):,} events,  embeddings shape: {embs_a.shape}")

    print(f"Loading {args.dir_b} ...")
    key2id_b, embs_b, index_b = load_store(args.dir_b)
    print(f"  {len(key2id_b):,} events,  embeddings shape: {embs_b.shape}")

    common_keys = list(set(key2id_a) & set(key2id_b))
    print(f"\nCommon events: {len(common_keys):,}  "
          f"(only-in-A: {len(key2id_a)-len(common_keys):,},  "
          f"only-in-B: {len(key2id_b)-len(common_keys):,})")

    if not common_keys:
        print("No common events — cannot compare.")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(common_keys, min(args.n, len(common_keys)))

    print(f"\n{'─'*80}")
    print(f"{'key (code | value | unit)':<50}  {'max_abs_diff':>12}  {'cos_sim':>8}  {'allclose':>8}")
    print(f"{'─'*80}")

    max_diffs, cos_sims, all_close_flags = [], [], []

    for key in sample:
        va = embs_a[key2id_a[key]].astype(np.float32)
        vb = embs_b[key2id_b[key]].astype(np.float32)

        max_diff = float(np.abs(va - vb).max())
        cos_sim  = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        close    = np.allclose(va, vb, atol=1e-5, rtol=1e-4)

        max_diffs.append(max_diff)
        cos_sims.append(cos_sim)
        all_close_flags.append(close)

        label = f"{key[0]} | {key[1]} | {key[2]}"
        print(f"{label[:50]:<50}  {max_diff:>12.6f}  {cos_sim:>8.6f}  {'✓' if close else '✗':>8}")

    print(f"{'─'*80}")
    print(f"{'SUMMARY':<50}  {max(max_diffs):>12.6f}  {sum(cos_sims)/len(cos_sims):>8.6f}  "
          f"{'all ✓' if all(all_close_flags) else f'{sum(all_close_flags)}/{len(all_close_flags)} ✓':>8}")


if __name__ == "__main__":
    main()
