"""
Merge per-task embedding parquets and redistribute into N balanced shards
for DDP training.

Each output shard contains a proportional slice of every (task, split, label)
combination, so all ranks see the same task/label distribution.

Usage:
    python shard_embedding_data.py \\
        --input_files data/embedding_inputs/new_diagnosis/*_m500_ntrain50000*.parquet \\
        --output_dir  data/embedding_inputs/sharded_m500 \\
        --n_shards    16

Then launch DDP with:
    torchrun --nproc_per_node=4 train_embedding_custom.py \\
        --data_paths data/embedding_inputs/sharded_m500/train_shard_*.parquet \\
        --val_data_paths data/embedding_inputs/sharded_m500/val.parquet \\
        ...

Notes:
  - Each shard contains ALL splits (train / val / test). The training script
    filters by split internally, so the same shards can be passed to both
    --data_paths and --val_data_paths.
  - Rows within each (task, split, label) group are shuffled then assigned to
    shards round-robin, guaranteeing perfect balance.
  - A combined val.parquet (all tasks, val split only) is also written for
    convenient single-file evaluation on rank 0.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--input_files", nargs="+", required=True,
        help="Input parquet files (supports shell glob expansion). "
             "Pass e.g. 'data/new_diagnosis/*_m500_ntrain50000*.parquet'.",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory where shards are written.",
    )
    p.add_argument(
        "--n_shards", type=int, default=16,
        help="Number of train shards to produce.",
    )
    p.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to include in the shards.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling within groups.",
    )
    p.add_argument(
        "--row_group_size", type=int, default=4096,
        help="Parquet row-group size (tune for read performance).",
    )
    return p.parse_args()


def expand_globs(patterns: list[str]) -> list[str]:
    """Expand any glob patterns in the list; return sorted unique paths."""
    paths = []
    for pat in patterns:
        expanded = glob.glob(pat)
        if not expanded:
            print(f"  WARNING: no files matched '{pat}'", file=sys.stderr)
        paths.extend(expanded)
    paths = sorted(set(paths))
    return paths


def main():
    args = parse_args()

    # ── Resolve input files ───────────────────────────────────────────────────
    input_files = expand_globs(args.input_files)
    if not input_files:
        sys.exit("ERROR: no input files found. Check --input_files.")

    print(f"Input files ({len(input_files)}):")
    for f in input_files:
        print(f"  {f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load and concatenate ──────────────────────────────────────────────────
    print(f"\nLoading {len(input_files)} parquet files...")
    dfs = []
    for f in tqdm(input_files, desc="  loading"):
        df = pd.read_parquet(f)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    print(f"\nLoaded {len(df_all):,} total rows.")
    print(f"Columns: {list(df_all.columns)}")
    print("\nDistribution before sharding:")
    dist = df_all.groupby(["task", "split", "label"]).size().rename("count")
    print(dist.to_string())

    # ── Filter to requested splits ────────────────────────────────────────────
    df_all = df_all[df_all["split"].isin(args.splits)].copy()
    print(f"\nAfter split filter {args.splits}: {len(df_all):,} rows.")

    # ── Assign shard indices via round-robin within each group ────────────────
    # Group key: (task, split, label) — ensures perfect balance across all dims.
    rng = np.random.default_rng(args.seed)

    shard_col = np.empty(len(df_all), dtype=np.int32)

    group_keys = ["task", "split", "label"]
    for keys, idx in tqdm(
        df_all.groupby(group_keys, sort=False).groups.items(),
        desc="  assigning shards",
    ):
        n = len(idx)
        # Shuffle within the group
        perm = rng.permutation(n)
        shuffled_idx = idx[perm]  # actual DataFrame indices
        # Round-robin assignment: row 0 → shard 0, row 1 → shard 1, ...
        assignments = np.arange(n) % args.n_shards
        shard_col[df_all.index.get_indexer(shuffled_idx)] = assignments

    df_all["_shard"] = shard_col

    # ── Write train shards ────────────────────────────────────────────────────
    print(f"\nWriting {args.n_shards} shards to {args.output_dir}/")
    shard_sizes = []
    for s in tqdm(range(args.n_shards), desc="  writing shards"):
        df_shard = df_all[df_all["_shard"] == s].drop(columns="_shard").reset_index(drop=True)
        shard_sizes.append(len(df_shard))
        out_path = os.path.join(args.output_dir, f"shard_{s:03d}.parquet")
        df_shard.to_parquet(
            out_path,
            index=False,
            row_group_size=args.row_group_size,
        )

    # ── Write combined val.parquet for easy single-file evaluation ────────────
    if "val" in args.splits:
        df_val = df_all[df_all["split"] == "val"].drop(columns="_shard").reset_index(drop=True)
        val_path = os.path.join(args.output_dir, "val.parquet")
        df_val.to_parquet(val_path, index=False, row_group_size=args.row_group_size)
        print(f"\nWrote val.parquet: {len(df_val):,} rows → {val_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Shard size summary ────────────────────────────────────────")
    shard_sizes = np.array(shard_sizes)
    print(f"  n_shards : {args.n_shards}")
    print(f"  min size : {shard_sizes.min():,}")
    print(f"  max size : {shard_sizes.max():,}")
    print(f"  mean size: {shard_sizes.mean():.0f}")
    print(f"  total    : {shard_sizes.sum():,}")

    # Verify balance: show distribution of first shard as a check
    first_shard_path = os.path.join(args.output_dir, "shard_000.parquet")
    df_check = pd.read_parquet(first_shard_path, columns=["task", "split", "label"])
    print(f"\n── shard_000 distribution (balance check) ───────────────────")
    print(df_check.groupby(["task", "split", "label"]).size().rename("count").to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
