#!/usr/bin/env python3
"""
validate_train_data.py

Validates prepare_task_data.py output under
data/prepared/ep0/.../train/shard_*.parquet

Checks:
  - Schema: task_idx (int16), label (int8), event_ids (list<int32>), source_row (int32)
  - task_idx values are all in TASK_2_IDX
  - label values are only 0 or 1
  - event_ids are non-empty
  - Per-task pos/neg counts
  - Group-level pos/neg ratio (every group_size consecutive rows)
  - Task interleaving across groups
"""

import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import pyarrow.parquet as pq

TRAIN_DIR = (
    "data/prepared/ep0/"
    "new_acutemi_new_celiac_new_hyperlipidemia_new_hypertension_new_lupus_new_pancan/"
    "train"
)

# Must match prepare_task_data.py
TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}
TASK_2_IDX  = {t: i for i, t in enumerate(sorted(TASK_2_DISEASE_NAME))}
IDX_2_TASK  = {v: k for k, v in TASK_2_IDX.items()}
VALID_IDXS  = set(TASK_2_IDX.values())

# Change these if you used different values in prepare_task_data.py
POS_PER_GROUP = 1
NEG_PER_GROUP = 1
GROUP_SIZE    = POS_PER_GROUP + NEG_PER_GROUP

import pyarrow as pa
EXPECTED_SCHEMA = {
    "task_idx":   pa.int16(),
    "label":      pa.int8(),
    "event_ids":  pa.list_(pa.int32()),
    "source_row": pa.int32(),
}

errors   = []
warnings = []

def err(msg):
    errors.append(msg)
    print(f"  [ERROR] {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"  [WARN]  {msg}")

def ok(msg):
    print(f"  [OK]    {msg}")


shard_files = sorted(Path(TRAIN_DIR).glob("shard_*.parquet"))
if not shard_files:
    print(f"No shard files found in {TRAIN_DIR}")
    sys.exit(1)

print(f"Found {len(shard_files)} shard(s) in {TRAIN_DIR}\n")

all_dfs = []

for fpath in shard_files:
    print(f"── {fpath.name} ──")
    pf = pq.ParquetFile(str(fpath))

    # ── Schema ────────────────────────────────────────────────────────────────
    schema = pf.schema_arrow
    for col, expected_type in EXPECTED_SCHEMA.items():
        if col not in schema.names:
            err(f"[{fpath.name}] Missing column: {col}")
        elif not schema.field(col).type.equals(expected_type):
            err(f"[{fpath.name}] Column '{col}': expected '{expected_type}', "
                f"got '{schema.field(col).type}'")

    df = pd.read_parquet(str(fpath))
    n  = len(df)
    all_dfs.append(df)

    # ── task_idx ──────────────────────────────────────────────────────────────
    bad_tidx = df[~df["task_idx"].isin(VALID_IDXS)]
    if len(bad_tidx):
        err(f"[{fpath.name}] {len(bad_tidx)} rows with unknown task_idx: "
            f"{bad_tidx['task_idx'].unique().tolist()}")
    else:
        ok(f"task_idx all valid")

    # ── label ─────────────────────────────────────────────────────────────────
    bad_lbl = df[~df["label"].isin([0, 1])]
    if len(bad_lbl):
        err(f"[{fpath.name}] {len(bad_lbl)} rows with invalid label values")

    n_pos   = (df["label"] == 1).sum()
    n_neg   = (df["label"] == 0).sum()
    ok(f"{n} rows  |  pos={n_pos}  neg={n_neg}  ratio={n_pos/n:.3f}")

    # ── event_ids ─────────────────────────────────────────────────────────────
    empty_eids = df[df["event_ids"].apply(len) == 0]
    if len(empty_eids):
        warn(f"[{fpath.name}] {len(empty_eids)} rows with empty event_ids")
    else:
        ok(f"event_ids all non-empty")

    eid_lens = df["event_ids"].apply(len)
    ok(f"event_ids length: min={eid_lens.min()}  max={eid_lens.max()}  "
       f"mean={eid_lens.mean():.1f}")

    # ── Group-level pos/neg ratio ─────────────────────────────────────────────
    if n % GROUP_SIZE != 0:
        warn(f"[{fpath.name}] {n} rows not divisible by GROUP_SIZE={GROUP_SIZE} "
             f"(remainder {n % GROUP_SIZE})")
    else:
        n_groups    = n // GROUP_SIZE
        bad_groups  = 0
        for i in range(n_groups):
            grp = df.iloc[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]
            if grp["label"].sum() != POS_PER_GROUP:
                bad_groups += 1
        if bad_groups:
            err(f"[{fpath.name}] {bad_groups}/{n_groups} groups violate "
                f"pos/neg ratio ({POS_PER_GROUP}:{NEG_PER_GROUP})")
        else:
            ok(f"All {n_groups} groups have correct pos/neg ratio "
               f"({POS_PER_GROUP}:{NEG_PER_GROUP})")

    # ── Task distribution in groups ───────────────────────────────────────────
    if n >= GROUP_SIZE:
        group_tasks = [
            df.iloc[i * GROUP_SIZE]["task_idx"]
            for i in range(n // GROUP_SIZE)
        ]
        task_counter = Counter(group_tasks)
        ok(f"Groups per task: " +
           "  ".join(f"{IDX_2_TASK.get(int(t), t)}={c}"
                     for t, c in sorted(task_counter.items())))

    print()


# ── Global stats across all shards ────────────────────────────────────────────
print("── Global stats ──")
full = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows: {len(full):,}")
for tidx in sorted(VALID_IDXS):
    sub   = full[full["task_idx"] == tidx]
    n_pos = (sub["label"] == 1).sum()
    n_neg = (sub["label"] == 0).sum()
    name  = IDX_2_TASK.get(tidx, str(tidx))
    print(f"  {name}: {len(sub):,} rows  pos={n_pos}  neg={n_neg}")

print()
print("="*60)
print(f"Errors:   {len(errors)}")
print(f"Warnings: {len(warnings)}")
if errors:
    print("\nAll errors:")
    for e in errors:
        print(f"  {e}")
sys.exit(1 if errors else 0)
