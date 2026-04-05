#!/usr/bin/env python3
"""
spot_check_train_data.py

Spot-checks prepare_task_data.py output by tracing source_row back to the
original build_task_data.py parquet and re-parsing the timeline to verify
that event_ids match exactly.

Uses PyArrow row-group navigation so only the specific row groups containing
the sampled rows are read — no full-file loads.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PREPARED_DIR = (
    "data/prepared/ep0/"
    "new_acutemi_new_celiac_new_hyperlipidemia_new_hypertension_new_lupus_new_pancan/"
    "train"
)
SOURCE_DIR = "EHRSHOT_ASSETS/llm_data_v6"
N_SAMPLES  = 50
SEED       = 42

TASK_2_IDX = {t: i for i, t in enumerate(sorted({
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}))}
IDX_2_TASK = {v: k for k, v in TASK_2_IDX.items()}


def parse_timeline(timeline: str) -> list[int]:
    eids = []
    for line in timeline.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = ev.get("embedding_idx")
        if eid is not None:
            eids.append(int(eid))
    return eids


def _is_positive(label_str) -> bool:
    if isinstance(label_str, bool):
        return label_str
    s = str(label_str).strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(float(s)) != 0
    except (ValueError, TypeError):
        return False


def read_rows_by_index(pf: pq.ParquetFile, abs_rows: set[int], columns: list[str]) -> dict[int, pd.Series]:
    """Read specific rows from a parquet file using row-group navigation.
    Only row groups that contain needed rows are read."""
    results = {}
    cumulative = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        rg_end  = cumulative + rg_rows
        needed  = {r for r in abs_rows if cumulative <= r < rg_end}
        if needed:
            df = pf.read_row_group(rg_idx, columns=columns).to_pandas()
            for abs_row in needed:
                results[abs_row] = df.iloc[abs_row - cumulative]
            if len(results) == len(abs_rows):
                break   # found everything
        cumulative = rg_end
    return results


# ── Step 1: sample N rows from prepared shards (scalar columns only) ──────────
shard_files = sorted(Path(PREPARED_DIR).glob("shard_*.parquet"))
print(f"Found {len(shard_files)} shard(s). Sampling (scalar columns only) …")

rng = random.Random(SEED)
# Track which shard file each global row belongs to
shard_row_counts = []
for fpath in shard_files:
    n = pq.read_metadata(str(fpath)).num_rows
    shard_row_counts.append(n)

total_rows  = sum(shard_row_counts)
sample_idxs = sorted(rng.sample(range(total_rows), min(N_SAMPLES, total_rows)))

# Map global row index → (shard_idx, local_row_idx)
shard_for_row = []
cumulative = 0
for si, cnt in enumerate(shard_row_counts):
    for _ in range(cnt):
        shard_for_row.append((si, len(shard_for_row) - cumulative))
    cumulative += cnt

# Group sampled rows by shard
by_shard: dict[int, set[int]] = defaultdict(set)
for gi in sample_idxs:
    si, li = shard_for_row[gi]
    by_shard[si].add(li)

# Read scalar columns only for sampled rows
scalar_cols = ["task_idx", "label", "source_row"]
sample_records: list[dict] = []
for si, local_rows in sorted(by_shard.items()):
    pf = pq.ParquetFile(str(shard_files[si]))
    rows = read_rows_by_index(pf, local_rows, scalar_cols)
    for li, row in rows.items():
        sample_records.append({
            "shard":      si,
            "local_row":  li,
            "task_idx":   int(row["task_idx"]),
            "label":      int(row["label"]),
            "source_row": int(row["source_row"]),
        })

print(f"  Sampled {len(sample_records)} rows\n")


# ── Step 2: fetch event_ids from prepared shards ──────────────────────────────
print("Fetching event_ids from prepared shards …")
for si, local_rows in sorted(by_shard.items()):
    pf   = pq.ParquetFile(str(shard_files[si]))
    rows = read_rows_by_index(pf, local_rows, ["event_ids"])
    for rec in sample_records:
        if rec["shard"] == si:
            rec["eids_prep"] = list(rows[rec["local_row"]]["event_ids"])
print("  Done\n")


# ── Step 3: fetch original rows from source parquets (by source_row) ──────────
print("Fetching original rows from source parquets …")
task_to_source_rows: dict[str, set[int]] = defaultdict(set)
for rec in sample_records:
    task_to_source_rows[IDX_2_TASK[rec["task_idx"]]].add(rec["source_row"])

orig_data: dict[str, dict[int, dict]] = {}
for task, needed in sorted(task_to_source_rows.items()):
    path = f"{SOURCE_DIR}/{task}/train.parquet"
    print(f"  {task}: reading {len(needed)} row(s) from {path}")
    pf   = pq.ParquetFile(path)
    rows = read_rows_by_index(pf, needed, ["label", "timeline"])
    orig_data[task] = {
        abs_row: {
            "label": 1 if _is_positive(row["label"]) else 0,
            "eids":  parse_timeline(row["timeline"] if isinstance(row["timeline"], str) else ""),
        }
        for abs_row, row in rows.items()
    }
print()


# ── Step 4: compare ────────────────────────────────────────────────────────────
print(f"Comparing {len(sample_records)} rows …\n")
n_pass = n_fail = 0
failures = []

for rec in sample_records:
    task       = IDX_2_TASK[rec["task_idx"]]
    source_row = rec["source_row"]
    orig       = orig_data[task].get(source_row)

    if orig is None:
        failures.append(f"[{task}] source_row={source_row} not found in original")
        n_fail += 1
        continue

    if rec["label"] != orig["label"]:
        failures.append(
            f"[{task}] source_row={source_row}: label mismatch "
            f"prepared={rec['label']} original={orig['label']}"
        )
        n_fail += 1
        continue

    if rec["eids_prep"] != orig["eids"]:
        same_len = len(rec["eids_prep"]) == len(orig["eids"])
        detail = (
            f"first diff at pos {next(i for i,(a,b) in enumerate(zip(rec['eids_prep'],orig['eids'])) if a!=b)}"
            if same_len else
            f"len prepared={len(rec['eids_prep'])} original={len(orig['eids'])}"
        )
        failures.append(f"[{task}] source_row={source_row}: event_ids mismatch — {detail}")
        n_fail += 1
        continue

    n_pass += 1

print("=" * 60)
print(f"Passed: {n_pass} / {n_pass + n_fail}")
if failures:
    print(f"\nFailures:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print("All spot-checks passed.")
    sys.exit(0)
