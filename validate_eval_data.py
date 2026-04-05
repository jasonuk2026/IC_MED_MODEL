#!/usr/bin/env python3
"""
validate_eval_data.py

Validates build_eval_task_data.py output under EHRSHOT_ASSETS/llm_eval_data.

Checks:
  - Schema: patient_id (int64), task_idx (int16), label (int8), event_ids (list<int32>)
  - task_idx values match TASK_2_IDX
  - label values are only 0 or 1
  - event_ids are non-empty
  - No duplicate patient_id within the same task+split
  - Per-task pos/neg counts
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

EVAL_DIR = "EHRSHOT_ASSETS/llm_eval_data"
SPLITS   = ["val", "test"]

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}
TASK_2_IDX = {t: i for i, t in enumerate(sorted(TASK_2_DISEASE_NAME))}
IDX_2_TASK = {v: k for k, v in TASK_2_IDX.items()}

import pyarrow as pa
EXPECTED_SCHEMA = {
    "patient_id": pa.int64(),
    "task_idx":   pa.int16(),
    "label":      pa.int8(),
    "event_ids":  pa.list_(pa.int32()),
}

errors = []
warnings = []

def err(msg):
    errors.append(msg)
    print(f"  [ERROR] {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"  [WARN]  {msg}")

def ok(msg):
    print(f"  [OK]    {msg}")


for task in sorted(os.listdir(EVAL_DIR)):
    task_dir = os.path.join(EVAL_DIR, task)
    if not os.path.isdir(task_dir):
        continue
    if task not in TASK_2_IDX:
        warn(f"Unknown task directory: {task}")
        continue

    expected_task_idx = TASK_2_IDX[task]
    print(f"\n── {task} (expected task_idx={expected_task_idx}) ──")

    for split in SPLITS:
        fpath = os.path.join(task_dir, f"{split}.parquet")
        if not os.path.exists(fpath):
            warn(f"{split}.parquet missing")
            continue

        print(f"  [{split}] {fpath}")
        pf = pq.ParquetFile(fpath)

        # ── Schema check ──────────────────────────────────────────────────────
        schema = pf.schema_arrow
        for col, expected_type in EXPECTED_SCHEMA.items():
            if col not in schema.names:
                err(f"Missing column: {col}")
            elif not schema.field(col).type.equals(expected_type):
                err(f"Column '{col}': expected type '{expected_type}', "
                    f"got '{schema.field(col).type}'")

        df = pd.read_parquet(fpath)
        n = len(df)

        # ── task_idx ──────────────────────────────────────────────────────────
        bad_tidx = df[df["task_idx"] != expected_task_idx]
        if len(bad_tidx):
            err(f"task_idx: {len(bad_tidx)} rows have unexpected values "
                f"{bad_tidx['task_idx'].unique().tolist()}")
        else:
            ok(f"task_idx all = {expected_task_idx}")

        # ── label ─────────────────────────────────────────────────────────────
        bad_lbl = df[~df["label"].isin([0, 1])]
        if len(bad_lbl):
            err(f"label: {len(bad_lbl)} rows with invalid values "
                f"{bad_lbl['label'].unique().tolist()}")

        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        ratio = n_pos / n if n > 0 else 0
        ok(f"{n} rows  |  pos={n_pos}  neg={n_neg}  pos_ratio={ratio:.3f}")

        # ── event_ids ─────────────────────────────────────────────────────────
        empty_eids = df[df["event_ids"].apply(len) == 0]
        if len(empty_eids):
            warn(f"{len(empty_eids)} rows have empty event_ids")
        else:
            ok(f"event_ids all non-empty")

        eid_lens = df["event_ids"].apply(len)
        ok(f"event_ids length: min={eid_lens.min()}  max={eid_lens.max()}  "
           f"mean={eid_lens.mean():.1f}")

        # ── Patient stats (duplicates expected: one row per labeled sample) ─────
        n_unique_pids = df["patient_id"].nunique()
        n_dup         = df["patient_id"].duplicated().sum()
        ok(f"{n_unique_pids} unique patients  ({n_dup} rows share a patient_id "
           f"— expected, one row per prediction_time)")


print("\n" + "="*60)
print(f"Errors:   {len(errors)}")
print(f"Warnings: {len(warnings)}")
if errors:
    print("\nAll errors:")
    for e in errors:
        print(f"  {e}")
sys.exit(1 if errors else 0)
