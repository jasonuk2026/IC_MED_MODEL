#!/usr/bin/env python3
"""
spot_check_eval_data.py

Spot-checks build_eval_task_data.py output by cross-referencing with
labeled_patients.csv (for label correctness) and the raw ehrshot.csv
(to verify that event_ids are valid embedding indices for events that
actually existed before the prediction_time for that patient).

For each sampled row:
  1. Find the matching record in labeled_patients.csv (patient_id + task)
     and verify the label matches.
  2. Load event_index.parquet and verify every event_id in the row is a
     valid embedding index.
  3. Verify that event_ids come exclusively from events before prediction_time
     (sampled from the pool — we can't reproduce the exact random draw, but
     we verify containment: every event_id must correspond to a valid
     pre-prediction_time, non-condition_occurrence event for that patient).
"""

import random
import sys
from pathlib import Path

import pandas as pd

EVAL_DIR       = "EHRSHOT_ASSETS/llm_eval_data"
BENCHMARK_DIR  = "EHRSHOT_ASSETS/benchmark"
EMBED_DIR      = "data/biolinkbert_embeddings"
EHR_PATH       = "EHRSHOT_ASSETS/data/ehrshot.csv"
SPLITS_PATH    = "EHRSHOT_ASSETS/splits/person_id_map.csv"
N_SAMPLES      = 30   # rows to check per task per split
SEED           = 42

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}
TASK_2_IDX = {t: i for i, t in enumerate(sorted(TASK_2_DISEASE_NAME))}


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


def normalise_event_key(code, value, unit) -> tuple:
    return (
        str(code  if code  is not None else "").strip(),
        str(value if value is not None else "").strip(),
        str(unit  if unit  is not None else "").strip(),
    )


# ── Load shared resources ────────────────────────────────────────────────────
print("Loading embedding index …")
df_eidx = pd.read_parquet(
    f"{EMBED_DIR}/event_index.parquet",
    columns=["event_id", "code", "value", "unit"],
)
embed_idx: dict[tuple, int] = {}
for row in df_eidx.itertuples(index=False):
    key = normalise_event_key(row.code, row.value, row.unit)
    embed_idx[key] = int(row.event_id)
valid_eids = set(embed_idx.values())
print(f"  {len(valid_eids):,} valid embedding indices\n")

print("Loading raw EHR data …")
df_ehr = pd.read_csv(EHR_PATH, low_memory=False, dtype={"value": str, "unit": str})
if df_ehr.columns[0].startswith("Unnamed"):
    df_ehr = df_ehr.drop(columns=[df_ehr.columns[0]])
df_ehr["start"] = pd.to_datetime(df_ehr["start"])
ehr_by_patient: dict[int, pd.DataFrame] = dict(list(df_ehr.groupby("patient_id")))
print(f"  {len(df_ehr):,} events, {len(ehr_by_patient):,} patients\n")

print("Loading patient splits …")
df_splits    = pd.read_csv(SPLITS_PATH)
pid_to_split = dict(zip(df_splits["omop_person_id"], df_splits["split"]))

# ── Per-task checks ───────────────────────────────────────────────────────────
rng = random.Random(SEED)
errors = []
n_pass = n_fail = 0

for task in sorted(TASK_2_DISEASE_NAME):
    labels_path = f"{BENCHMARK_DIR}/{task}/labeled_patients.csv"
    df_labels   = pd.read_csv(labels_path)
    df_labels["prediction_time"] = pd.to_datetime(df_labels["prediction_time"])
    # Build lookup: (patient_id, prediction_time) → label
    label_lookup = {
        (int(r.patient_id), r.prediction_time): (1 if _is_positive(r.value) else 0)
        for r in df_labels.itertuples(index=False)
    }

    for split in ["val", "test"]:
        fpath = f"{EVAL_DIR}/{task}/{split}.parquet"
        if not Path(fpath).exists():
            continue

        df_eval = pd.read_parquet(fpath)
        sample  = df_eval.sample(min(N_SAMPLES, len(df_eval)), random_state=SEED)
        print(f"[{task}/{split}] checking {len(sample)} rows …")

        for row in sample.itertuples(index=False):
            pid        = int(row.patient_id)
            label_eval = int(row.label)
            eids_eval  = list(row.event_ids)
            pred_time  = df_labels.loc[df_labels["patient_id"] == pid, "prediction_time"]

            # ── 1. Label check via labeled_patients.csv ───────────────────────
            # Find prediction_times for this patient in this split
            patient_labels = df_labels[df_labels["patient_id"] == pid]
            # There may be multiple prediction_times; at least one must match the label
            matching = [
                (1 if _is_positive(r.value) else 0)
                for r in patient_labels.itertuples(index=False)
            ]
            if label_eval not in matching:
                msg = (f"  [FAIL] {task}/{split} pid={pid}: "
                       f"label={label_eval} not found in labeled_patients.csv "
                       f"(got {matching})")
                errors.append(msg)
                print(msg)
                n_fail += 1
                continue

            # ── 2. event_ids are all valid embedding indices ──────────────────
            bad_eids = [e for e in eids_eval if e not in valid_eids]
            if bad_eids:
                msg = (f"  [FAIL] {task}/{split} pid={pid}: "
                       f"{len(bad_eids)} event_ids not in embedding index: "
                       f"{bad_eids[:5]}")
                errors.append(msg)
                print(msg)
                n_fail += 1
                continue

            # ── 3. event_ids must be reachable from patient's EHR events ─────
            # For each event_id, check it can be derived from a pre-prediction-time
            # non-condition_occurrence event for this patient.
            patient_events = ehr_by_patient.get(pid)
            if patient_events is None:
                msg = f"  [FAIL] {task}/{split} pid={pid}: patient not found in EHR"
                errors.append(msg)
                print(msg)
                n_fail += 1
                continue

            # Build set of reachable embedding indices for this patient
            # (union across all prediction_times — we don't store which pred_time
            # was used in the eval row, so we take the union)
            reachable_eids: set[int] = set()
            for pred_t in patient_labels["prediction_time"]:
                evs = patient_events[patient_events["start"] < pred_t]
                for ev in evs.itertuples(index=False):
                    if ev.omop_table == "condition_occurrence":
                        continue
                    val  = ev.value if isinstance(ev.value, str) else ""
                    unit = ev.unit  if isinstance(ev.unit,  str) else ""
                    key  = normalise_event_key(ev.code, val, unit)
                    eid  = embed_idx.get(key)
                    if eid is not None:
                        reachable_eids.add(eid)

            unreachable = [e for e in eids_eval if e not in reachable_eids]
            if unreachable:
                msg = (f"  [FAIL] {task}/{split} pid={pid}: "
                       f"{len(unreachable)}/{len(eids_eval)} event_ids not reachable "
                       f"from patient's pre-prediction EHR events: {unreachable[:5]}")
                errors.append(msg)
                print(msg)
                n_fail += 1
                continue

            n_pass += 1

    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("="*60)
print(f"Passed: {n_pass} / {n_pass + n_fail}")
if errors:
    print(f"\nFailures ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("All spot-checks passed.")
    sys.exit(0)
