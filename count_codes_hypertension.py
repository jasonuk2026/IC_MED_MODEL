"""
Count unique codes in the valid input data for hypertension patients.
"Valid input" = all events in ehrshot.csv that occur BEFORE each patient's prediction_time.
"""

import pandas as pd
from collections import Counter, defaultdict

# ── 1. Load hypertension labels ───────────────────────────────────────────────
labels = pd.read_csv("EHRSHOT_ASSETS/benchmark/new_hypertension/labeled_patients.csv",
                     parse_dates=["prediction_time"])
print(f"Hypertension labels: {len(labels):,} rows, {labels['patient_id'].nunique():,} unique patients")
print(f"Positive: {labels['value'].sum():,}  Negative: {(~labels['value']).sum():,}\n")

# For each patient, the valid cutoff is their EARLIEST prediction_time
# (all events before ANY prediction are valid context for that patient)
# More precisely: each (patient, prediction_time) pair has its own valid window.
# We use the max prediction_time per patient to capture all possibly-valid events.
patient_max_pred = labels.groupby("patient_id")["prediction_time"].max()
patient_ids = set(patient_max_pred.index)

# ── 2. Stream ehrshot.csv, filter to hypertension patients & before cutoff ────
CHUNKSIZE = 500_000
code_counter = Counter()
total_events = 0
skipped = 0

print("Scanning ehrshot.csv ...")
for chunk in pd.read_csv("EHRSHOT_ASSETS/data/ehrshot.csv",
                         parse_dates=["start"],
                         chunksize=CHUNKSIZE,
                         usecols=["patient_id", "start", "code"]):
    # filter to hypertension patients
    chunk = chunk[chunk["patient_id"].isin(patient_ids)].copy()
    if chunk.empty:
        continue

    # filter: event must be BEFORE the patient's max prediction_time
    chunk["cutoff"] = chunk["patient_id"].map(patient_max_pred)
    valid = chunk[chunk["start"] < chunk["cutoff"]]

    code_counter.update(valid["code"])
    total_events += len(valid)
    skipped += len(chunk) - len(valid)

print(f"Valid events:  {total_events:,}")
print(f"Skipped (after cutoff): {skipped:,}")
print(f"\nTotal unique codes: {len(code_counter):,}\n")

# ── 3. Break down by vocabulary ───────────────────────────────────────────────
vocab_unique  = defaultdict(set)
vocab_occ     = defaultdict(int)
for code, cnt in code_counter.items():
    vocab = code.split("/")[0] if "/" in code else "OTHER"
    vocab_unique[vocab].add(code)
    vocab_occ[vocab] += cnt

print(f"{'Vocabulary':<25} {'Unique codes':>12} {'Occurrences':>14}")
print("-" * 53)
for vocab in sorted(vocab_unique, key=lambda v: -len(vocab_unique[v])):
    print(f"{vocab:<25} {len(vocab_unique[vocab]):>12,} {vocab_occ[vocab]:>14,}")

# ── 4. Top-20 most frequent codes ─────────────────────────────────────────────
print("\nTop 20 most frequent codes:")
print(f"{'Code':<35} {'Count':>10}")
print("-" * 47)
for code, cnt in code_counter.most_common(20):
    print(f"{code:<35} {cnt:>10,}")
