#!/usr/bin/env python3
"""
prepare_task_data_v2.py

Assembles patient-de-duplicated training batches from build_task_data.py output.

For each input parquet (e.g. train_000.parquet):
  - Parses all timelines to extract event_ids.
  - Groups records by (label, patient_id).
  - Assembles blocks of max_batch_size rows where:
      even indices (0, 2, 4, …) = positive samples
      odd  indices (1, 3, 5, …) = negative samples
    Within each block all positive patients are unique, all negative patients
    are unique, and no patient appears in both halves.
  - A record (row) is never reused across blocks. Once a patient's record is
    consumed, if they still have unused records they remain available for
    future blocks.
  - Stops when fewer than half unique patients with unused records are available
    for either class.
  - Writes train_prepared_000.parquet + train_prepared_000.json alongside the
    input file.

Output parquet schema:
    patient_id  int64           for traceability
    task_idx    int16           stable task index (matches training script)
    label       int8            0 = negative, 1 = positive
    event_ids   list<int32>     embedding indices parsed from timeline
    source_row  int32           row index in the source train_*.parquet

JSON metadata (same directory, same stem + .json):
    {
        "source_parquet":  "train_000.parquet",
        "max_batch_size":  64,
        "num_batches":     150,
        "total_rows":      10000,
        "used_rows":       9600,
        "wasted_rows":     400
    }

Usage:
    python prepare_task_data_v2.py \\
        --input_paths EHRSHOT_ASSETS/llm_data_v6/new_acutemi/train_000.parquet \\
                      EHRSHOT_ASSETS/llm_data_v6/new_lupus/train_000.parquet \\
        --max_batch_size 64 --seed 42 --num_workers 16
"""

import argparse
import json
import multiprocessing as mp
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Task constants (must stay in sync with training script) ───────────────────

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}
TASK_2_IDX: dict[str, int] = {t: i for i, t in enumerate(sorted(TASK_2_DISEASE_NAME))}

OUTPUT_SCHEMA = pa.schema([
    pa.field("patient_id", pa.int64()),
    pa.field("task_idx",   pa.int16()),
    pa.field("label",      pa.int8()),
    pa.field("event_ids",  pa.list_(pa.int32())),
    pa.field("source_row", pa.int32()),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_positive(label_str) -> bool:
    if isinstance(label_str, bool):
        return label_str
    s = str(label_str).strip().lower()
    if s == "true":  return True
    if s == "false": return False
    try:
        return int(float(s)) != 0
    except (ValueError, TypeError):
        return False


def _parse_timeline(timeline: str) -> list[int]:
    """Parse JSONL timeline → list of embedding_idx ints."""
    eids: list[int] = []
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


def _derive_output_stem(stem: str) -> str:
    """
    'train_000'  → 'train_prepared_000'
    'train_001'  → 'train_prepared_001'
    Falls back to '{stem}_prepared' if the stem doesn't end with a numeric suffix.
    """
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"{parts[0]}_prepared_{parts[1]}"
    return f"{stem}_prepared"


# ── Core processing ───────────────────────────────────────────────────────────

def process_file(input_path: str, max_batch_size: int, seed: int, num_workers: int) -> None:
    """Process one input parquet → write prepared parquet + JSON metadata."""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[SKIP] {input_path} not found.")
        return

    out_stem    = _derive_output_stem(input_path.stem)
    out_parquet = input_path.parent / f"{out_stem}.parquet"
    out_json    = input_path.parent / f"{out_stem}.json"

    print(f"\n{'─'*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {out_parquet}")

    # ── Read input ────────────────────────────────────────────────────────────
    print("  Reading parquet …")
    pf = pq.ParquetFile(str(input_path))
    # Read only needed columns
    df = pf.read(columns=["patient_id", "label", "timeline", "task_name"]).to_pandas()
    total_rows = len(df)
    print(f"  {total_rows:,} rows total")

    # ── Resolve task_idx ──────────────────────────────────────────────────────
    task_names = df["task_name"].dropna().unique().tolist()
    if not task_names:
        print("  [ERROR] No task_name column or all null — skipping.")
        return
    task_name = task_names[0]
    task_idx  = TASK_2_IDX.get(task_name)
    if task_idx is None:
        print(f"  [ERROR] Unknown task_name {task_name!r} — skipping.")
        return
    print(f"  Task: {task_name}  (task_idx={task_idx})")

    # ── Parse timelines in parallel ───────────────────────────────────────────
    print("  Parsing timelines …")
    timelines = df["timeline"].fillna("").tolist()
    with mp.Pool(processes=num_workers) as pool:
        chunksize = max(1, len(timelines) // (num_workers * 10))
        all_eids: list[list[int]] = list(tqdm(
            pool.imap(_parse_timeline, timelines, chunksize=chunksize),
            total=len(timelines), desc="    timelines", leave=False,
        ))

    # ── Build record list ─────────────────────────────────────────────────────
    # Each entry: (source_row, patient_id, label, event_ids_array)
    records: list[tuple[int, int, int, np.ndarray]] = []
    for i, row in enumerate(df.itertuples(index=False)):
        pid   = int(row.patient_id)
        label = 1 if _is_positive(row.label) else 0
        eids  = np.array(all_eids[i], dtype=np.int32)
        records.append((i, pid, label, eids))

    # ── Group by (label, patient_id) → available record index lists ───────────
    # pos_avail / neg_avail: patient_id → list of indices into `records` (shuffled)
    pos_avail: dict[int, list[int]] = defaultdict(list)
    neg_avail: dict[int, list[int]] = defaultdict(list)

    for ri, (_, pid, label, _) in enumerate(records):
        (pos_avail if label == 1 else neg_avail)[pid].append(ri)

    rng = random.Random(seed)
    for idxs in pos_avail.values():
        rng.shuffle(idxs)
    for idxs in neg_avail.values():
        rng.shuffle(idxs)

    n_pos_patients = len(pos_avail)
    n_neg_patients = len(neg_avail)
    print(f"  Pos patients: {n_pos_patients:,}  (records: {sum(len(v) for v in pos_avail.values()):,})")
    print(f"  Neg patients: {n_neg_patients:,}  (records: {sum(len(v) for v in neg_avail.values()):,})")

    half = max_batch_size // 2
    output_rows: list[dict] = []
    n_batches  = 0
    used_rows  = 0

    # ── Assemble blocks ───────────────────────────────────────────────────────
    while True:
        # Available pos patients with ≥1 unused record
        pos_pids_avail = [pid for pid, idxs in pos_avail.items() if idxs]
        if len(pos_pids_avail) < half:
            break  # can't fill half positive patients

        sampled_pos_pids = rng.sample(pos_pids_avail, half)
        sampled_pos_set  = set(sampled_pos_pids)

        # Available neg patients with ≥1 unused record, excluding sampled pos patients
        neg_pids_avail = [
            pid for pid, idxs in neg_avail.items()
            if idxs and pid not in sampled_pos_set
        ]
        if len(neg_pids_avail) < half:
            break  # can't fill half negative patients (or overlap conflict)

        sampled_neg_pids = rng.sample(neg_pids_avail, half)

        # Build interleaved block: [pos0, neg0, pos1, neg1, ...]
        for i in range(half):
            pos_pid = sampled_pos_pids[i]
            neg_pid = sampled_neg_pids[i]

            pos_ri = pos_avail[pos_pid].pop(0)  # pre-shuffled → random pick
            neg_ri = neg_avail[neg_pid].pop(0)

            src_row_p, pid_p, _, eids_p = records[pos_ri]
            src_row_n, pid_n, _, eids_n = records[neg_ri]

            output_rows.append({
                "patient_id": pid_p,
                "task_idx":   task_idx,
                "label":      1,
                "event_ids":  eids_p,
                "source_row": src_row_p,
            })
            output_rows.append({
                "patient_id": pid_n,
                "task_idx":   task_idx,
                "label":      0,
                "event_ids":  eids_n,
                "source_row": src_row_n,
            })

        n_batches += 1
        used_rows += max_batch_size

        # Remove exhausted patients from availability
        for pid in sampled_pos_pids:
            if not pos_avail[pid]:
                del pos_avail[pid]
        for pid in sampled_neg_pids:
            if not neg_avail[pid]:
                del neg_avail[pid]

    wasted_rows = total_rows - used_rows
    pct_wasted  = 100.0 * wasted_rows / max(total_rows, 1)
    print(f"  Assembled: {n_batches:,} batches × {max_batch_size} = {used_rows:,} rows used, "
          f"{wasted_rows:,} wasted ({pct_wasted:.1f}%)")

    if n_batches == 0:
        print("  [WARN] No batches assembled — output not written.")
        return

    # ── Write parquet ─────────────────────────────────────────────────────────
    print(f"  Writing {out_parquet} …")
    table = pa.table(
        {
            "patient_id": pa.array([r["patient_id"] for r in output_rows], type=pa.int64()),
            "task_idx":   pa.array([r["task_idx"]   for r in output_rows], type=pa.int16()),
            "label":      pa.array([r["label"]       for r in output_rows], type=pa.int8()),
            "event_ids":  pa.array(
                [r["event_ids"].tolist() for r in output_rows],
                type=pa.list_(pa.int32()),
            ),
            "source_row": pa.array([r["source_row"] for r in output_rows], type=pa.int32()),
        },
        schema=OUTPUT_SCHEMA,
    )
    pq.write_table(table, str(out_parquet), row_group_size=max_batch_size)

    # ── Write JSON metadata ───────────────────────────────────────────────────
    meta = {
        "source_parquet": input_path.name,
        "max_batch_size": max_batch_size,
        "num_batches":    n_batches,
        "total_rows":     total_rows,
        "used_rows":      used_rows,
        "wasted_rows":    wasted_rows,
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {out_json}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Assemble patient-de-duplicated training batches from build_task_data.py output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_paths", nargs="+", required=True,
        help="One or more train_*.parquet files from build_task_data.py.",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=64,
        help="Rows per assembled block (half pos, half neg, all from unique patients).",
    )
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Workers for parallel timeline parsing (default: cpu_count).")
    args = parser.parse_args()

    n_workers = args.num_workers or mp.cpu_count()
    print(f"max_batch_size={args.max_batch_size}  seed={args.seed}  workers={n_workers}")
    print(f"Processing {len(args.input_paths)} file(s) …")

    for path in args.input_paths:
        process_file(path, args.max_batch_size, args.seed, n_workers)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
