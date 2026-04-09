#!/usr/bin/env python3
"""
prepare_task_data_v3.py

Balanced training-batch assembly without patient de-duplication.

Compared with prepare_task_data_v2.py:
  - still parses timelines into event_ids
  - still writes train_prepared_XXX.parquet + .json
  - still builds max_batch_size blocks with half positives / half negatives
  - BUT removes the constraint that patient_id must be unique within a block

This is intended for retrieval-style training where using more rows is often
more valuable than enforcing patient-level uniqueness inside each batch.
"""

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


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


def _is_positive(label_str) -> bool:
    if isinstance(label_str, bool):
        return label_str
    s = str(label_str).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    try:
        return int(float(s)) != 0
    except (ValueError, TypeError):
        return False


def _parse_timeline(timeline: str) -> list[int]:
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
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"{parts[0]}_prepared_{parts[1]}"
    return f"{stem}_prepared"


def process_file(input_path: str, max_batch_size: int, seed: int, num_workers: int) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[SKIP] {input_path} not found.")
        return

    if max_batch_size % 2 != 0:
        raise ValueError(f"--max_batch_size must be even, got {max_batch_size}")

    out_stem = _derive_output_stem(input_path.stem)
    out_parquet = input_path.parent / f"{out_stem}.parquet"
    out_json = input_path.parent / f"{out_stem}.json"

    print(f"\n{'─'*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {out_parquet}")

    print("  Reading parquet …")
    pf = pq.ParquetFile(str(input_path))
    df = pf.read(columns=["patient_id", "label", "timeline", "task_name"]).to_pandas()
    total_rows = len(df)
    print(f"  {total_rows:,} rows total")

    task_names = df["task_name"].dropna().unique().tolist()
    if not task_names:
        print("  [ERROR] No task_name column or all null — skipping.")
        return
    task_name = task_names[0]
    task_idx = TASK_2_IDX.get(task_name)
    if task_idx is None:
        print(f"  [ERROR] Unknown task_name {task_name!r} — skipping.")
        return
    print(f"  Task: {task_name}  (task_idx={task_idx})")

    print("  Parsing timelines …")
    timelines = df["timeline"].fillna("").tolist()
    with mp.Pool(processes=num_workers) as pool:
        chunksize = max(1, len(timelines) // max(num_workers * 10, 1))
        all_eids: list[list[int]] = list(tqdm(
            pool.imap(_parse_timeline, timelines, chunksize=chunksize),
            total=len(timelines),
            desc="    timelines",
            leave=False,
        ))

    records: list[tuple[int, int, int, np.ndarray]] = []
    pos_indices: list[int] = []
    neg_indices: list[int] = []
    for i, row in enumerate(df.itertuples(index=False)):
        pid = int(row.patient_id)
        label = 1 if _is_positive(row.label) else 0
        eids = np.array(all_eids[i], dtype=np.int32)
        records.append((i, pid, label, eids))
        if label == 1:
            pos_indices.append(i)
        else:
            neg_indices.append(i)

    rng = random.Random(seed)
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    print(f"  Pos rows: {len(pos_indices):,}")
    print(f"  Neg rows: {len(neg_indices):,}")

    half = max_batch_size // 2
    n_batches = min(len(pos_indices) // half, len(neg_indices) // half)
    used_pos = n_batches * half
    used_neg = n_batches * half
    used_rows = used_pos + used_neg
    wasted_rows = total_rows - used_rows
    pct_wasted = 100.0 * wasted_rows / max(total_rows, 1)

    print(
        f"  Assembled: {n_batches:,} batches × {max_batch_size} = {used_rows:,} rows used, "
        f"{wasted_rows:,} wasted ({pct_wasted:.1f}%)"
    )

    if n_batches == 0:
        print("  [WARN] No batches assembled — output not written.")
        return

    output_rows: list[dict] = []
    for batch_idx in range(n_batches):
        pos_chunk = pos_indices[batch_idx * half : (batch_idx + 1) * half]
        neg_chunk = neg_indices[batch_idx * half : (batch_idx + 1) * half]
        for i in range(half):
            pos_row_idx = pos_chunk[i]
            neg_row_idx = neg_chunk[i]

            src_row_p, pid_p, _, eids_p = records[pos_row_idx]
            src_row_n, pid_n, _, eids_n = records[neg_row_idx]

            output_rows.append({
                "patient_id": pid_p,
                "task_idx": task_idx,
                "label": 1,
                "event_ids": eids_p,
                "source_row": src_row_p,
            })
            output_rows.append({
                "patient_id": pid_n,
                "task_idx": task_idx,
                "label": 0,
                "event_ids": eids_n,
                "source_row": src_row_n,
            })

    print(f"  Writing {out_parquet} …")
    table = pa.table(
        {
            "patient_id": pa.array([r["patient_id"] for r in output_rows], type=pa.int64()),
            "task_idx": pa.array([r["task_idx"] for r in output_rows], type=pa.int16()),
            "label": pa.array([r["label"] for r in output_rows], type=pa.int8()),
            "event_ids": pa.array([r["event_ids"].tolist() for r in output_rows], type=pa.list_(pa.int32())),
            "source_row": pa.array([r["source_row"] for r in output_rows], type=pa.int32()),
        },
        schema=OUTPUT_SCHEMA,
    )
    pq.write_table(table, str(out_parquet))

    meta = {
        "source_parquet": input_path.name,
        "max_batch_size": max_batch_size,
        "num_batches": n_batches,
        "total_rows": total_rows,
        "used_rows": used_rows,
        "wasted_rows": wasted_rows,
        "patient_deduplicated": False,
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {out_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble balanced training batches without patient de-duplication.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="One or more train_*.parquet files from build_task_data.py.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=64,
        help="Rows per assembled block (half positives, half negatives, patient duplication allowed).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Workers for parallel timeline parsing (default: cpu_count).",
    )
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
