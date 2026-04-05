#!/usr/bin/env python3
"""
prepare_task_data.py

Offline preprocessing for one or more EHRSHOT tasks (v6 parquet format).

Reads timeline JSONL from build_task_data.py output, extracts BioLinkBERT
embedding indices, arranges rows with guaranteed pos/neg ratio, interleaves
tasks, and writes N output shards ready for sequential training.

Output schema (per shard):
    task_idx    int16          stable task index (matches TASK_2_IDX in training)
    label       int8           0 or 1
    event_ids   list<int32>    embedding indices extracted from timeline
    source_row  int32          absolute row index in the original per-task parquet
                               (combined with task_idx, uniquely identifies the
                               original build_task_data.py output row for verification)

Train split layout
──────────────────
Rows are arranged so that every group_size = pos_per_group + neg_per_group
consecutive rows contain exactly pos_per_group positives and neg_per_group
negatives.  Groups are interleaved round-robin across tasks, then globally
shuffled, so the sequential read order mixes tasks and is random at the
group level.

Example: 2 tasks, pos_per_group=1, neg_per_group=1, batch_size=32
    → every batch of 32 rows = 16 groups = 16 pos + 16 neg,
      drawn from a mix of tasks

Val / test split layout
────────────────────────
All rows from all tasks combined and shuffled (no ratio constraint).

N output shards
───────────────
The final row sequence is split into --num_shards roughly equal parquet
files: shard_0000.parquet, shard_0001.parquet, ...
Each shard respects the group boundary so pos/neg ratio holds within
each shard as well.

Usage
─────
Single task, 1 shard:
    python prepare_task_data.py --tasks new_hypertension

Multiple tasks, 8 shards:
    python prepare_task_data.py \\
        --tasks new_hypertension new_hyperlipidemia new_lupus \\
        --num_shards 8

All tasks in parallel (one process per task, then merge):
    for task in new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi; do
        python prepare_task_data.py --tasks $task --num_shards 4 &
    done
    wait
"""

import argparse
import json
import multiprocessing as mp
import os
import random
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

SPLITS = ["train", "val", "test"]

OUTPUT_SCHEMA = pa.schema([
    pa.field("task_idx",   pa.int16()),
    pa.field("label",      pa.int8()),
    pa.field("event_ids",  pa.list_(pa.int32())),
    pa.field("source_row", pa.int32()),   # row index in original per-task parquet
])


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _parse_worker(timeline: str) -> list[int]:
    return _parse_timeline(timeline)


# ── Core: read one split from a parquet file ──────────────────────────────────

# Each sample is stored as a named tuple-like dict to keep source_row together
# with the data throughout the pipeline.
#
# Record = {"eids": np.ndarray[int32], "label": int, "source_row": int}

def read_split(
    parquet_path: str,
    target_split: str,
    num_workers: int,
) -> tuple[list[dict], list[dict]]:
    """Stream-read a parquet file; return (pos_records, neg_records).

    Each record is {"eids": np.ndarray[int32], "label": int, "source_row": int}.
    source_row is the absolute row index within the original parquet file.
    """
    pos_records: list[dict] = []
    neg_records: list[dict] = []

    pf      = pq.ParquetFile(parquet_path)
    n_rg    = pf.metadata.num_row_groups
    pool    = mp.Pool(processes=num_workers)
    abs_row = 0   # running absolute row counter across all row groups

    try:
        for batch in tqdm(
            pf.iter_batches(columns=["split", "label", "timeline"]),
            total=n_rg,
            desc=f"    {target_split} row-groups",
            leave=False,
        ):
            splits    = batch.column("split").to_pylist()
            labels    = batch.column("label").to_pylist()
            timelines = batch.column("timeline").to_pylist()

            # Collect (abs_row, label, timeline) for rows in this split
            rows_in_split = []
            for sp, lbl, tl in zip(splits, labels, timelines):
                if sp == target_split:
                    rows_in_split.append((abs_row, lbl, tl if isinstance(tl, str) else ""))
                abs_row += 1

            if not rows_in_split:
                continue

            row_abs_rows  = [r[0] for r in rows_in_split]
            row_labels    = [r[1] for r in rows_in_split]
            row_timelines = [r[2] for r in rows_in_split]

            # Parallel timeline parsing within this row group
            chunksize = max(1, len(row_timelines) // (num_workers * 4))
            parsed    = pool.map(_parse_worker, row_timelines, chunksize=chunksize)

            for src_row, lbl, eids in zip(row_abs_rows, row_labels, parsed):
                record = {
                    "eids":       np.array(eids, dtype=np.int32),
                    "label":      1 if _is_positive(lbl) else 0,
                    "source_row": int(src_row),
                }
                if record["label"] == 1:
                    pos_records.append(record)
                else:
                    neg_records.append(record)
    finally:
        pool.close()
        pool.join()

    return pos_records, neg_records


# ── Arrangement helpers ───────────────────────────────────────────────────────

def arrange_train(
    all_pos: list[dict],
    all_neg: list[dict],
    pos_per_group: int,
    neg_per_group: int,
    rng: random.Random,
) -> list[dict]:
    """Shuffle, form groups with guaranteed ratio, globally shuffle groups.

    Returns a flat list of records in the final output order.
    Excess samples that cannot fill a complete group are discarded.
    """
    rng.shuffle(all_pos)
    rng.shuffle(all_neg)

    n_groups   = min(len(all_pos) // pos_per_group, len(all_neg) // neg_per_group)
    n_drop_pos = len(all_pos) - n_groups * pos_per_group
    n_drop_neg = len(all_neg) - n_groups * neg_per_group
    if n_drop_pos or n_drop_neg:
        print(f"      Dropped {n_drop_pos} pos / {n_drop_neg} neg "
              f"(cannot form complete groups)")

    groups: list[list[dict]] = []
    for i in range(n_groups):
        g = (all_pos[i * pos_per_group : (i + 1) * pos_per_group] +
             all_neg[i * neg_per_group : (i + 1) * neg_per_group])
        groups.append(g)

    rng.shuffle(groups)

    return [rec for g in groups for rec in g]


def interleave_tasks_train(
    task_records: dict[str, list[dict]],
    pos_per_group: int,
    neg_per_group: int,
    rng: random.Random,
) -> list[dict]:
    """Arrange each task independently then interleave groups round-robin.

    Each task's groups are shuffled independently first.  Then groups are
    interleaved round-robin across tasks (task0_g0, task1_g0, task0_g1, …)
    and the resulting group sequence is globally shuffled.
    """
    group_size   = pos_per_group + neg_per_group
    task_groups: dict[str, list[list[dict]]] = {}

    for task, records in task_records.items():
        pos = [r for r in records if r["label"] == 1]
        neg = [r for r in records if r["label"] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)

        n_groups   = min(len(pos) // pos_per_group, len(neg) // neg_per_group)
        n_drop_pos = len(pos) - n_groups * pos_per_group
        n_drop_neg = len(neg) - n_groups * neg_per_group
        if n_drop_pos or n_drop_neg:
            print(f"      [{task}] dropped {n_drop_pos} pos / {n_drop_neg} neg")

        groups = []
        for i in range(n_groups):
            g = (pos[i * pos_per_group : (i + 1) * pos_per_group] +
                 neg[i * neg_per_group : (i + 1) * neg_per_group])
            groups.append(g)
        rng.shuffle(groups)
        task_groups[task] = groups

    # Round-robin interleave across tasks
    all_groups: list[list[dict]] = []
    task_iters = {t: iter(gs) for t, gs in task_groups.items()}
    tasks_sorted = sorted(task_groups.keys())
    while True:
        added_any = False
        for t in tasks_sorted:
            try:
                all_groups.append(next(task_iters[t]))
                added_any = True
            except StopIteration:
                pass
        if not added_any:
            break

    rng.shuffle(all_groups)
    return [rec for g in all_groups for rec in g]


def arrange_eval(
    task_records: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict]:
    """Combine all tasks and shuffle for val/test splits."""
    combined = [rec for records in task_records.values() for rec in records]
    rng.shuffle(combined)
    return combined


# ── Parquet writer ─────────────────────────────────────────────────────────────

def write_shards(
    out_dir: str,
    split: str,
    task_idx_map: dict[str, int],
    all_records: list[dict],
    num_shards: int,
    row_group_size: int,
    group_size: int,
) -> None:
    """Split all_records into num_shards parquet files, respecting group boundaries."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    n_total  = len(all_records)
    # For train, align shard boundaries to group boundaries
    if group_size > 1:
        n_groups       = n_total // group_size
        groups_per_shard = max(1, n_groups // num_shards)
        shard_boundaries = [i * groups_per_shard * group_size for i in range(num_shards)]
        shard_boundaries.append(n_total)
    else:
        rows_per_shard   = max(1, n_total // num_shards)
        shard_boundaries = [i * rows_per_shard for i in range(num_shards)]
        shard_boundaries.append(n_total)

    n_digits = len(str(num_shards - 1))

    for shard_idx in range(num_shards):
        start = shard_boundaries[shard_idx]
        end   = shard_boundaries[shard_idx + 1]
        if start >= n_total:
            break

        shard_records = all_records[start:end]
        fname    = f"shard_{shard_idx:0{n_digits}d}.parquet"
        out_path = os.path.join(out_dir, fname)

        writer = pq.ParquetWriter(out_path, OUTPUT_SCHEMA)
        for rg_start in range(0, len(shard_records), row_group_size):
            chunk = shard_records[rg_start : rg_start + row_group_size]
            n     = len(chunk)
            table = pa.table(
                {
                    "task_idx":   pa.array([rec["task_idx"] for rec in chunk], type=pa.int16()),
                    "label":      pa.array([rec["label"]    for rec in chunk], type=pa.int8()),
                    "event_ids":  pa.array(
                        [rec["eids"].tolist() for rec in chunk],
                        type=pa.list_(pa.int32()),
                    ),
                    "source_row": pa.array([rec["source_row"] for rec in chunk], type=pa.int32()),
                },
                schema=OUTPUT_SCHEMA,
            )
            writer.write_table(table)
        writer.close()

    # Report
    shard_files = sorted(Path(out_dir).glob("shard_*.parquet"))
    rows_written = sum(
        pq.read_metadata(str(f)).num_rows for f in shard_files
        if f.name.startswith("shard_")
    )
    print(f"    {len(shard_files)} shard(s) → {out_dir}  "
          f"({rows_written:,} rows total)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline preprocessing: arrange task data for sequential training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="+", required=True,
                        choices=list(TASK_2_DISEASE_NAME.keys()),
                        help="One or more task names to process together")
    parser.add_argument("--input_dir",      default="EHRSHOT_ASSETS/llm_data_v6",
                        help="Directory containing {task}/{split}.parquet")
    parser.add_argument("--output_dir",     default="data/prepared",
                        help="Output root; split dirs are created inside")
    parser.add_argument("--pos_per_group",  type=int, default=1)
    parser.add_argument("--neg_per_group",  type=int, default=1)
    parser.add_argument("--num_shards",     type=int, default=1,
                        help="Number of output parquet files per split")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--num_workers",    type=int, default=None,
                        help="Workers for parallel timeline parsing "
                             "(default: cpu_count)")
    parser.add_argument("--row_group_size", type=int, default=4096,
                        help="Rows per output parquet row group")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        choices=["train", "val", "test"],
                        help="Which splits to process (default: train only)")
    args = parser.parse_args()

    n_workers  = args.num_workers or mp.cpu_count()
    group_size = args.pos_per_group + args.neg_per_group
    rng        = random.Random(args.seed)

    tag = "_".join(sorted(args.tasks))
    print(f"Tasks : {', '.join(sorted(args.tasks))}")
    print(f"Ratio : {args.pos_per_group}:{args.neg_per_group}  "
          f"(group_size={group_size})")
    print(f"Shards: {args.num_shards}")
    print(f"Workers: {n_workers}")
    print()

    for split in args.splits:
        print(f"── {split} ─────────────────────────────")
        task_records: dict[str, list[dict]] = {}

        for task in sorted(args.tasks):
            in_path = os.path.join(args.input_dir, task, f"{split}.parquet")
            if not os.path.exists(in_path):
                print(f"  [{task}] {in_path} not found — skipping.")
                continue

            print(f"  [{task}] Reading {in_path} …")
            pos_records, neg_records = read_split(in_path, split, n_workers)
            print(f"    {len(pos_records):,} pos  /  {len(neg_records):,} neg")

            # Attach task_idx to each record
            tidx = TASK_2_IDX[task]
            for r in pos_records:
                r["task_idx"] = tidx
            for r in neg_records:
                r["task_idx"] = tidx

            task_records[task] = pos_records + neg_records

        if not task_records:
            print(f"  No data found for split '{split}', skipping.\n")
            continue

        print(f"  Arranging …")
        if split == "train":
            all_records = interleave_tasks_train(
                task_records, args.pos_per_group, args.neg_per_group, rng,
            )
            n_groups = len(all_records) // group_size
            print(f"    {n_groups:,} groups × {group_size} = {len(all_records):,} rows")
        else:
            all_records = arrange_eval(task_records, rng)
            print(f"    {len(all_records):,} rows (shuffled)")

        out_dir = os.path.join(args.output_dir, tag, split)
        write_shards(
            out_dir, split,
            TASK_2_IDX, all_records,
            args.num_shards, args.row_group_size,
            group_size if split == "train" else 1,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
