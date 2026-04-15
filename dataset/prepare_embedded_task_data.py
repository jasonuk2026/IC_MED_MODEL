#!/usr/bin/env python3
"""
Precompute padded event embeddings into epoch-shuffled parquet files.

This script is intended to remove online embedding lookup / padding / shuffle
work from training. It reads task parquet files containing `event_ids`, loads
`embeddings.npy`, converts each sample to a fixed-shape padded array, and writes
one parquet per epoch with a different deterministic shuffle order.

Chronological convention is preserved:
  - input event_ids are assumed to be ordered old -> new
  - if a sample is longer than max_events, keep the latest max_events
  - after padding, the left side is zero padding and the right side is the
    real suffix, so within the real suffix events still appear old -> new

Performance notes:
  - padded_eids (N, max_events) int32 matrix is built once (~400 MB for 100k×1024)
    and shared read-only across all epoch threads.
  - Each epoch thread does vectorized embeddings[flat_eids] per chunk — no Python
    loops over rows/events.
  - PyArrow arrays are built from numpy buffers via FixedSizeListArray.from_arrays
    (zero Python-level element iteration).
  - Multiple epochs are written in parallel (--num_workers) since they are
    independent and only read shared immutable data.
  - Use --half to store float16 instead of float32 (~half the output size/time).

Example:
  source ~/miniforge3/bin/activate torch
  python dataset/prepare_embedded_task_data.py \\
    --task new_hyperlipidemia \\
    --input_dir upsampled_data \\
    --output_dir upsampled_assembled_data \\
    --event_embedding_dir encode_events_result/bert \\
    --epochs 5 \\
    --max_events 1024 \\
    --num_workers 5 \\
    --half
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_TASKS = [
    "new_hypertension",
    "new_hyperlipidemia",
    "new_pancan",
    "new_celiac",
    "new_lupus",
    "new_acutemi",
]

# Rows processed per write call. Memory ≈ chunk_size × max_events × event_dim × dtype_bytes.
# 512 × 1024 × 768 × 4 ≈ 1.5 GB/chunk (fp32), × 0.5 if --half.
DEFAULT_CHUNK_SIZE = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_binary_label_array(series: pd.Series) -> np.ndarray:
    arr = series.to_numpy()
    if arr.dtype == np.bool_:
        return arr.astype(np.int8)
    if np.issubdtype(arr.dtype, np.integer):
        return (arr != 0).astype(np.int8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr != 0.0).astype(np.int8)
    # Fallback: element-wise string / object parsing
    out = np.empty(len(arr), dtype=np.int8)
    for i, v in enumerate(arr):
        text = str(v).strip().lower()
        if text in {"true", "t", "yes", "y", "1", "1.0"}:
            out[i] = 1
        elif text in {"false", "f", "no", "n", "0", "0.0"}:
            out[i] = 0
        else:
            try:
                out[i] = int(float(text) != 0.0)
            except ValueError as exc:
                raise ValueError(f"Unsupported binary label value: {v!r}") from exc
    return out


def _pa_float_type(use_half: bool) -> pa.DataType:
    return pa.float16() if use_half else pa.float32()


def build_schema(max_events: int, event_dim: int, use_half: bool = False) -> pa.Schema:
    float_t = _pa_float_type(use_half)
    row_type = pa.list_(float_t, list_size=event_dim)
    emb_type = pa.list_(row_type, list_size=max_events)
    mask_type = pa.list_(pa.int8(), list_size=max_events)
    return pa.schema([
        pa.field("patient_id", pa.int64()),
        pa.field("label", pa.int8()),
        pa.field("num_events", pa.int32()),
        pa.field("event_mask", mask_type),
        pa.field("event_embs", emb_type),
        pa.field("source_row", pa.int32()),
        pa.field("prediction_time", pa.string()),
        pa.field("split", pa.string()),
        pa.field("task_name", pa.string()),
        pa.field("oversample_copy_idx", pa.float32()),
    ])


def infer_max_events_from_train_parquet(input_dir: str, task: str) -> int:
    path = Path(input_dir) / task / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing train parquet for inferring max_events: {path}")
    df = pd.read_parquet(path, columns=["event_ids"])
    if len(df) == 0:
        raise ValueError(f"Empty parquet: {path}")
    return int(df["event_ids"].map(len).max())


# ---------------------------------------------------------------------------
# Core pre-processing: build compact index matrix once
# ---------------------------------------------------------------------------

def build_padded_eids(
    df: pd.DataFrame,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a (N, max_events) int32 matrix of event IDs (right-aligned, zero-padded)
    and a (N,) int32 vector of real event counts.

    Fast path: all rows have exactly max_events events → use np.array() directly.
    General path: variable-length rows need right-alignment + zero-padding.
    """
    N = len(df)
    event_ids_list = df["event_ids"].tolist()

    lengths = np.fromiter((len(e) for e in event_ids_list), dtype=np.int32, count=N)
    real_ns = np.minimum(lengths, max_events)

    if int(lengths.min()) == max_events and int(lengths.max()) == max_events:
        print("  fast path: all rows have exactly max_events events, using np.array()")
        padded_eids = np.array(event_ids_list, dtype=np.int32)
        return padded_eids, real_ns

    padded_eids = np.zeros((N, max_events), dtype=np.int32)
    for i, eids in enumerate(tqdm(event_ids_list, desc="  indexing events", dynamic_ncols=True)):
        n = real_ns[i]
        if n > 0:
            padded_eids[i, max_events - n:] = np.asarray(eids, dtype=np.int32)[-n:]

    return padded_eids, real_ns


def collect_meta(df: pd.DataFrame) -> dict[str, np.ndarray]:
    N = len(df)

    def _str_col(col: str) -> np.ndarray:
        if col in df.columns:
            return df[col].fillna("").astype(str).to_numpy()
        return np.full(N, "", dtype=object)

    def _float_col(col: str) -> np.ndarray:
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float32)
        return np.full(N, np.nan, dtype=np.float32)

    patient_ids = (
        df["patient_id"].to_numpy(dtype=np.int64)
        if "patient_id" in df.columns
        else np.full(N, -1, dtype=np.int64)
    )

    return {
        "patient_id": patient_ids,
        "label": parse_binary_label_array(df["label"]),
        "source_row": np.arange(N, dtype=np.int32),
        "prediction_time": _str_col("prediction_time"),
        "split": _str_col("split"),
        "task_name": _str_col("task_name"),
        "oversample_copy_idx": _float_col("oversample_copy_idx"),
    }


# ---------------------------------------------------------------------------
# Per-epoch writing: chunked vectorized lookup + fast Arrow construction
# ---------------------------------------------------------------------------

def write_epoch_parquet(
    padded_eids: np.ndarray,     # (N, max_events) int32  — shared read-only
    real_ns: np.ndarray,         # (N,) int32
    meta: dict[str, np.ndarray],
    embeddings: np.ndarray,      # (V, event_dim) — shared read-only
    out_path: Path,
    schema: pa.Schema,
    seed: int,
    chunk_size: int,
    use_half: bool,
    epoch_label: str = "",
) -> None:
    N = len(padded_eids)
    max_events = padded_eids.shape[1]
    event_dim = embeddings.shape[1]
    np_float = np.float16 if use_half else np.float32
    pa_float = _pa_float_type(use_half)

    rng = np.random.default_rng(seed)
    order = rng.permutation(N).astype(np.int32)
    col_arange = np.arange(max_events, dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, schema)

    n_chunks = (N + chunk_size - 1) // chunk_size
    prefix = f"[{epoch_label}] " if epoch_label else ""
    for chunk_i in tqdm(range(n_chunks), desc=f"  {prefix}chunks", dynamic_ncols=True, leave=False):
        idx = order[chunk_i * chunk_size : (chunk_i + 1) * chunk_size]
        C = len(idx)

        chunk_eids = padded_eids[idx]       # (C, max_events) int32
        chunk_real_ns = real_ns[idx]         # (C,)

        # Vectorized embedding lookup
        flat_embs = embeddings[chunk_eids.reshape(-1)]  # (C*max_events, D)
        chunk_padded = flat_embs.reshape(C, max_events, event_dim).astype(np_float, copy=False)

        # Vectorized mask: position >= max_events - real_n
        start_cols = (max_events - chunk_real_ns)[:, np.newaxis]   # (C, 1)
        chunk_mask = (col_arange[np.newaxis, :] >= start_cols).astype(np.int8)  # (C, max_events)

        # Zero out padding slots
        chunk_padded *= chunk_mask[:, :, np.newaxis].astype(np_float)

        # Build Arrow arrays from numpy buffers — zero Python iteration
        emb_arr = pa.FixedSizeListArray.from_arrays(
            pa.FixedSizeListArray.from_arrays(
                pa.array(chunk_padded.reshape(-1), type=pa_float),
                list_size=event_dim,
            ),
            list_size=max_events,
        )
        mask_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(chunk_mask.reshape(-1), type=pa.int8()),
            list_size=max_events,
        )

        writer.write_table(pa.table(
            {
                "patient_id": pa.array(meta["patient_id"][idx], type=pa.int64()),
                "label": pa.array(meta["label"][idx], type=pa.int8()),
                "num_events": pa.array(chunk_real_ns, type=pa.int32()),
                "event_mask": mask_arr,
                "event_embs": emb_arr,
                "source_row": pa.array(meta["source_row"][idx], type=pa.int32()),
                "prediction_time": pa.array(meta["prediction_time"][idx].tolist(), type=pa.string()),
                "split": pa.array(meta["split"][idx].tolist(), type=pa.string()),
                "task_name": pa.array(meta["task_name"][idx].tolist(), type=pa.string()),
                "oversample_copy_idx": pa.array(meta["oversample_copy_idx"][idx], type=pa.float32()),
            },
            schema=schema,
        ))

    writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute padded embedded task parquet files with per-epoch shuffles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--input_dir", default="extract_task_data_oversampled/output")
    parser.add_argument("--output_dir", default="embedded_task_data")
    parser.add_argument("--event_embedding_dir", required=True,
                        help="Directory containing embeddings.npy")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of differently shuffled output parquet files to write")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_events", type=int, default=None,
                        help="If unset, infer from <input_dir>/<task>/train.parquet")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Rows per write call. Memory ≈ chunk_size×max_events×event_dim×dtype_bytes")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel epoch writers. Each worker holds one chunk in memory at a time. "
                             "Set to --epochs to write all epochs simultaneously. "
                             "Only beneficial when I/O or CPU (not disk bandwidth) is the bottleneck.")
    parser.add_argument("--half", action="store_true",
                        help="Store embeddings as float16 instead of float32 (~half output size/time)")
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_path = Path(args.event_embedding_dir) / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

    print(f"Loading embeddings from {embeddings_path} ...")
    embeddings = np.load(embeddings_path)
    event_dim = int(embeddings.shape[1])
    print(f"Embeddings shape: {embeddings.shape}  dtype={embeddings.dtype}")

    for task in args.task:
        input_path = Path(args.input_dir) / task / f"{args.split}.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input parquet: {input_path}")

        max_events = (
            args.max_events
            if args.max_events is not None
            else infer_max_events_from_train_parquet(args.input_dir, task)
        )
        schema = build_schema(max_events, event_dim, use_half=args.half)

        dtype_label = "float16" if args.half else "float32"
        bytes_per_epoch = max_events * event_dim * (2 if args.half else 4)

        print("\n" + "─" * 60)
        print(f"Task      : {task}")
        print(f"Split     : {args.split}")
        print(f"Input     : {input_path}")
        print(f"max_events: {max_events}")
        print(f"dtype     : {dtype_label}")
        print(f"chunk_size: {args.chunk_size}  "
              f"(≈{args.chunk_size * max_events * event_dim * (2 if args.half else 4) / 1e9:.2f} GB/chunk)")
        print(f"num_workers: {args.num_workers}")

        df = pd.read_parquet(input_path)
        N = len(df)
        print(f"Loaded {N:,} rows  "
              f"(≈{N * bytes_per_epoch / 1e9:.1f} GB uncompressed per epoch)")

        padded_eids, real_ns = build_padded_eids(df, max_events)
        meta = collect_meta(df)

        pos = int(meta["label"].sum())
        neg = N - pos
        print(f"Labels: {N:,} total  pos={pos:,}  neg={neg:,}  ratio={pos / max(neg, 1):.4f}")

        task_out_dir = Path(args.output_dir) / task
        task_out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "task": task,
            "split": args.split,
            "input_parquet": str(input_path),
            "embeddings_path": str(embeddings_path),
            "epochs": args.epochs,
            "max_events": max_events,
            "event_dim": event_dim,
            "dtype": dtype_label,
            "num_rows": N,
            "num_pos": pos,
            "num_neg": neg,
            "seed": args.seed,
        }
        with open(task_out_dir / f"{args.split}_embedded_meta.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Build epoch params list
        epoch_params = [
            (
                epoch_idx,
                task_out_dir / f"{args.split}_embedded_{epoch_idx:03d}.parquet",
                args.seed + epoch_idx * 1337,
            )
            for epoch_idx in range(args.epochs)
        ]

        num_workers = min(args.num_workers, args.epochs)
        if num_workers <= 1:
            for epoch_idx, out_path, epoch_seed in epoch_params:
                print(f"Epoch {epoch_idx+1}/{args.epochs}  ->  {out_path.name}  (seed={epoch_seed})")
                write_epoch_parquet(
                    padded_eids=padded_eids,
                    real_ns=real_ns,
                    meta=meta,
                    embeddings=embeddings,
                    out_path=out_path,
                    schema=schema,
                    seed=epoch_seed,
                    chunk_size=args.chunk_size,
                    use_half=args.half,
                    epoch_label=f"epoch {epoch_idx+1}/{args.epochs}",
                )
        else:
            print(f"Writing {args.epochs} epochs in parallel with {num_workers} workers ...")

            def _task(epoch_idx, out_path, epoch_seed):
                write_epoch_parquet(
                    padded_eids=padded_eids,
                    real_ns=real_ns,
                    meta=meta,
                    embeddings=embeddings,
                    out_path=out_path,
                    schema=schema,
                    seed=epoch_seed,
                    chunk_size=args.chunk_size,
                    use_half=args.half,
                    epoch_label=f"epoch {epoch_idx+1}/{args.epochs}",
                )
                return epoch_idx

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_task, epoch_idx, out_path, epoch_seed): epoch_idx
                    for epoch_idx, out_path, epoch_seed in epoch_params
                }
                for fut in as_completed(futures):
                    epoch_idx = fut.result()  # re-raises any exception
                    print(f"  epoch {epoch_idx+1}/{args.epochs} done")

        print(f"Done: {args.epochs} parquet(s) -> {task_out_dir}/")


if __name__ == "__main__":
    main()
