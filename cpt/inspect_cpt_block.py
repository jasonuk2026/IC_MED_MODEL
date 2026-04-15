#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", default="EHRSHOT_ASSETS/cpt_blocks/qwen3_0.6b_block2048.parquet")
    p.add_argument("--tokenizer_name", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--row_idx", type=int, default=0, help="Global row index to inspect")
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    pf = pq.ParquetFile(args.data_path)
    total_rows = pf.metadata.num_rows
    if not (0 <= args.row_idx < total_rows):
        raise IndexError(f"row_idx={args.row_idx} is out of range for {total_rows} rows")

    row_group_offsets = []
    total = 0
    for i in range(pf.metadata.num_row_groups):
        row_group_offsets.append(total)
        total += pf.metadata.row_group(i).num_rows

    rg_idx = 0
    for i, offset in enumerate(row_group_offsets):
        next_offset = row_group_offsets[i + 1] if i + 1 < len(row_group_offsets) else total_rows
        if offset <= args.row_idx < next_offset:
            rg_idx = i
            break
    row_in_rg = args.row_idx - row_group_offsets[rg_idx]

    table = pf.read_row_group(rg_idx, columns=["patient_id", "block_idx", "num_tokens", "input_ids"])
    record = {
        "patient_id": table.column("patient_id")[row_in_rg].as_py(),
        "block_idx": table.column("block_idx")[row_in_rg].as_py(),
        "num_tokens": table.column("num_tokens")[row_in_rg].as_py(),
        "input_ids": table.column("input_ids")[row_in_rg].as_py(),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        local_files_only=args.local_files_only,
    )
    decoded = tokenizer.decode(record["input_ids"], skip_special_tokens=False)

    print(f"data_path: {args.data_path}")
    print(f"tokenizer_name: {args.tokenizer_name}")
    print(f"row_idx: {args.row_idx}")
    print(f"patient_id: {record['patient_id']}")
    print(f"block_idx: {record['block_idx']}")
    print(f"num_tokens: {record['num_tokens']}")
    print(f"input_ids_len: {len(record['input_ids'])}")
    print("\n--- Decoded Block ---")
    print(decoded)
    print("--- End Block ---")


if __name__ == "__main__":
    main()
