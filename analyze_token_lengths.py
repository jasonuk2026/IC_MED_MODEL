"""
Analyze token length distribution of EHR embedding parquet data.

Usage:
    python analyze_token_lengths.py \
        --data_path data/embedding_inputs/new_diagnosis/new_hypertension_all_p0.5_m1000_ntrain1000_nval200_ntest200.parquet \
        --model_name Qwen/Qwen3-Embedding-4B
"""

import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from train_embedding import TASK_2_DISEASE_NAME, build_prompt

_tokenizer = None


def _init_worker(model_name: str):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name)


def _count_tokens(args: tuple) -> int:
    task, events = args
    disease_name = TASK_2_DISEASE_NAME.get(task, task)
    text = build_prompt(disease_name, events)
    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    # Verify it loads before spawning workers
    AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading data: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    if args.split != "all":
        df = df[df["split"] == args.split]
    print(f"  {len(df)} samples (split={args.split})")

    print(f"Tokenizing with {args.num_workers} workers...")
    worker_args = list(zip(df["task"], df["events"]))
    for _, events in worker_args:
        assert len(events) <= 1000, f"Event longer than 1000 detected, which shouldn't happen."
    with mp.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(args.model_name,),
    ) as pool:
        lengths = pool.map(_count_tokens, worker_args, chunksize=64)
    lengths = np.array(lengths)

    print("\n── Token length distribution ──────────────────────")
    print(f"  count : {len(lengths)}")
    print(f"  min   : {lengths.min()}")
    print(f"  p25   : {int(np.percentile(lengths, 25))}")
    print(f"  median: {int(np.percentile(lengths, 50))}")
    print(f"  mean  : {lengths.mean():.1f}")
    print(f"  p75   : {int(np.percentile(lengths, 75))}")
    print(f"  p90   : {int(np.percentile(lengths, 90))}")
    print(f"  p95   : {int(np.percentile(lengths, 95))}")
    print(f"  p99   : {int(np.percentile(lengths, 99))}")
    print(f"  max   : {lengths.max()}")

    print("\n── Fraction truncated at common max_seq_length caps ──")
    for cap in [10000, 20000, 30000]:
        pct = (lengths > cap).mean() * 100
        print(f"  > {cap:5d} tokens: {pct:5.1f}% of samples truncated")


if __name__ == "__main__":
    main()
