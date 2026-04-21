#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from encoders import get_encoder


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Tokenize all unique events and summarize token-length statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--unique_events_path", default="data/01_outputs/unique_events.parquet")
    p.add_argument("--encoder", default="qwen3")
    p.add_argument("--model_name", default=None,
                   help="Base model name for encoder defaults. If omitted, encoder default is used.")
    p.add_argument("--tokenizer_name", default=None,
                   help="Tokenizer source. Defaults to encoder/model_name.")
    p.add_argument("--template_path", default=None)
    p.add_argument("--append_token_name", default=None)
    p.add_argument("--append_token_text", default=None)
    p.add_argument("--pooling_mode", default="mean", choices=["mean", "suffix_only", "all_suffix_mean"])
    p.add_argument("--top_k", type=int, default=20,
                   help="How many longest events to print/save.")
    p.add_argument("--quantiles", nargs="+", type=float, default=[0.5, 0.9, 0.95, 0.99, 0.999])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--output_json", default=None,
                   help="Optional JSON path to save summary statistics.")
    p.add_argument("--output_csv", default=None,
                   help="Optional CSV path to save the top-k longest events.")
    return p.parse_args()


def render_event_texts(event_df: pd.DataFrame, encoder) -> pd.DataFrame:
    rows = []
    dropped = 0
    for row in event_df.itertuples(index=False):
        text = encoder.format_event_text(
            code=str(row.code or "").strip(),
            description=str(row.description or "").strip(),
            value=str(row.value or "").strip(),
            unit=str(row.unit or "").strip(),
            omop_table=str(row.omop_table or "").strip(),
            event_type=str(row.event_type or "").strip(),
        )
        if text is None:
            dropped += 1
            continue
        rows.append({
            "event_id": int(row.event_id),
            "omop_table": str(row.omop_table or "").strip(),
            "event_type": str(row.event_type or "").strip(),
            "code": str(row.code or "").strip(),
            "description": str(row.description or "").strip(),
            "value": str(row.value or "").strip(),
            "unit": str(row.unit or "").strip(),
            "event_text": text,
        })
    logger.info("Rendered %d encodable events (%d dropped by template).", len(rows), dropped)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    encoder = get_encoder(
        args.encoder,
        model_name=args.model_name,
        template_path=args.template_path,
        append_token_name=args.append_token_name,
        append_token_text=args.append_token_text,
        pooling_mode=args.pooling_mode,
    )
    tokenizer_source = args.tokenizer_name or encoder.model_name
    tokenizer_kwargs = {"local_files_only": True} if args.local_files_only else {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    encoder.resolve_append_token(tokenizer)

    logger.info("Loading unique events from %s", args.unique_events_path)
    event_df = pd.read_parquet(args.unique_events_path)
    rendered_df = render_event_texts(event_df, encoder)
    if rendered_df.empty:
        raise ValueError("No encodable events found.")

    lengths = []
    token_examples = []
    for row in rendered_df.itertuples(index=False):
        input_ids = tokenizer(
            row.event_text,
            add_special_tokens=encoder.ADD_SPECIAL_TOKENS,
            return_attention_mask=False,
            truncation=False,
        )["input_ids"]
        token_len = len(input_ids)
        lengths.append(token_len)
        token_examples.append({
            "event_id": row.event_id,
            "omop_table": row.omop_table,
            "event_type": row.event_type,
            "code": row.code,
            "description": row.description,
            "value": row.value,
            "unit": row.unit,
            "event_text": row.event_text,
            "token_length": token_len,
        })

    lengths_np = np.asarray(lengths, dtype=np.int32)
    summary = {
        "num_events": int(lengths_np.size),
        "min_length": int(lengths_np.min()),
        "max_length": int(lengths_np.max()),
        "mean_length": float(lengths_np.mean()),
        "std_length": float(lengths_np.std()),
        "tokenizer_name": tokenizer_source,
        "encoder": args.encoder,
        "append_token_name": args.append_token_name,
        "append_token_text": encoder.append_token_text,
        "add_special_tokens": bool(encoder.ADD_SPECIAL_TOKENS),
        "quantiles": {
            str(q): float(np.quantile(lengths_np, q)) for q in args.quantiles
        },
    }

    logger.info("Tokenizer: %s", tokenizer_source)
    logger.info(
        "Lengths: n=%d min=%d mean=%.2f std=%.2f max=%d",
        summary["num_events"],
        summary["min_length"],
        summary["mean_length"],
        summary["std_length"],
        summary["max_length"],
    )
    for q in args.quantiles:
        logger.info("  q=%.3f -> %.2f tokens", q, summary["quantiles"][str(q)])

    longest = sorted(token_examples, key=lambda x: (-x["token_length"], x["event_id"]))[: args.top_k]
    logger.info("Top %d longest events:", len(longest))
    for item in longest:
        logger.info(
            "  event_id=%d len=%d code=%s value=%s unit=%s text=%r",
            item["event_id"],
            item["token_length"],
            item["code"],
            item["value"],
            item["unit"],
            item["event_text"],
        )

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary JSON -> %s", output_json)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(longest).to_csv(output_csv, index=False)
        logger.info("Saved top-k CSV -> %s", output_csv)


if __name__ == "__main__":
    main()
