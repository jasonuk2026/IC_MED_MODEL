#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from train_disease_soft_token_classifier import TASK_2_DISEASE_NAME, TASK_2_DISEASE_QUERY_TEXT

GEN_META_DIR = Path(__file__).resolve().parent / "01_gen_meta"
if str(GEN_META_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_META_DIR))

from encoders import get_encoder


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate pure-prompt JSONL files for querying an LLM on EHR interpretation tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_data_dir", default="data/eval_data_latest")
    p.add_argument("--task", required=True, choices=sorted(TASK_2_DISEASE_NAME))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--unique_events_path", default="data/01_outputs/unique_events.parquet")
    p.add_argument("--encoder", default="qwen3")
    p.add_argument("--model_name", default=None,
                   help="Optional encoder model name. Not required when only using the event template.")
    p.add_argument("--template_path", default=None)
    p.add_argument("--max_events", type=int, default=1024)
    p.add_argument("--truncate_side", choices=["first", "last"], default="last")
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--output_path", default=None)
    p.add_argument("--include_label", action="store_true",
                   help="Include the ground-truth label in the output JSONL metadata.")
    return p.parse_args()


def truncate_event_ids(event_ids, max_events: int | None, truncate_side: str):
    if max_events is not None and len(event_ids) > max_events:
        if truncate_side == "last":
            return event_ids[-max_events:]
        return event_ids[:max_events]
    return event_ids


def build_event_text_map(args) -> dict[int, str]:
    encoder = get_encoder(
        args.encoder,
        model_name=args.model_name,
        template_path=args.template_path,
        append_token_name=None,
        append_token_text=None,
        pooling_mode="mean",
    )
    event_df = pd.read_parquet(args.unique_events_path)
    required_cols = {"event_id", "omop_table", "event_type", "code", "description", "value", "unit"}
    missing = sorted(required_cols - set(event_df.columns))
    if missing:
        raise ValueError("Unique events parquet missing columns: %s" % ", ".join(missing))

    event_text_map = {}
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
        event_text_map[int(row.event_id)] = text
    logger.info(
        "Prepared event text map for %d unique events (%d dropped by template).",
        len(event_text_map),
        dropped,
    )
    return event_text_map


def build_prompt(task: str, event_texts: list[str]) -> str:
    disease_name = TASK_2_DISEASE_NAME[task]
    disease_query = TASK_2_DISEASE_QUERY_TEXT[task]
    event_lines = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(event_texts))
    return (
        "You are a careful clinical language model helping interpret a longitudinal EHR.\n\n"
        f"Target disease:\n{disease_query}\n\n"
        "Below is a chronological list of the patient's EHR events from oldest to newest.\n"
        "Use only the evidence in these events.\n\n"
        "Patient EHR events:\n"
        f"{event_lines}\n\n"
        f"Question: Based on the EHR above, does this patient likely have {disease_name}?\n"
        "Please answer in JSON with the following keys:\n"
        '  "answer": "yes" or "no"\n'
        '  "confidence": a number between 0 and 1\n'
        '  "evidence": a short list of the most relevant clues from the EHR\n'
        '  "reasoning": a concise clinical rationale\n'
    )


def main():
    args = parse_args()
    eval_path = Path(args.eval_data_dir) / args.task / f"{args.split}.parquet"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval parquet: {eval_path}")

    output_path = Path(args.output_path) if args.output_path else (
        Path("output/llm_prompts") / f"{args.task}_{args.split}_prompts.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_text_map = build_event_text_map(args)
    df = pd.read_parquet(eval_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    logger.info("Generating prompts from %s (%d rows)", eval_path, len(df))
    count_written = 0
    with output_path.open("w") as f:
        for row in df.itertuples(index=False):
            kept_ids = truncate_event_ids(row.event_ids, args.max_events, args.truncate_side)
            event_texts = [event_text_map[int(eid)] for eid in kept_ids if int(eid) in event_text_map]
            if not event_texts:
                continue
            record = {
                "task": args.task,
                "split": args.split,
                "patient_id": int(row.patient_id),
                "num_events": len(event_texts),
                "prompt": build_prompt(args.task, event_texts),
            }
            if args.include_label:
                record["label"] = int(row.label)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count_written += 1

    logger.info("Wrote %d prompts -> %s", count_written, output_path)


if __name__ == "__main__":
    main()
