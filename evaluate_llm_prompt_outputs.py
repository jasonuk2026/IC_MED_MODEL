#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


YES_SET = {"yes", "y", "true", "positive", "1"}
NO_SET = {"no", "n", "false", "negative", "0"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LLM prompt-query outputs against labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_path", required=True, help="JSONL output from query_openai_compatible_prompts.py")
    p.add_argument("--output_csv", default=None, help="Optional CSV with parsed predictions")
    p.add_argument("--output_json", default=None, help="Optional JSON summary")
    return p.parse_args()


def normalize_answer(x: str | None) -> int | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in YES_SET:
        return 1
    if s in NO_SET:
        return 0
    return None


def try_parse_json_answer(text: str) -> tuple[int | None, str]:
    try:
        obj = json.loads(text)
    except Exception:
        return None, "json_decode_failed"

    if isinstance(obj, dict):
        pred = normalize_answer(obj.get("final_answer"))
        if pred is not None:
            return pred, "json_final_answer"
        pred = normalize_answer(obj.get("answer"))
        if pred is not None:
            return pred, "json_answer"
    return None, "json_missing_answer"


def try_parse_regex_answer(text: str) -> tuple[int | None, str]:
    patterns = [
        (r'"final_answer"\s*:\s*"(yes|no)"', "regex_final_answer"),
        (r'"answer"\s*:\s*"(yes|no)"', "regex_answer"),
        (r"\bfinal_answer\b\s*[:=]\s*(yes|no)\b", "regex_final_answer_loose"),
        (r"\banswer\b\s*[:=]\s*(yes|no)\b", "regex_answer_loose"),
    ]
    for pattern, source in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            pred = normalize_answer(m.group(1))
            if pred is not None:
                return pred, source
    return None, "regex_failed"


def parse_prediction(text: str | None) -> tuple[int | None, str]:
    if text is None or not str(text).strip():
        return None, "empty_response"
    text = str(text).strip()
    pred, source = try_parse_json_answer(text)
    if pred is not None:
        return pred, source
    pred, source = try_parse_regex_answer(text)
    if pred is not None:
        return pred, source
    return None, "unparsed"


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    rows = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    parsed_rows = []
    for row in rows:
        label = row.get("label")
        pred, parse_source = parse_prediction(row.get("response_text"))
        parsed_rows.append(
            {
                "patient_id": row.get("patient_id"),
                "task": row.get("task"),
                "split": row.get("split"),
                "label": label,
                "prediction": pred,
                "correct": (pred == int(label)) if (pred is not None and label is not None) else None,
                "parse_source": parse_source,
                "error": row.get("error"),
            }
        )

    df = pd.DataFrame(parsed_rows)
    has_label = df["label"].notna()
    has_pred = df["prediction"].notna()
    valid = df[has_label & has_pred].copy()

    summary = {
        "num_rows": int(len(df)),
        "num_with_label": int(has_label.sum()),
        "num_with_prediction": int(has_pred.sum()),
        "num_valid_for_accuracy": int(len(valid)),
        "num_unparsed": int((df["prediction"].isna()).sum()),
        "parse_source_counts": df["parse_source"].value_counts(dropna=False).to_dict(),
        "error_count": int(df["error"].notna().sum()),
    }

    if len(valid) > 0:
        valid["label"] = valid["label"].astype(int)
        valid["prediction"] = valid["prediction"].astype(int)
        accuracy = float((valid["label"] == valid["prediction"]).mean())
        tp = int(((valid["label"] == 1) & (valid["prediction"] == 1)).sum())
        tn = int(((valid["label"] == 0) & (valid["prediction"] == 0)).sum())
        fp = int(((valid["label"] == 0) & (valid["prediction"] == 1)).sum())
        fn = int(((valid["label"] == 1) & (valid["prediction"] == 0)).sum())
        summary.update(
            {
                "accuracy": accuracy,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "label_counts": valid["label"].value_counts().to_dict(),
                "prediction_counts": valid["prediction"].value_counts().to_dict(),
            }
        )

    logger.info("Input: %s", input_path)
    for k, v in summary.items():
        logger.info("%s: %s", k, v)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info("Saved parsed rows -> %s", output_csv)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary -> %s", output_json)


if __name__ == "__main__":
    main()
