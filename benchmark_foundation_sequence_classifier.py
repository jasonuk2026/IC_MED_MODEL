#!/usr/bin/env python3

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import benchmark_foundation_simple_classifier as bench


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
        description="Benchmark foundation models by encoding the full chronological event sequence per patient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help="HF model names or local checkpoint paths to benchmark.",
    )
    p.add_argument("--model_labels", nargs="+", default=None,
                   help="Optional labels for --model_paths. Must have same length when provided.")
    p.add_argument("--tokenizer_name", default=None,
                   help="Optional tokenizer source shared across model paths. Defaults to each model path.")
    p.add_argument("--encoder", default="qwen3")
    p.add_argument("--template_path", default=None)
    p.add_argument("--append_token_name", default="pad_token")
    p.add_argument("--append_token_text", default=None)
    p.add_argument("--pool_max_tokens", type=int, default=None)
    p.add_argument("--pooling_mode", default="mean", choices=["mean", "suffix_only", "all_suffix_mean"])
    p.add_argument("--unique_events_path", default="data/01_outputs/unique_events.parquet")
    p.add_argument("--eval_data_dir", default="data/eval_data_latest")
    p.add_argument("--tasks", nargs="+", default=bench.TASKS, choices=bench.TASKS)
    p.add_argument("--train_split", default="val", choices=["val", "test"])
    p.add_argument("--test_split", default="test", choices=["val", "test"])
    p.add_argument("--max_events", type=int, default=10)
    p.add_argument("--truncate_side", default="last", choices=["first", "last"],
                   help="Which side of the event list to keep before concatenation.")
    p.add_argument("--event_separator", default="")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Sequence encoding batch size.")
    p.add_argument("--classifier_batch_size", type=int, default=512)
    p.add_argument("--classifier_eval_batch_size", type=int, default=2048)
    p.add_argument("--classifier_epochs", type=int, default=20)
    p.add_argument("--classifier_lr", type=float, default=1e-3)
    p.add_argument("--classifier_weight_decay", type=float, default=1e-4)
    p.add_argument("--classifier_dropout", type=float, default=0.0)
    p.add_argument("--classifier_hidden_dim", type=int, default=0)
    p.add_argument("--classifier_early_stop_patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch_dtype", default="bf16", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--output_dir", default="output/foundation-sequence-classifier-benchmark")
    return p.parse_args()


def sanitize_label(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip("/"))
    return clean or "model"


def resolve_model_labels(model_paths: List[str], model_labels: Optional[List[str]]) -> List[str]:
    if model_labels is not None:
        if len(model_labels) != len(model_paths):
            raise ValueError("--model_labels must have the same length as --model_paths")
        return model_labels
    labels = []
    for path in model_paths:
        p = Path(path)
        labels.append(sanitize_label(p.name if p.name else path))
    return labels


def resolve_torch_dtype(name: str):
    if name == "auto":
        return None
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("Unsupported torch_dtype=%r" % name)


def build_event_text_map(args) -> Dict[int, str]:
    encoder = get_encoder(
        args.encoder,
        model_name=args.model_paths[0],
        template_path=args.template_path,
        append_token_text=None,
        append_token_name=None,
        pool_max_tokens=args.pool_max_tokens,
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


def truncate_event_ids(event_ids: np.ndarray, max_events: Optional[int], truncate_side: str) -> np.ndarray:
    if max_events is not None and len(event_ids) > max_events:
        if truncate_side == "last":
            return event_ids[-max_events:]
        return event_ids[:max_events]
    return event_ids


def build_sequence_rows(
    rows: List[Tuple[np.ndarray, int]],
    event_text_map: Dict[int, str],
    max_events: Optional[int],
    truncate_side: str,
) -> List[Tuple[List[str], int]]:
    seq_rows = []
    dropped = 0
    for event_ids, label in rows:
        kept_ids = truncate_event_ids(event_ids, max_events, truncate_side)
        parts = [event_text_map[int(eid)] for eid in kept_ids if int(eid) in event_text_map]
        if not parts:
            dropped += 1
            seq_rows.append(([], label))
            continue
        seq_rows.append((parts, label))
    if dropped:
        logger.info("Encountered %d rows with no encodable events after truncation.", dropped)
    return seq_rows


def mean_pool_hidden(hidden_states: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
    pool_mask_f = pool_mask.float().unsqueeze(-1)
    summed = (hidden_states * pool_mask_f).sum(dim=1)
    counts = pool_mask_f.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def get_sequence_embeddings(outputs, enc, tokenizer, encoder):
    if encoder.pooling_mode in {"suffix_only", "all_suffix_mean"}:
        return encoder.get_embeddings(outputs, enc, tokenizer)

    pool_mask = enc["attention_mask"].bool()
    if encoder.append_token_id is not None:
        # For concatenated event sequences we want every per-event separator token
        # excluded from mean pooling, not just the final appended one.
        pool_mask = pool_mask & (enc["input_ids"] != encoder.append_token_id)

    if encoder.pool_max_tokens is not None:
        if encoder.pool_max_tokens <= 0:
            raise ValueError("pool_max_tokens must be positive, got %s" % encoder.pool_max_tokens)
        token_ord = torch.cumsum(pool_mask.long(), dim=1)
        pool_mask = pool_mask & (token_ord <= encoder.pool_max_tokens)

    return mean_pool_hidden(outputs.last_hidden_state, pool_mask).float()


@torch.inference_mode()
def encode_sequence_rows(
    seq_rows: List[Tuple[List[str], int]],
    model,
    tokenizer,
    encoder,
    batch_size: int,
    event_separator: str,
    device: torch.device,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    event_lists = [events for events, _ in seq_rows]
    labels = np.asarray([float(label) for _, label in seq_rows], dtype=np.float32)
    hidden_size = int(model.config.hidden_size)
    features = np.zeros((len(event_lists), hidden_size), dtype=np.float32)
    encoder.resolve_append_token(tokenizer)
    if encoder.append_token_text:
        ok = encoder._set_append_token_id_from_existing_vocab(tokenizer)
        if not ok:
            raise ValueError("Sequence benchmark requires append token to already exist in tokenizer vocab.")

    sep_ids = []
    if event_separator:
        sep_ids = tokenizer(
            event_separator,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

    for start in tqdm(range(0, len(event_lists), batch_size), desc=desc, dynamic_ncols=True):
        batch_events = event_lists[start:start + batch_size]
        batch_ids = []
        for events in batch_events:
            flat_ids = []
            for event_idx, event_text in enumerate(events):
                event_ids = tokenizer(
                    event_text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                flat_ids.extend(event_ids)
                if encoder.append_token_id is not None:
                    flat_ids.append(int(encoder.append_token_id))
                if sep_ids and event_idx != len(events) - 1:
                    flat_ids.extend(sep_ids)
            batch_ids.append(flat_ids)

        max_len = max((len(ids) for ids in batch_ids), default=1)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer %s has no pad_token_id" % tokenizer.__class__.__name__)
        input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=device)
        for row_idx, ids in enumerate(batch_ids):
            if not ids:
                continue
            input_ids[row_idx, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row_idx, :len(ids)] = 1

        enc = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        outputs = model(**enc, return_dict=True)
        batch_embs = get_sequence_embeddings(outputs, enc, tokenizer, encoder)
        batch_embs = encoder.postprocess_embeddings(batch_embs).cpu().numpy()
        if encoder.pooling_mode == "all_suffix_mean":
            suffix_counts = (input_ids == int(encoder.append_token_id)).sum(dim=1).tolist()
            if start == 0:
                logger.info(
                    "First sequence batch suffix counts: min=%d p50=%d max=%d",
                    int(min(suffix_counts)),
                    int(sorted(suffix_counts)[len(suffix_counts) // 2]),
                    int(max(suffix_counts)),
                )
            if min(suffix_counts) <= 0:
                raise ValueError("all_suffix_mean encountered a sequence with no suffix tokens in the encoded batch")
        features[start:start + len(batch_events)] = batch_embs

    return features, labels


def build_classifier_args(args):
    class Obj(object):
        pass
    out = Obj()
    out.hidden_dim = args.classifier_hidden_dim
    out.dropout = args.classifier_dropout
    out.lr = args.classifier_lr
    out.weight_decay = args.classifier_weight_decay
    out.batch_size = args.classifier_batch_size
    out.eval_batch_size = args.classifier_eval_batch_size
    out.num_workers = args.num_workers
    out.early_stop_patience = args.classifier_early_stop_patience
    out.epochs = args.classifier_epochs
    return out


def load_model_and_tokenizer(args, model_path: str):
    tokenizer_source = args.tokenizer_name or model_path
    tokenizer_kwargs = {"local_files_only": True} if args.local_files_only else {}
    model_kwargs = {}
    if args.local_files_only:
        model_kwargs["local_files_only"] = True
    dtype = resolve_torch_dtype(args.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    return tokenizer, model


def main():
    args = parse_args()
    bench.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = bench.resolve_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    model_labels = resolve_model_labels(args.model_paths, args.model_labels)
    event_text_map = build_event_text_map(args)

    logger.info("Output dir: %s", out_dir)
    logger.info("Device: %s", device)
    logger.info("Tasks: %s", ", ".join(args.tasks))
    logger.info("Train split: %s | Test split: %s", args.train_split, args.test_split)
    logger.info("Event truncation: keep %s %s events", args.max_events, args.truncate_side)

    rows_cache = {}
    seq_rows_cache = {}
    for task in args.tasks:
        train_rows = bench.read_eval_rows(args.eval_data_dir, task, args.train_split)
        test_rows = bench.read_eval_rows(args.eval_data_dir, task, args.test_split)
        rows_cache[(task, "train")] = train_rows
        rows_cache[(task, "test")] = test_rows
        seq_rows_cache[(task, "train")] = build_sequence_rows(
            train_rows, event_text_map, args.max_events, args.truncate_side, args.event_separator
        )
        seq_rows_cache[(task, "test")] = build_sequence_rows(
            test_rows, event_text_map, args.max_events, args.truncate_side, args.event_separator
        )

    all_results = {}
    summary_rows = []
    classifier_args = build_classifier_args(args)

    for model_label, model_path in zip(model_labels, args.model_paths):
        logger.info("=" * 100)
        logger.info("Benchmarking sequence model: %s -> %s", model_label, model_path)
        logger.info("=" * 100)

        tokenizer, model = load_model_and_tokenizer(args, model_path)
        encoder = get_encoder(
            args.encoder,
            model_name=model_path,
            template_path=args.template_path,
            append_token_text=args.append_token_text,
            append_token_name=args.append_token_name,
            pool_max_tokens=args.pool_max_tokens,
            pooling_mode=args.pooling_mode,
        )
        tokenizer, model = encoder.configure_tokenizer_and_model(tokenizer, model)
        model = model.to(device)
        model.eval()

        task_results = {}
        for task in args.tasks:
            logger.info("  Task: %s", task)
            train_x, train_y = encode_sequence_rows(
                seq_rows_cache[(task, "train")],
                model=model,
                tokenizer=tokenizer,
                encoder=encoder,
                batch_size=args.batch_size,
                event_separator=args.event_separator,
                device=device,
                desc="%s train encode" % task,
            )
            test_x, test_y = encode_sequence_rows(
                seq_rows_cache[(task, "test")],
                model=model,
                tokenizer=tokenizer,
                encoder=encoder,
                batch_size=args.batch_size,
                event_separator=args.event_separator,
                device=device,
                desc="%s test encode" % task,
            )
            train_x, [test_x], _, _ = bench.standardize(train_x, [test_x])

            logger.info(
                "    sequence features: train=%s test=%s",
                tuple(train_x.shape),
                tuple(test_x.shape),
            )
            train_pos = int(train_y.sum())
            train_neg = int(len(train_y) - train_pos)
            test_pos = int(test_y.sum())
            test_neg = int(len(test_y) - test_pos)
            pos_weight = float(train_neg / max(train_pos, 1))
            logger.info(
                "    labels: train_pos=%d/%d train_neg=%d/%d test_pos=%d/%d test_neg=%d/%d pos_weight=%.4f",
                train_pos,
                len(train_y),
                train_neg,
                len(train_y),
                test_pos,
                len(test_y),
                test_neg,
                len(test_y),
                pos_weight,
            )

            task_result = bench.train_single_task(
                train_x=train_x,
                train_y=train_y,
                val_x=train_x,
                val_y=train_y,
                test_x=test_x,
                test_y=test_y,
                args=classifier_args,
                device=device,
            )
            task_results[task] = task_result
            logger.info(
                "    best train: auc=%.4f auprc=%.4f bal_acc=%.4f f1=%.4f epoch=%d",
                task_result["train"]["auc"],
                task_result["train"]["auprc"],
                task_result["train"]["balanced_accuracy"],
                task_result["train"]["f1"],
                int(task_result["best_epoch"]["epoch"]),
            )
            logger.info(
                "    test: auc=%.4f auprc=%.4f bal_acc=%.4f f1=%.4f",
                task_result["test"]["auc"],
                task_result["test"]["auprc"],
                task_result["test"]["balanced_accuracy"],
                task_result["test"]["f1"],
            )

        aggregate = {
            "train": bench.aggregate_task_metrics(task_results, "train"),
            "test": bench.aggregate_task_metrics(task_results, "test"),
        }
        all_results[model_label] = {
            "model_path": model_path,
            "aggregate": aggregate,
            "tasks": task_results,
        }
        summary_rows.append({
            "model_label": model_label,
            "model_path": model_path,
            **bench.prefix_metrics(aggregate["train"], "train"),
            **bench.prefix_metrics(aggregate["test"], "test"),
        })

        logger.info(
            "Summary for %s: macro_test_auc=%.4f macro_test_auprc=%.4f macro_test_bal_acc=%.4f",
            model_label,
            aggregate["test"]["macro_auc"],
            aggregate["test"]["macro_auprc"],
            aggregate["test"]["macro_balanced_accuracy"],
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("test_macro_auc", ascending=False).reset_index(drop=True)

    summary_path = out_dir / "summary.csv"
    results_path = out_dir / "results.json"
    config_path = out_dir / "config.json"
    summary_df.to_csv(summary_path, index=False)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Saved summary -> %s", summary_path)
    logger.info("Saved detailed results -> %s", results_path)
    if not summary_df.empty:
        logger.info("Final ranking by macro test AUC:")
        for row in summary_df.itertuples(index=False):
            logger.info(
                "  %s: macro_test_auc=%.4f macro_test_auprc=%.4f macro_test_bal_acc=%.4f",
                row.model_label,
                row.test_macro_auc,
                row.test_macro_auprc,
                row.test_macro_balanced_accuracy,
            )


if __name__ == "__main__":
    main()
