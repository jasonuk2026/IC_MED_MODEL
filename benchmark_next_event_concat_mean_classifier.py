#!/usr/bin/env python3

import argparse
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import benchmark_foundation_simple_classifier as bench
from train_next_event_concat_mean import NextEventConcatMeanModel, collate_concat_event_sequences

GEN_META_DIR = Path(__file__).resolve().parent / "01_gen_meta"
import sys

if str(GEN_META_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_META_DIR))

from build_next_event_train_parquet import format_event_row, load_event_template


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark concat-mean next-event models on downstream patient classification tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint_paths", nargs="+", required=True,
                   help="Paths to train_next_event_concat_mean checkpoint directories containing model.pt.")
    p.add_argument("--checkpoint_labels", nargs="+", default=None,
                   help="Optional labels for --checkpoint_paths.")
    p.add_argument("--tokenizer_name", default=None,
                   help="Optional tokenizer override; defaults to model_name stored in the checkpoint args.")
    p.add_argument("--template_path", default="01_gen_meta/templates/biolinkbert_event.j2")
    p.add_argument("--unique_events_path", default="data/01_outputs/unique_events.parquet")
    p.add_argument("--eval_data_dir", default="data/eval_data_latest")
    p.add_argument("--tasks", nargs="+", default=bench.TASKS, choices=bench.TASKS)
    p.add_argument("--train_split", default="val", choices=["val", "test"])
    p.add_argument("--test_split", default="test", choices=["val", "test"])
    p.add_argument("--max_events", type=int, default=1000)
    p.add_argument("--truncate_side", default="last", choices=["first", "last"])
    p.add_argument("--max_event_tokens", type=int, default=None,
                   help="Optional override; defaults to the training checkpoint setting.")
    p.add_argument("--event_truncate_side", default=None, choices=["first", "last"],
                   help="Optional override; defaults to the training checkpoint setting.")
    p.add_argument("--append_eos_per_event", action="store_true",
                   help="Append tokenizer pad/eos token to every event before encoding.")
    p.add_argument("--sequence_pooling", default="mean", choices=["mean", "last"])
    p.add_argument("--encode_batch_size", type=int, default=8,
                   help="Number of patient sequences to encode at once.")
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
    p.add_argument("--output_dir", default="output/next-event-concat-mean-classifier-benchmark")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)
    return p.parse_args()


def sanitize_label(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip("/"))
    return clean or "checkpoint"


def resolve_labels(paths: List[str], labels: Optional[List[str]]) -> List[str]:
    if labels is not None:
        if len(labels) != len(paths):
            raise ValueError("--checkpoint_labels must have the same length as --checkpoint_paths")
        return labels
    return [sanitize_label(Path(path).name or path) for path in paths]


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


def build_event_token_map(
    *,
    unique_events_path: str,
    tokenizer,
    template_path: str,
    append_eos_per_event: bool,
) -> Dict[int, List[int]]:
    event_df = pd.read_parquet(unique_events_path)
    required_cols = {"event_id", "omop_table", "event_type", "code", "description", "value", "unit"}
    missing = sorted(required_cols - set(event_df.columns))
    if missing:
        raise ValueError("Unique events parquet missing columns: %s" % ", ".join(missing))

    template = load_event_template(template_path)
    pad_token_id = tokenizer.pad_token_id
    if append_eos_per_event and pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id but --append_eos_per_event was requested")

    token_map = {}
    dropped = 0
    for row in event_df.itertuples(index=False):
        text = format_event_row(pd.Series({
            "omop_table": row.omop_table,
            "event_type": row.event_type,
            "code": row.code,
            "description": row.description,
            "value": row.value,
            "unit": row.unit,
        }), False, template)
        if not text:
            dropped += 1
            continue
        ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        ids = [int(x) for x in ids]
        if append_eos_per_event:
            ids.append(int(pad_token_id))
        token_map[int(row.event_id)] = ids
    logger.info(
        "Prepared tokenized event map for %d unique events (%d dropped by template).",
        len(token_map),
        dropped,
    )
    return token_map


def truncate_event_ids(event_ids: np.ndarray, max_events: Optional[int], truncate_side: str) -> np.ndarray:
    if max_events is not None and len(event_ids) > max_events:
        if truncate_side == "last":
            return event_ids[-max_events:]
        return event_ids[:max_events]
    return event_ids


def build_sequence_rows(
    rows: List[Tuple[np.ndarray, int]],
    event_token_map: Dict[int, List[int]],
    max_events: Optional[int],
    truncate_side: str,
) -> List[Tuple[List[List[int]], int]]:
    seq_rows = []
    dropped = 0
    for event_ids, label in rows:
        kept_ids = truncate_event_ids(event_ids, max_events, truncate_side)
        parts = [event_token_map[int(eid)] for eid in kept_ids if int(eid) in event_token_map]
        if not parts:
            dropped += 1
            seq_rows.append(([], label))
            continue
        seq_rows.append((parts, label))
    if dropped:
        logger.info("Encountered %d rows with no encodable events after truncation.", dropped)
    return seq_rows


def mean_pool_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    summed = (hidden_states * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class SequenceRowsDataset(Dataset):
    def __init__(self, seq_rows: List[Tuple[List[List[int]], int]]):
        self.seq_rows = seq_rows

    def __len__(self) -> int:
        return len(self.seq_rows)

    def __getitem__(self, idx: int) -> Tuple[List[List[int]], float]:
        events, label = self.seq_rows[idx]
        return events, float(label)


@torch.inference_mode()
def encode_sequence_rows(
    seq_rows: List[Tuple[List[List[int]], int]],
    model: NextEventConcatMeanModel,
    pad_token_id: int,
    max_events: int,
    max_event_tokens: int,
    event_truncate_side: str,
    sequence_pooling: str,
    batch_size: int,
    device: torch.device,
    desc: str,
    num_workers: int,
    wandb_run=None,
    wandb_prefix: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    hidden_size = int(model.hidden_size)
    features = np.zeros((len(seq_rows), hidden_size), dtype=np.float32)
    labels = np.zeros(len(seq_rows), dtype=np.float32)
    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    ds = SequenceRowsDataset(seq_rows)
    dl_kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": lambda batch: {
            **collate_concat_event_sequences(
                [events for events, _ in batch],
                pad_token_id=pad_token_id,
                max_events=max_events,
                max_event_tokens=max_event_tokens,
                sequence_truncate_side="last",
                event_truncate_side=event_truncate_side,
            ),
            "labels": torch.tensor([label for _, label in batch], dtype=torch.float32),
        },
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 8
    dl = DataLoader(**dl_kwargs)

    offset = 0
    for batch_idx, collated in enumerate(tqdm(dl, desc=desc, dynamic_ncols=True), start=1):
        batch_n = int(collated["labels"].shape[0])
        labels[offset:offset + batch_n] = collated["labels"].numpy()
        collated = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in collated.items()}
        synchronize()
        t0 = datetime.now()
        seq_embs, _ = model.encode_sequence_embeddings(
            input_ids=collated["input_ids"],
            attention_mask=collated["attention_mask"],
            token_event_index=collated["token_event_index"],
            sequence_event_mask=collated["sequence_event_mask"],
        )
        synchronize()
        elapsed = (datetime.now() - t0).total_seconds()
        seq_mask = collated["sequence_event_mask"].bool()
        if sequence_pooling == "last":
            lengths = seq_mask.long().sum(dim=1).clamp(min=1) - 1
            row_idx = torch.arange(seq_embs.shape[0], device=seq_embs.device)
            pooled = seq_embs[row_idx, lengths]
        else:
            pooled = mean_pool_hidden(seq_embs, seq_mask)
        features[offset:offset + batch_n] = pooled.float().cpu().numpy()
        if wandb_run is not None and wandb_prefix is not None:
            wandb_run.log(
                {
                    f"{wandb_prefix}/encode_step_seconds": elapsed,
                    f"{wandb_prefix}/encode_step_batch_size": batch_n,
                    f"{wandb_prefix}/encode_step_index": batch_idx,
                }
            )
        offset += batch_n
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


def load_checkpoint_model(args, checkpoint_dir: str, device: torch.device):
    checkpoint_dir = Path(checkpoint_dir)
    payload = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    train_args = payload["args"]

    model_name = train_args["model_name"]
    tokenizer_source = args.tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)

    train_torch_dtype = torch.float32
    if train_args.get("bf16"):
        train_torch_dtype = torch.bfloat16
    elif train_args.get("fp16"):
        train_torch_dtype = torch.float16
    attn_impl = "flash_attention_2" if train_args.get("flash_attn") else "eager"
    model = NextEventConcatMeanModel(
        model_name=model_name,
        max_events=int(train_args["max_events"]),
        predictor_hidden_size=int(train_args["predictor_hidden_size"]),
        predictor_num_heads=int(train_args["predictor_num_heads"]),
        predictor_ffn_dim=int(train_args["predictor_ffn_dim"]),
        predictor_num_layers=int(train_args["predictor_num_layers"]),
        predictor_dropout=float(train_args["predictor_dropout"]),
        freeze_event_encoder=bool(train_args.get("freeze_event_encoder", False)),
        torch_dtype=train_torch_dtype,
        attn_implementation=attn_impl,
        local_files_only=args.local_files_only,
    )
    model.load_state_dict(payload["state_dict"])
    dtype = resolve_torch_dtype(args.torch_dtype)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    default_max_event_tokens = int(train_args["max_event_tokens"])
    default_event_truncate_side = str(train_args.get("event_truncate_side", "last"))
    return tokenizer, model, train_args, default_max_event_tokens, default_event_truncate_side


def main():
    args = parse_args()
    bench.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = bench.resolve_device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb_project is not None:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"concat-mean-benchmark-{timestamp}",
            tags=args.wandb_tags,
            config=vars(args),
        )

    checkpoint_labels = resolve_labels(args.checkpoint_paths, args.checkpoint_labels)
    rows_cache = {}
    for task in args.tasks:
        rows_cache[(task, "train")] = bench.read_eval_rows(args.eval_data_dir, task, args.train_split)
        rows_cache[(task, "test")] = bench.read_eval_rows(args.eval_data_dir, task, args.test_split)

    logger.info("Output dir: %s", out_dir)
    logger.info("Device: %s", device)
    logger.info("Tasks: %s", ", ".join(args.tasks))
    logger.info("Train split: %s | Test split: %s", args.train_split, args.test_split)
    logger.info("Event truncation: keep %s %s events", args.max_events, args.truncate_side)
    logger.info("Sequence pooling: %s", args.sequence_pooling)
    logger.info("Append eos per event: %s", args.append_eos_per_event)

    classifier_args = build_classifier_args(args)
    all_results = {}
    summary_rows = []

    for checkpoint_label, checkpoint_path in zip(checkpoint_labels, args.checkpoint_paths):
        logger.info("=" * 100)
        logger.info("Benchmarking concat-mean checkpoint: %s -> %s", checkpoint_label, checkpoint_path)
        logger.info("=" * 100)

        tokenizer, model, train_args, default_max_event_tokens, default_event_truncate_side = load_checkpoint_model(
            args, checkpoint_path, device
        )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id")

        max_event_tokens = args.max_event_tokens or default_max_event_tokens
        event_truncate_side = args.event_truncate_side or default_event_truncate_side
        event_token_map = build_event_token_map(
            unique_events_path=args.unique_events_path,
            tokenizer=tokenizer,
            template_path=args.template_path,
            append_eos_per_event=args.append_eos_per_event,
        )

        seq_rows_cache = {}
        for task in args.tasks:
            seq_rows_cache[(task, "train")] = build_sequence_rows(
                rows_cache[(task, "train")], event_token_map, args.max_events, args.truncate_side
            )
            seq_rows_cache[(task, "test")] = build_sequence_rows(
                rows_cache[(task, "test")], event_token_map, args.max_events, args.truncate_side
            )

        task_results = {}
        for task in args.tasks:
            logger.info("  Task: %s", task)
            train_x, train_y = encode_sequence_rows(
                seq_rows_cache[(task, "train")],
                model=model,
                pad_token_id=pad_token_id,
                max_events=min(args.max_events, int(train_args["max_events"])),
                max_event_tokens=max_event_tokens,
                event_truncate_side=event_truncate_side,
                sequence_pooling=args.sequence_pooling,
                batch_size=args.encode_batch_size,
                device=device,
                desc=f"{task} train encode",
                num_workers=args.num_workers,
                wandb_run=wandb_run,
                wandb_prefix=f"{checkpoint_label}/{task}/train",
            )
            test_x, test_y = encode_sequence_rows(
                seq_rows_cache[(task, "test")],
                model=model,
                pad_token_id=pad_token_id,
                max_events=min(args.max_events, int(train_args["max_events"])),
                max_event_tokens=max_event_tokens,
                event_truncate_side=event_truncate_side,
                sequence_pooling=args.sequence_pooling,
                batch_size=args.encode_batch_size,
                device=device,
                desc=f"{task} test encode",
                num_workers=args.num_workers,
                wandb_run=wandb_run,
                wandb_prefix=f"{checkpoint_label}/{task}/test",
            )
            train_x, [test_x], _, _ = bench.standardize(train_x, [test_x])

            logger.info("    sequence features: train=%s test=%s", tuple(train_x.shape), tuple(test_x.shape))
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
        all_results[checkpoint_label] = {
            "checkpoint_path": checkpoint_path,
            "train_args": train_args,
            "aggregate": aggregate,
            "tasks": task_results,
        }
        summary_rows.append({
            "checkpoint_label": checkpoint_label,
            "checkpoint_path": checkpoint_path,
            **bench.prefix_metrics(aggregate["train"], "train"),
            **bench.prefix_metrics(aggregate["test"], "test"),
        })

        logger.info(
            "Summary for %s: macro_test_auc=%.4f macro_test_auprc=%.4f macro_test_bal_acc=%.4f",
            checkpoint_label,
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
                row.checkpoint_label,
                row.test_macro_auc,
                row.test_macro_auprc,
                row.test_macro_balanced_accuracy,
            )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
