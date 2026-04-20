#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_concat_classifier import DiseaseEventConcatClassifier
from model_cross_attention_classifier import DiseaseEventCrossAttentionClassifier
from model_soft_token_classifier import DiseaseEventSoftTokenClassifier
from train_disease_soft_token_classifier import (
    TASK_2_DISEASE_NAME,
    TASK_2_IDX,
    EmbeddingStore,
    EvalBatchDataset,
    _binary_roc_auc,
    build_task_text_embs,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot per-task positive/negative score distributions for disease-conditioned classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_type", choices=["concat", "cross_attn", "soft_token"], required=True)
    p.add_argument("--eval_data_paths", nargs="+", required=True)
    p.add_argument("--bert_embeddings", required=True)
    p.add_argument("--disease_model_name", default="michiyasunaga/BioLinkBERT-base")
    p.add_argument("--disease_model_path", default=None)
    p.add_argument("--disease_tokenizer_name", default=None)
    p.add_argument("--tasks", nargs="+", default=list(sorted(TASK_2_DISEASE_NAME.keys())))
    p.add_argument("--output_dir", default="figures/classifier_score_distributions")
    p.add_argument("--output_name", default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--pad_to_num_events", type=int, default=1000)
    p.add_argument("--bins", type=int, default=40)
    return p.parse_args()


def _load_model(args, task_text_embs: torch.Tensor, device: torch.device, dtype: torch.dtype):
    ckpt = Path(args.checkpoint)
    if args.model_type == "concat":
        return DiseaseEventConcatClassifier.load_checkpoint(
            ckpt, task_text_embs=task_text_embs, device=device, dtype=dtype
        )
    if args.model_type == "cross_attn":
        return DiseaseEventCrossAttentionClassifier.load_checkpoint(
            ckpt, task_text_embs=task_text_embs, device=device, dtype=dtype
        )
    if args.model_type == "soft_token":
        return DiseaseEventSoftTokenClassifier.load_checkpoint(
            ckpt, task_text_embs=task_text_embs, device=device, dtype=dtype
        )
    raise ValueError(f"Unsupported model_type={args.model_type}")


@torch.inference_mode()
def _collect_scores(args, model, store: EmbeddingStore, device: torch.device) -> pd.DataFrame:
    rows: list[tuple[np.ndarray, int, int]] = []
    for path in args.eval_data_paths:
        df = pd.read_parquet(path, columns=["task_idx", "label", "event_ids"])
        for row in df.itertuples(index=False):
            rows.append((np.array(row.event_ids, dtype=np.int32), int(row.task_idx), int(row.label)))

    entries = [(eids, task_idx) for eids, task_idx, _ in rows]
    labels = torch.tensor([label for _, _, label in rows], dtype=torch.long)
    task_idxs = torch.tensor([task_idx for _, task_idx, _ in rows], dtype=torch.long)
    eval_ds = EvalBatchDataset(entries, store, args.eval_batch_size, args.pad_to_num_events)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)

    logits_list = []
    for batch in tqdm(eval_dl, desc="Scoring", dynamic_ncols=True):
        logits = model(
            batch["event_embs"].to(device),
            batch["event_mask"].to(device),
            batch["task_idxs"].to(device),
        )
        logits_list.append(logits.float().cpu())

    logits = torch.cat(logits_list, dim=0)
    probs = torch.sigmoid(logits)
    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    task_names = [idx_to_name[int(t)] for t in task_idxs.tolist()]
    return pd.DataFrame(
        {
            "task": task_names,
            "task_idx": task_idxs.tolist(),
            "label": labels.tolist(),
            "logit": logits.tolist(),
            "prob": probs.tolist(),
        }
    )


def _plot_distributions(df: pd.DataFrame, output_path: Path, bins: int):
    tasks = list(sorted(df["task"].unique()))
    n_tasks = len(tasks)
    ncols = min(3, max(1, n_tasks))
    nrows = ceil(n_tasks / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.8 * nrows), squeeze=False)
    colors = {0: "#4E5B6E", 1: "#C44E3B"}

    global_min = float(df["prob"].min())
    global_max = float(df["prob"].max())
    if global_min == global_max:
        global_min, global_max = 0.0, 1.0

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, task in zip(axes.flat, tasks):
        ax.set_visible(True)
        tdf = df[df["task"] == task]
        neg = tdf.loc[tdf["label"] == 0, "prob"].to_numpy(dtype=float)
        pos = tdf.loc[tdf["label"] == 1, "prob"].to_numpy(dtype=float)
        task_auc = _binary_roc_auc(
            torch.tensor(tdf["prob"].to_numpy(dtype=float)),
            torch.tensor(tdf["label"].to_numpy(dtype=int)),
        )

        if neg.size > 0:
            ax.hist(
                neg,
                bins=bins,
                range=(global_min, global_max),
                density=True,
                alpha=0.55,
                color=colors[0],
                label=f"Negative (n={neg.size})",
            )
            ax.axvline(float(np.mean(neg)), color=colors[0], linestyle="--", linewidth=1.2)
        if pos.size > 0:
            ax.hist(
                pos,
                bins=bins,
                range=(global_min, global_max),
                density=True,
                alpha=0.55,
                color=colors[1],
                label=f"Positive (n={pos.size})",
            )
            ax.axvline(float(np.mean(pos)), color=colors[1], linestyle="--", linewidth=1.2)

        ax.set_title(f"{task}\nAUC={task_auc:.4f}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.15, linewidth=0.6)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Disease-Conditioned Score Distributions", fontsize=16, y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    logger.info("Loading embeddings from %s", args.bert_embeddings)
    store = EmbeddingStore(args.bert_embeddings)
    task_text_embs = build_task_text_embs(args, device, rank=0, is_ddp=False, expected_dim=store.dim)
    model = _load_model(args, task_text_embs, device, dtype)
    model.eval()

    logger.info("Scoring evaluation data with %s", args.model_type)
    df = _collect_scores(args, model, store, device)

    output_dir = Path(args.output_dir)
    output_name = args.output_name or f"{args.model_type}_score_distributions.png"
    output_path = output_dir / output_name
    _plot_distributions(df, output_path, args.bins)

    summary = (
        df.groupby(["task", "label"])["prob"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values(["task", "label"])
    )
    summary_path = output_dir / f"{Path(output_name).stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved plot -> %s", output_path)
    logger.info("Saved summary -> %s", summary_path)


if __name__ == "__main__":
    main()
