#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_concat_classifier import DiseaseEventConcatClassifier
from train_embedding_disease_cond_v2 import (
    BERT_DIM,
    TASK_2_IDX,
    EmbeddingStore,
    EvalBatchDataset,
    _binary_roc_auc,
    _precision_recall_at_k,
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
        description="Evaluate disease->patient retrieval using concat-classifier hidden/logit path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_data_paths", nargs="+", required=True)
    p.add_argument("--bert_embeddings", required=True)
    p.add_argument("--disease_model_name", default="michiyasunaga/BioLinkBERT-base")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--pad_to_num_events", type=int, default=1000)
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    dummy = argparse.Namespace(
        disease_model_name=args.disease_model_name,
        bf16=args.bf16,
        fp16=args.fp16,
    )
    task_text_embs = build_task_text_embs(dummy, device, rank=0, is_ddp=False)
    model = DiseaseEventConcatClassifier.load_checkpoint(
        Path(args.checkpoint),
        task_text_embs=task_text_embs,
        device=device,
        dtype=dtype,
    )
    model.eval()

    store = EmbeddingStore(args.bert_embeddings)

    rows: list[tuple[np.ndarray, int, int]] = []
    import pandas as pd
    for path in args.eval_data_paths:
        df = pd.read_parquet(path, columns=["task_idx", "label", "event_ids"])
        for row in df.itertuples(index=False):
            rows.append((np.array(row.event_ids, dtype=np.int32), int(row.task_idx), int(row.label)))

    entries = [(eids, task_idx) for eids, task_idx, _ in rows]
    labels = torch.tensor([label for _, _, label in rows], dtype=torch.long)
    task_idxs = torch.tensor([task_idx for _, task_idx, _ in rows], dtype=torch.long)
    eval_ds = EvalBatchDataset(entries, store, args.eval_batch_size, args.pad_to_num_events)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)

    all_logits = []
    all_hidden = []
    for batch in eval_dl:
        logits, _patient, _disease, hidden = model(
            batch["event_embs"].to(device),
            batch["event_mask"].to(device),
            batch["task_idxs"].to(device),
            return_features=True,
        )
        all_logits.append(logits.cpu())
        all_hidden.append(hidden.cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = torch.sigmoid(logits)
    hidden = torch.cat(all_hidden, dim=0)

    overall_auc = _binary_roc_auc(probs, labels)
    p10, r10 = _precision_recall_at_k(probs, labels, 10)
    p50, r50 = _precision_recall_at_k(probs, labels, 50)

    logger.info("[classifier_retrieval auc] %.4f", overall_auc)
    logger.info("[classifier_retrieval counts] n=%d pos=%d neg=%d", int(labels.numel()), int((labels == 1).sum().item()), int((labels == 0).sum().item()))
    logger.info("[classifier_retrieval p@10/r@10] %.4f/%.4f", p10, r10)
    logger.info("[classifier_retrieval p@50/r@50] %.4f/%.4f", p50, r50)
    logger.info("[classifier_hidden] mean_l2_norm=%.4f", hidden.norm(dim=1).mean().item())

    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    for t_idx in sorted(set(task_idxs.tolist())):
        mask = task_idxs == t_idx
        task_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
        task_probs = probs[mask]
        task_labels = labels[mask]
        tp10, tr10 = _precision_recall_at_k(task_probs, task_labels, 10)
        tp50, tr50 = _precision_recall_at_k(task_probs, task_labels, 50)
        logger.info(
            "%s: auc=%.4f  p@10=%.4f  r@10=%.4f  p@50=%.4f  r@50=%.4f",
            task_name,
            _binary_roc_auc(task_probs, task_labels),
            tp10,
            tr10,
            tp50,
            tr50,
        )


if __name__ == "__main__":
    main()
