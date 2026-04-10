#!/usr/bin/env python3
"""
train_disease_jepa.py

JEPA-style disease-to-patient retrieval training.

- Patient events are encoded by an online shared backbone, pooled, and passed
  through a predictor head.
- Disease text embeddings are encoded by an EMA teacher copy of the same
  backbone.
- The retrieval objective aligns patient predictions with teacher disease
  targets while retaining the same auxiliary SupCon + variance regularisation
  used by the retrieval baseline.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_jepa import DiseasePatientJEPAModel
from train_embedding_disease_cond_v2 import (
    BERT_DIM,
    TASK_2_DISEASE_NAME,
    TASK_2_IDX,
    EmbeddingStore,
    EpochBatchDataset,
    EvalBatchDataset,
    _binary_roc_auc,
    _epoch_worker_init,
    _precision_recall_at_k,
    _supervised_contrastive_loss,
    build_task_text_embs,
    precompute_eval_query_pairs,
    precompute_eval_triplets,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="JEPA-style disease-to-patient retrieval training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", default=None)

    p.add_argument("--train_data_dir", default=None)
    p.add_argument("--train_data_epochs", nargs="+", type=int, default=None)
    p.add_argument("--tasks", nargs="+", default=list(sorted(TASK_2_DISEASE_NAME.keys())))
    p.add_argument("--eval_data_paths", nargs="+", default=None)
    p.add_argument("--bert_embeddings", required=True)
    p.add_argument("--disease_model_name", default="michiyasunaga/BioLinkBERT-base")

    p.add_argument("--shallow_encoder_type", choices=["simple", "mlp", "transformer"], default="simple")
    p.add_argument("--shallow_num_layers", type=int, default=0)
    p.add_argument("--shallow_num_heads", type=int, default=4)
    p.add_argument("--shallow_intermediate_size", type=int, default=None)
    p.add_argument("--predictor_layers", type=int, default=1)
    p.add_argument("--predictor_intermediate_size", type=int, default=None)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--objective", choices=["retrieval_jepa", "mse_margin", "signed_mse", "negative_only_mse"], default="retrieval_jepa")

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--compile", action="store_true")

    p.add_argument("--output_dir", default="output/disease-jepa")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_steps", type=int, default=10)

    p.add_argument("--retrieval_temperature", type=float, default=0.1)
    p.add_argument("--retrieval_supcon_weight", type=float, default=0.2)
    p.add_argument("--stage1_temperature", type=float, default=0.1)
    p.add_argument("--var_reg_weight", type=float, default=0.1)
    p.add_argument("--jepa_mse_weight", type=float, default=0.2)
    p.add_argument("--neg_margin", type=float, default=1.0)
    p.add_argument("--neg_margin_weight", type=float, default=1.0)

    p.add_argument("--n_eval_triplets_per_task", type=int, default=1024)
    p.add_argument("--pad_to_num_events", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)
    return p.parse_args()


def _triplet_correct(x: torch.Tensor, metric: str) -> torch.Tensor:
    anchors = x[0::3]
    positives = x[1::3]
    negatives = x[2::3]
    if metric == "l2":
        return (anchors - positives).norm(dim=1) < (anchors - negatives).norm(dim=1)
    if metric == "cosine":
        sim_ap = torch.nn.functional.cosine_similarity(anchors, positives, dim=1)
        sim_an = torch.nn.functional.cosine_similarity(anchors, negatives, dim=1)
        return sim_ap > sim_an
    raise ValueError(f"Unknown metric: {metric}")


def _jepa_retrieval_loss(
    model: torch.nn.Module,
    patient_pre: torch.Tensor,
    task_idxs: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_model = model.module if isinstance(model, DDP) else model
    feat = torch.nn.functional.normalize(patient_pre.float(), p=2, dim=-1)
    task_id = task_idxs[0]
    query = raw_model.encode_disease_teacher(task_id.unsqueeze(0).to(feat.device)).squeeze(0)
    scores = (feat @ query) / temperature
    pos_mask = labels == 1
    if not pos_mask.any():
        return feat.new_zeros(()), {
            "q_pos": float("nan"),
            "q_neg": scores[~pos_mask].mean().item() if (~pos_mask).any() else float("nan"),
        }
    loss = -torch.logsumexp(scores[pos_mask], dim=0) + torch.logsumexp(scores, dim=0)
    return loss, {
        "q_pos": scores[pos_mask].mean().item(),
        "q_neg": scores[~pos_mask].mean().item() if (~pos_mask).any() else float("nan"),
    }


def _jepa_positive_mse_loss(
    model: torch.nn.Module,
    patient_pre: torch.Tensor,
    task_idxs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    raw_model = model.module if isinstance(model, DDP) else model
    pos_mask = labels == 1
    if not pos_mask.any():
        return patient_pre.new_zeros(())

    pos_pred = torch.nn.functional.normalize(patient_pre[pos_mask].float(), p=2, dim=-1)
    task_id = task_idxs[0]
    teacher_query = raw_model.encode_disease_teacher(task_id.unsqueeze(0).to(patient_pre.device))
    teacher_query = torch.nn.functional.normalize(teacher_query.float(), p=2, dim=-1)
    teacher_query = teacher_query.expand(pos_pred.size(0), -1)
    return torch.mean((pos_pred - teacher_query) ** 2)


def _mse_margin_loss(
    model: torch.nn.Module,
    patient_pre: torch.Tensor,
    task_idxs: torch.Tensor,
    labels: torch.Tensor,
    neg_margin_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_model = model.module if isinstance(model, DDP) else model
    patient_pred = torch.nn.functional.normalize(patient_pre.float(), p=2, dim=-1)
    task_id = task_idxs[0]
    teacher_query = raw_model.encode_disease_teacher(task_id.unsqueeze(0).to(patient_pre.device))
    teacher_query = torch.nn.functional.normalize(teacher_query.float(), p=2, dim=-1)
    teacher_query = teacher_query.expand(patient_pred.size(0), -1)

    dists = torch.norm(patient_pred - teacher_query, dim=-1)
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_loss = ((patient_pred[pos_mask] - teacher_query[pos_mask]) ** 2).mean() if pos_mask.any() else patient_pre.new_zeros(())
    neg_cos = torch.nn.functional.cosine_similarity(
        patient_pred[neg_mask],
        teacher_query[neg_mask],
        dim=-1,
    ).mean() if neg_mask.any() else patient_pre.new_zeros(())
    loss = pos_loss + neg_margin_weight * neg_cos
    stats = {
        "q_pos": (-dists[pos_mask]).mean().item() if pos_mask.any() else float("nan"),
        "q_neg": (-dists[neg_mask]).mean().item() if neg_mask.any() else float("nan"),
        "pos_mse": pos_loss.item(),
        "neg_cos": neg_cos.item(),
    }
    return loss, stats


def _signed_mse_loss(
    model: torch.nn.Module,
    patient_pre: torch.Tensor,
    task_idxs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_model = model.module if isinstance(model, DDP) else model
    patient_pred = torch.nn.functional.normalize(patient_pre.float(), p=2, dim=-1)
    task_id = task_idxs[0]
    teacher_query = raw_model.encode_disease_teacher(task_id.unsqueeze(0).to(patient_pre.device))
    teacher_query = torch.nn.functional.normalize(teacher_query.float(), p=2, dim=-1)
    teacher_query = teacher_query.expand(patient_pred.size(0), -1)

    sign = torch.where(
        labels.view(-1, 1) == 1,
        torch.ones_like(patient_pred),
        -torch.ones_like(patient_pred),
    )
    targets = sign * teacher_query
    loss = torch.mean((patient_pred - targets) ** 2)

    dists = torch.norm(patient_pred - teacher_query, dim=-1)
    pos_mask = labels == 1
    neg_mask = labels == 0
    stats = {
        "q_pos": (-dists[pos_mask]).mean().item() if pos_mask.any() else float("nan"),
        "q_neg": (-dists[neg_mask]).mean().item() if neg_mask.any() else float("nan"),
        "pos_mse": torch.mean((patient_pred[pos_mask] - teacher_query[pos_mask]) ** 2).item() if pos_mask.any() else float("nan"),
        "neg_mse_to_neg_query": torch.mean((patient_pred[neg_mask] + teacher_query[neg_mask]) ** 2).item() if neg_mask.any() else float("nan"),
    }
    return loss, stats


def _negative_only_mse_loss(
    model: torch.nn.Module,
    patient_pre: torch.Tensor,
    task_idxs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_model = model.module if isinstance(model, DDP) else model
    patient_pred = torch.nn.functional.normalize(patient_pre.float(), p=2, dim=-1)
    task_id = task_idxs[0]
    teacher_query = raw_model.encode_disease_teacher(task_id.unsqueeze(0).to(patient_pre.device))
    teacher_query = torch.nn.functional.normalize(teacher_query.float(), p=2, dim=-1)
    teacher_query = teacher_query.expand(patient_pred.size(0), -1)

    neg_mask = labels == 0
    pos_mask = labels == 1
    neg_mse = torch.mean((patient_pred[neg_mask] - teacher_query[neg_mask]) ** 2) if neg_mask.any() else patient_pre.new_zeros(())
    dists = torch.norm(patient_pred - teacher_query, dim=-1)
    stats = {
        "q_pos": (-dists[pos_mask]).mean().item() if pos_mask.any() else float("nan"),
        "q_neg": (-dists[neg_mask]).mean().item() if neg_mask.any() else float("nan"),
        "neg_mse": neg_mse.item(),
    }
    return neg_mse, stats


@torch.inference_mode()
def evaluate_jepa(
    model: torch.nn.Module,
    *,
    eval_triplet_path: Path,
    eval_query_pair_path: Path | None,
    eval_data_paths: list[str] | None,
    store: EmbeddingStore,
    args,
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    negative_query_semantics = args.objective == "negative_only_mse"

    import pandas as pd

    df = pd.read_parquet(str(eval_triplet_path))
    entries = []
    for row in df.itertuples(index=False):
        t = int(row.task_idx)
        entries.append((row.anchor_eids, t))
        entries.append((row.positive_eids, t))
        entries.append((row.negative_eids, t))

    eval_ds = EvalBatchDataset(entries, store, args.eval_batch_size, args.pad_to_num_events)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)

    all_embs = []
    all_pre = []
    all_raw = []
    for batch in tqdm(eval_dl, desc="Evaluating", dynamic_ncols=True):
        event_embs = batch["event_embs"].to(device)
        event_mask = batch["event_mask"].to(device)
        emb, pre = raw_model.encode_patient_online(event_embs, event_mask, return_pre=True)
        mask_f = event_mask.float().unsqueeze(-1)
        raw_pre = (event_embs.float() * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        all_embs.append(emb.cpu())
        all_pre.append(pre.cpu())
        all_raw.append(raw_pre.cpu())

    embs = torch.cat(all_embs, dim=0)
    pre_embs = torch.cat(all_pre, dim=0)
    raw_pre_embs = torch.cat(all_raw, dim=0)

    correct_main = _triplet_correct(embs, "l2")
    correct_pre_cos = _triplet_correct(pre_embs, "cosine")
    correct_pre_l2 = _triplet_correct(pre_embs, "l2")
    correct_raw_cos = _triplet_correct(raw_pre_embs, "cosine")
    correct_raw_l2 = _triplet_correct(raw_pre_embs, "l2")

    task_idxs_arr = df["task_idx"].values
    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    task_acc: dict[str, float] = {}
    extra_overall = {
        "pre_cosine_triplet_acc": correct_pre_cos.float().mean().item(),
        "pre_l2_triplet_acc": correct_pre_l2.float().mean().item(),
        "raw_cosine_triplet_acc": correct_raw_cos.float().mean().item(),
        "raw_l2_triplet_acc": correct_raw_l2.float().mean().item(),
    }
    extra_task: dict[str, dict[str, float]] = {}

    for t_idx in sorted(set(task_idxs_arr.tolist())):
        mask = torch.from_numpy(task_idxs_arr == t_idx)
        task_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
        task_acc[task_name] = correct_main[mask].float().mean().item()
        extra_task[task_name] = {
            "pre_cosine_triplet_acc": correct_pre_cos[mask].float().mean().item(),
            "pre_l2_triplet_acc": correct_pre_l2[mask].float().mean().item(),
            "raw_cosine_triplet_acc": correct_raw_cos[mask].float().mean().item(),
            "raw_l2_triplet_acc": correct_raw_l2[mask].float().mean().item(),
        }

    if eval_query_pair_path is not None and eval_query_pair_path.exists():
        qdf = pd.read_parquet(str(eval_query_pair_path), columns=["task_idx", "positive_eids", "negative_eids"])
        q_entries = []
        for row in qdf.itertuples(index=False):
            t = int(row.task_idx)
            q_entries.append((row.positive_eids, t))
            q_entries.append((row.negative_eids, t))
        q_ds = EvalBatchDataset(q_entries, store, args.eval_batch_size, args.pad_to_num_events)
        q_dl = DataLoader(q_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)

        q_pre, q_tasks = [], []
        for batch in q_dl:
            _, pre = raw_model.encode_patient_online(
                batch["event_embs"].to(device),
                batch["event_mask"].to(device),
                return_pre=True,
            )
            q_pre.append(pre.cpu())
            q_tasks.append(batch["task_idxs"].cpu())
        pair_pre = torch.cat(q_pre, dim=0)
        pair_task_idxs = torch.cat(q_tasks, dim=0)
        pos_pre = pair_pre[0::2]
        neg_pre = pair_pre[1::2]
        pair_tasks = pair_task_idxs[0::2]

        all_task_idx = torch.arange(raw_model.task_text_embs.size(0), device=device, dtype=torch.long)
        q_vecs = raw_model.encode_disease_teacher(all_task_idx).cpu()[pair_tasks]
        sim_pos = torch.nn.functional.cosine_similarity(q_vecs, pos_pre, dim=1)
        sim_neg = torch.nn.functional.cosine_similarity(q_vecs, neg_pre, dim=1)
        q_cos_correct = sim_neg > sim_pos if negative_query_semantics else sim_pos > sim_neg
        d_pos = (q_vecs - pos_pre).norm(dim=1)
        d_neg = (q_vecs - neg_pre).norm(dim=1)
        q_l2_correct = d_neg < d_pos if negative_query_semantics else d_pos < d_neg
        extra_overall["query_cosine_pair_acc"] = q_cos_correct.float().mean().item()
        extra_overall["query_l2_pair_acc"] = q_l2_correct.float().mean().item()
        for t_idx in sorted(set(pair_tasks.tolist())):
            mask = pair_tasks == t_idx
            task_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
            extra_task.setdefault(task_name, {})
            extra_task[task_name]["query_cosine_pair_acc"] = q_cos_correct[mask].float().mean().item()
            extra_task[task_name]["query_l2_pair_acc"] = q_l2_correct[mask].float().mean().item()

    if eval_data_paths:
        import numpy as np
        import pandas as pd

        val_rows = []
        for path in eval_data_paths:
            vdf = pd.read_parquet(path, columns=["task_idx", "label", "event_ids"])
            for row in vdf.itertuples(index=False):
                val_rows.append((np.array(row.event_ids, dtype=np.int32), int(row.task_idx), int(row.label)))

        if val_rows:
            val_entries = [(eids, task_idx) for eids, task_idx, _ in val_rows]
            val_labels = torch.tensor([label for _, _, label in val_rows], dtype=torch.long)
            val_task_idxs = torch.tensor([task_idx for _, task_idx, _ in val_rows], dtype=torch.long)
            val_ds = EvalBatchDataset(val_entries, store, args.eval_batch_size, args.pad_to_num_events)
            val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)

            all_val_pre = []
            for batch in val_dl:
                _, pre = raw_model.encode_patient_online(
                    batch["event_embs"].to(device),
                    batch["event_mask"].to(device),
                    return_pre=True,
                )
                all_val_pre.append(pre.cpu())
            val_pre = torch.cat(all_val_pre, dim=0)

            all_task_idx = torch.arange(raw_model.task_text_embs.size(0), device=device, dtype=torch.long)
            query_bank = raw_model.encode_disease_teacher(all_task_idx).cpu()
            query_vecs = query_bank[val_task_idxs]
            query_scores = torch.nn.functional.cosine_similarity(query_vecs, val_pre, dim=1)
            query_eval_labels = (val_labels == 0).long() if negative_query_semantics else val_labels

            extra_overall["query_auc"] = _binary_roc_auc(query_scores, query_eval_labels)
            extra_overall["query_num_samples"] = float(val_labels.numel())
            extra_overall["query_num_pos"] = float((query_eval_labels == 1).sum().item())
            extra_overall["query_num_neg"] = float((query_eval_labels == 0).sum().item())
            p10, r10 = _precision_recall_at_k(query_scores, query_eval_labels, 10)
            p50, r50 = _precision_recall_at_k(query_scores, query_eval_labels, 50)
            extra_overall["query_precision_at_10"] = p10
            extra_overall["query_recall_at_10"] = r10
            extra_overall["query_precision_at_50"] = p50
            extra_overall["query_recall_at_50"] = r50

            for t_idx in sorted(set(val_task_idxs.tolist())):
                mask = val_task_idxs == t_idx
                task_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
                task_scores = query_scores[mask]
                task_labels = query_eval_labels[mask]
                extra_task.setdefault(task_name, {})
                extra_task[task_name]["query_num_samples"] = float(task_labels.numel())
                extra_task[task_name]["query_num_pos"] = float((task_labels == 1).sum().item())
                extra_task[task_name]["query_num_neg"] = float((task_labels == 0).sum().item())
                extra_task[task_name]["query_auc"] = _binary_roc_auc(task_scores, task_labels)
                tp10, tr10 = _precision_recall_at_k(task_scores, task_labels, 10)
                tp50, tr50 = _precision_recall_at_k(task_scores, task_labels, 50)
                extra_task[task_name]["query_precision_at_10"] = tp10
                extra_task[task_name]["query_recall_at_10"] = tr10
                extra_task[task_name]["query_precision_at_50"] = tp50
                extra_task[task_name]["query_recall_at_50"] = tr50

    overall_acc = correct_main.float().mean().item()
    raw_model.train()
    return overall_acc, task_acc, extra_overall, extra_task


def _log_eval(prefix: str, val_acc: float, task_acc: dict[str, float], extra_overall: dict[str, float], extra_task: dict[str, dict[str, float]]):
    logger.info("%s val triplet accuracy: %.4f", prefix, val_acc)
    logger.info("    [pre_emb cosine] %.4f", extra_overall["pre_cosine_triplet_acc"])
    logger.info("    [pre_emb l2] %.4f", extra_overall["pre_l2_triplet_acc"])
    logger.info("    [raw_event cosine] %.4f", extra_overall["raw_cosine_triplet_acc"])
    logger.info("    [raw_event l2] %.4f", extra_overall["raw_l2_triplet_acc"])
    if "query_cosine_pair_acc" in extra_overall:
        logger.info("    [disease_query cosine] %.4f", extra_overall["query_cosine_pair_acc"])
        logger.info("    [disease_query l2] %.4f", extra_overall["query_l2_pair_acc"])
    if "query_auc" in extra_overall:
        logger.info("    [disease_query auc] %.4f", extra_overall["query_auc"])
        logger.info(
            "    [disease_query counts] n=%d  pos=%d  neg=%d",
            int(extra_overall["query_num_samples"]),
            int(extra_overall["query_num_pos"]),
            int(extra_overall["query_num_neg"]),
        )
        logger.info(
            "    [disease_query p@10/r@10] %.4f/%.4f",
            extra_overall["query_precision_at_10"],
            extra_overall["query_recall_at_10"],
        )
        logger.info(
            "    [disease_query p@50/r@50] %.4f/%.4f",
            extra_overall["query_precision_at_50"],
            extra_overall["query_recall_at_50"],
        )

    for task, acc in task_acc.items():
        logger.info("    %s: %.4f", task, acc)
        logger.info(
            "      pre_cos=%.4f  pre_l2=%.4f  raw_cos=%.4f  raw_l2=%.4f",
            extra_task[task]["pre_cosine_triplet_acc"],
            extra_task[task]["pre_l2_triplet_acc"],
            extra_task[task]["raw_cosine_triplet_acc"],
            extra_task[task]["raw_l2_triplet_acc"],
        )
        if "query_auc" in extra_task[task]:
            logger.info(
                "      query_cos=%.4f  query_l2=%.4f  query_auc=%.4f  p@10=%.4f  r@10=%.4f  p@50=%.4f  r@50=%.4f",
                extra_task[task].get("query_cosine_pair_acc", float("nan")),
                extra_task[task].get("query_l2_pair_acc", float("nan")),
                extra_task[task]["query_auc"],
                extra_task[task]["query_precision_at_10"],
                extra_task[task]["query_recall_at_10"],
                extra_task[task]["query_precision_at_50"],
                extra_task[task]["query_recall_at_50"],
            )


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_ddp:
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if not args.eval_only and args.train_data_dir is None:
        raise ValueError("--train_data_dir is required unless --eval_only is set.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run dir: %s", run_dir)
        logger.info("World size: %s", world_size)

    use_wandb = (args.wandb_project is not None) and (rank == 0)
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or timestamp,
            tags=args.wandb_tags,
            config=vars(args),
            dir=str(run_dir),
            settings=wandb.Settings(console="wrap"),
        )
        logger.info("wandb run: %s", wandb_run.url)

    store = EmbeddingStore(args.bert_embeddings)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    task_text_embs = build_task_text_embs(args, device, rank, is_ddp)

    if args.checkpoint:
        logger.info("Loading checkpoint from %s", args.checkpoint)
        model = DiseasePatientJEPAModel.load_checkpoint(
            Path(args.checkpoint),
            task_text_embs=task_text_embs,
            device=device,
            dtype=dtype,
        )
    else:
        model = DiseasePatientJEPAModel(
            bert_dim=BERT_DIM,
            task_text_embs=task_text_embs,
            shallow_encoder_type=args.shallow_encoder_type,
            shallow_num_layers=args.shallow_num_layers,
            shallow_num_heads=args.shallow_num_heads,
            shallow_intermediate_size=args.shallow_intermediate_size,
            predictor_layers=args.predictor_layers,
            predictor_intermediate_size=args.predictor_intermediate_size,
            dtype=dtype,
        ).to(device)

    if rank == 0:
        logger.info(
            "JEPA model: encoder=%s layers=%d heads=%d predictor_layers=%d ema=%.4f objective=%s dtype=%s",
            args.shallow_encoder_type,
            args.shallow_num_layers,
            args.shallow_num_heads,
            args.predictor_layers,
            args.ema_decay,
            args.objective,
            dtype,
        )

    if args.compile and rank == 0:
        logger.warning(
            "--compile is currently ignored in train_disease_jepa.py because the training "
            "loop directly calls custom encoder methods instead of only model.forward()."
        )

    if is_ddp and not args.eval_only:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, static_graph=True)

    eval_triplet_path = None
    eval_query_pair_path = None
    if rank == 0 and args.eval_data_paths:
        eval_triplet_path = run_dir / "eval_triplets.parquet"
        precompute_eval_triplets(args.eval_data_paths, args.n_eval_triplets_per_task, args.seed, eval_triplet_path)
        eval_query_pair_path = run_dir / "eval_query_pairs.parquet"
        precompute_eval_query_pairs(args.eval_data_paths, args.n_eval_triplets_per_task, args.seed, eval_query_pair_path)
    if is_ddp:
        obj = [eval_triplet_path, eval_query_pair_path]
        dist.broadcast_object_list(obj, src=0)
        eval_triplet_path, eval_query_pair_path = obj

    if args.eval_only:
        if args.eval_data_paths is None:
            raise ValueError("--eval_only requires --eval_data_paths")
        val_acc, task_acc, extra_overall, extra_task = evaluate_jepa(
            model,
            eval_triplet_path=eval_triplet_path,
            eval_query_pair_path=eval_query_pair_path,
            eval_data_paths=args.eval_data_paths,
            store=store,
            args=args,
            device=device,
        )
        if rank == 0:
            _log_eval("", val_acc, task_acc, extra_overall, extra_task)
        if is_ddp:
            dist.destroy_process_group()
        return

    first_task = sorted(args.tasks)[0]
    task_dir = Path(args.train_data_dir) / first_task
    data_epoch_files = sorted(task_dir.glob("train_prepared_*.parquet"))
    available_data_epochs = sorted(int(p.stem.split("_")[-1]) for p in data_epoch_files)
    if not available_data_epochs:
        raise ValueError(f"No train_prepared_*.parquet found in {task_dir}")
    if args.train_data_epochs is not None:
        selected = list(dict.fromkeys(args.train_data_epochs))
        missing = [ep for ep in selected if ep not in available_data_epochs]
        if missing:
            raise ValueError(f"Requested epochs {missing}, available={available_data_epochs}")
        selected_data_epochs = selected
    else:
        selected_data_epochs = available_data_epochs[: args.epochs]

    if rank == 0:
        logger.info("Selected train data epoch(s): %s", selected_data_epochs)

    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )

    schedule_lengths = []
    for schedule_idx, data_epoch in enumerate(selected_data_epochs):
        ds = EpochBatchDataset(
            args.train_data_dir, data_epoch, args.tasks, store,
            args.batch_size, schedule_idx, args.seed, world_size, rank,
            args.pad_to_num_events,
        )
        schedule_lengths.append(len(ds))
    total_steps = sum(math.ceil(n / args.grad_accum) for n in schedule_lengths)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)

    global_step = 0
    best_auc = float("-inf")

    for schedule_idx, data_epoch in enumerate(selected_data_epochs):
        if rank == 0:
            logger.info("Pass %d/%d: prepared data epoch %d …", schedule_idx + 1, len(selected_data_epochs), data_epoch)

        epoch_ds = EpochBatchDataset(
            args.train_data_dir, data_epoch, args.tasks, store,
            args.batch_size, schedule_idx, args.seed, world_size, rank,
            args.pad_to_num_events,
        )
        train_loader = DataLoader(
            epoch_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda b: b[0],
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_epoch_worker_init,
            persistent_workers=False,
        )

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {schedule_idx + 1}/{len(selected_data_epochs)}", disable=(rank != 0), dynamic_ncols=True, total=len(epoch_ds))

        for batch_idx, batch in enumerate(pbar):
            labels_t = batch["labels"]
            task_idxs_t = batch["task_idxs"].to(device)
            is_update_step = ((batch_idx + 1) % args.grad_accum == 0) or ((batch_idx + 1) == len(epoch_ds))
            sync_ctx = nullcontext() if (is_update_step or not is_ddp) else model.no_sync()

            with sync_ctx:
                emb, pre_emb = raw_model.encode_patient_online(
                    batch["event_embs"].to(device),
                    batch["event_mask"].to(device),
                    return_pre=True,
                )
                if args.objective == "retrieval_jepa":
                    retrieval_loss, retrieval_stats = _jepa_retrieval_loss(
                        model,
                        pre_emb,
                        task_idxs_t,
                        labels_t.to(device),
                        args.retrieval_temperature,
                    )
                    jepa_mse = _jepa_positive_mse_loss(
                        model,
                        pre_emb,
                        task_idxs_t,
                        labels_t.to(device),
                    )
                    supcon_aux = _supervised_contrastive_loss(pre_emb, labels_t.to(device), args.stage1_temperature)
                    main_loss = (
                        retrieval_loss
                        + args.jepa_mse_weight * jepa_mse
                        + args.retrieval_supcon_weight * supcon_aux
                    )
                    loss_name = "lret"
                    aux_name = "ljepa"
                    aux_value = jepa_mse
                elif args.objective == "mse_margin":
                    retrieval_loss, retrieval_stats = _mse_margin_loss(
                        model,
                        pre_emb,
                        task_idxs_t,
                        labels_t.to(device),
                        args.neg_margin_weight,
                    )
                    jepa_mse = retrieval_loss.new_zeros(())
                    supcon_aux = _supervised_contrastive_loss(pre_emb, labels_t.to(device), args.stage1_temperature)
                    main_loss = retrieval_loss + args.retrieval_supcon_weight * supcon_aux
                    loss_name = "lmse"
                    aux_name = "lneg"
                    aux_value = retrieval_loss.new_tensor(retrieval_stats["neg_cos"])
                elif args.objective == "signed_mse":
                    retrieval_loss, retrieval_stats = _signed_mse_loss(
                        model,
                        pre_emb,
                        task_idxs_t,
                        labels_t.to(device),
                    )
                    jepa_mse = retrieval_loss.new_zeros(())
                    supcon_aux = _supervised_contrastive_loss(pre_emb, labels_t.to(device), args.stage1_temperature)
                    main_loss = retrieval_loss + args.retrieval_supcon_weight * supcon_aux
                    loss_name = "lsigned"
                    aux_name = "lneg"
                    aux_value = retrieval_loss.new_tensor(retrieval_stats["neg_mse_to_neg_query"])
                else:
                    retrieval_loss, retrieval_stats = _negative_only_mse_loss(
                        model,
                        pre_emb,
                        task_idxs_t,
                        labels_t.to(device),
                    )
                    jepa_mse = retrieval_loss.new_zeros(())
                    supcon_aux = _supervised_contrastive_loss(pre_emb, labels_t.to(device), args.stage1_temperature)
                    main_loss = retrieval_loss + args.retrieval_supcon_weight * supcon_aux
                    loss_name = "lnegonly"
                    aux_name = "lneg"
                    aux_value = retrieval_loss.new_tensor(retrieval_stats["neg_mse"])
                if args.var_reg_weight > 0.0:
                    std = torch.sqrt(pre_emb.var(dim=0) + 1e-4)
                    var_loss = torch.relu(1.0 - std).mean()
                else:
                    var_loss = pre_emb.new_zeros(())
                loss = (main_loss + args.var_reg_weight * var_loss) / args.grad_accum
                loss.backward()

            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
                raw_model.update_teacher(args.ema_decay)
                if global_step < warmup_steps:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            else:
                grad_norm = float("nan")

            epoch_loss += loss.item() * args.grad_accum
            if rank == 0:
                pbar.set_postfix(
                    loss=f"{loss.item() * args.grad_accum:.4f}",
                    **{loss_name: f"{retrieval_loss.item():.4f}"},
                    **{aux_name: f"{aux_value.item():.4f}"},
                    lsup=f"{supcon_aux.item():.4f}",
                    qpos=f"{retrieval_stats['q_pos']:.3f}" if not math.isnan(retrieval_stats["q_pos"]) else "nan",
                    qneg=f"{retrieval_stats['q_neg']:.3f}" if not math.isnan(retrieval_stats["q_neg"]) else "nan",
                    prevar=f"{pre_emb.var(dim=0).mean().item():.4f}",
                    gnorm=f"{float(grad_norm):.3f}" if not isinstance(grad_norm, float) or not math.isnan(grad_norm) else "nan",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if rank == 0:
            logger.info("Epoch %d/%d avg_loss=%.4f", schedule_idx + 1, len(selected_data_epochs), epoch_loss / max(len(epoch_ds), 1))
        val_acc, task_acc, extra_overall, extra_task = evaluate_jepa(
            model,
            eval_triplet_path=eval_triplet_path,
            eval_query_pair_path=eval_query_pair_path,
            eval_data_paths=args.eval_data_paths,
            store=store,
            args=args,
            device=device,
        )
        if rank == 0:
            _log_eval("  ", val_acc, task_acc, extra_overall, extra_task)
            epoch_dir = run_dir / f"epoch_{schedule_idx + 1}"
            raw_model.save_checkpoint(epoch_dir)
            if extra_overall.get("query_auc", float("-inf")) > best_auc:
                best_auc = extra_overall["query_auc"]
                raw_model.save_checkpoint(run_dir / "best")
            if use_wandb:
                import wandb
                log_dict = {
                    "epoch": schedule_idx + 1,
                    "train/avg_loss": epoch_loss / max(len(epoch_ds), 1),
                    "eval/triplet_acc": val_acc,
                    "eval/pre_cosine_triplet_acc": extra_overall["pre_cosine_triplet_acc"],
                    "eval/pre_l2_triplet_acc": extra_overall["pre_l2_triplet_acc"],
                    "eval/raw_cosine_triplet_acc": extra_overall["raw_cosine_triplet_acc"],
                    "eval/raw_l2_triplet_acc": extra_overall["raw_l2_triplet_acc"],
                }
                if "query_cosine_pair_acc" in extra_overall:
                    log_dict["eval/query_cosine_pair_acc"] = extra_overall["query_cosine_pair_acc"]
                    log_dict["eval/query_l2_pair_acc"] = extra_overall["query_l2_pair_acc"]
                if "query_auc" in extra_overall:
                    log_dict["eval/query_auc"] = extra_overall["query_auc"]
                    log_dict["eval/query_precision_at_10"] = extra_overall["query_precision_at_10"]
                    log_dict["eval/query_recall_at_10"] = extra_overall["query_recall_at_10"]
                    log_dict["eval/query_precision_at_50"] = extra_overall["query_precision_at_50"]
                    log_dict["eval/query_recall_at_50"] = extra_overall["query_recall_at_50"]
                wandb.log(log_dict)

    if use_wandb:
        wandb_run.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
