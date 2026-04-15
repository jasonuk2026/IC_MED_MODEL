#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from model_soft_token_classifier import DiseaseEventSoftTokenClassifier
from train_disease_concat_classifier import _log_eval
from train_embedding_disease_cond_v2 import (
    TASK_2_DISEASE_NAME,
    TASK_2_DISEASE_QUERY_TEXT,
    TASK_2_IDX,
    _binary_roc_auc,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_binary_label(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value != 0)
    if isinstance(value, (float, np.floating)):
        return int(float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return 1
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return 0
    try:
        return int(float(text) != 0.0)
    except ValueError as exc:
        raise ValueError(f"Unsupported binary label value: {value!r}") from exc


def infer_encoder_backend(event_embedding_dir: str, event_dim: int) -> tuple[str, str]:
    hint = str(event_embedding_dir).lower()
    if "qwen" in hint:
        return "qwen", "Qwen/Qwen3-Embedding-0.6B"
    if "bert" in hint or "biolink" in hint:
        return "bert", "michiyasunaga/BioLinkBERT-base"
    if event_dim == 768:
        return "bert", "michiyasunaga/BioLinkBERT-base"
    return "qwen", "Qwen/Qwen3-Embedding-0.6B"


def load_embedding_source_hint(data_dir: str, task: str, split: str) -> str | None:
    meta_path = Path(data_dir) / task / f"{split}_embedded_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("embeddings_path")


def build_task_text_embs_auto(*, event_embedding_dir: str, event_dim: int, device: torch.device, rank: int, is_ddp: bool, local_files_only: bool) -> tuple[torch.Tensor, str, str]:
    tasks_sorted = sorted(TASK_2_DISEASE_NAME)
    backend, model_name = infer_encoder_backend(event_embedding_dir, event_dim)
    disease_dim = 768 if backend == "bert" else event_dim
    task_text_embs = torch.zeros(len(tasks_sorted), disease_dim, dtype=torch.float32, device=device)
    if rank == 0:
        logger.info("Loading disease text encoder backend=%s model=%s", backend, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        model.eval()
        texts = [TASK_2_DISEASE_QUERY_TEXT[t] for t in tasks_sorted]
        tok_kwargs = dict(padding=True, truncation=True, max_length=128, return_tensors="pt")
        if backend == "qwen":
            tok_kwargs["add_special_tokens"] = False
        enc = tokenizer(texts, **tok_kwargs).to(device)
        out = model(**enc)
        if backend == "bert":
            special_ids = set(tokenizer.all_special_ids)
            special = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
            for sid in special_ids:
                special |= enc["input_ids"] == sid
            pool_mask = enc["attention_mask"].bool() & ~special
            pool_mask_f = pool_mask.float().unsqueeze(-1)
            task_text_embs = ((out.last_hidden_state.float() * pool_mask_f).sum(dim=1) / pool_mask_f.sum(dim=1).clamp(min=1e-9))
        else:
            mask = enc["attention_mask"].float().unsqueeze(-1)
            task_text_embs = ((out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9))
    if is_ddp:
        dist.broadcast(task_text_embs, src=0)
    return task_text_embs.cpu(), backend, model_name


def find_epoch_file(task_dir: Path, split: str, epoch_idx: int) -> Path:
    candidate = task_dir / f"{split}_embedded_{epoch_idx:03d}.parquet"
    if candidate.exists():
        return candidate
    if split != "train":
        fallback = task_dir / f"{split}_embedded_000.parquet"
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Missing embedded parquet for split={split} epoch={epoch_idx}: {candidate}")


def infer_shape_from_embedded_parquet(data_dir: str, task: str, split: str, epoch_idx: int) -> tuple[int, int]:
    path = find_epoch_file(Path(data_dir) / task, split, epoch_idx)
    df = pd.read_parquet(path, columns=["event_embs"])
    if len(df) == 0:
        raise ValueError(f"Empty embedded parquet: {path}")
    arr = np.asarray(df.iloc[0]["event_embs"], dtype=np.float32)
    return int(arr.shape[0]), int(arr.shape[1])


class PreEmbeddedBatchDataset(Dataset):
    def __init__(self, data_dir: str, split: str, data_epoch_idx: int, tasks: list[str], batch_size: int, training_epoch: int, seed: int, world_size: int, rank: int):
        rows: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        for task in sorted(tasks):
            path = find_epoch_file(Path(data_dir) / task, split, data_epoch_idx)
            df = pd.read_parquet(path, columns=["label", "event_mask", "event_embs"])
            for row in df.itertuples(index=False):
                rows.append((np.asarray(row.event_embs, dtype=np.float32), np.asarray(row.event_mask, dtype=np.int64), int(TASK_2_IDX[task]), parse_binary_label(row.label)))
        if split == "train":
            rng = random.Random(seed + training_epoch * 1337)
            rng.shuffle(rows)
        if world_size > 1:
            rows = rows[rank::world_size]
        self.samples = rows
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.batch_size
        end = min(len(self.samples), start + self.batch_size)
        sub = self.samples[start:end]
        return {
            "event_embs": torch.from_numpy(np.stack([x[0] for x in sub], axis=0)),
            "event_mask": torch.from_numpy(np.stack([x[1] for x in sub], axis=0)),
            "task_idxs": torch.tensor([x[2] for x in sub], dtype=torch.long),
            "labels": torch.tensor([x[3] for x in sub], dtype=torch.long),
        }


@torch.inference_mode()
def evaluate_classifier(model: torch.nn.Module, *, eval_data_dir: str, eval_split: str, eval_epoch_idx: int, tasks: list[str], args, device: torch.device) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rows: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for task in sorted(tasks):
        path = find_epoch_file(Path(eval_data_dir) / task, eval_split, eval_epoch_idx)
        df = pd.read_parquet(path, columns=["label", "event_mask", "event_embs"])
        for row_idx, row in enumerate(df.itertuples(index=False)):
            if row_idx % world_size != rank:
                continue
            rows.append((np.asarray(row.event_embs, dtype=np.float32), np.asarray(row.event_mask, dtype=np.int64), int(TASK_2_IDX[task]), parse_binary_label(row.label)))
    all_logits = []
    all_labels = []
    all_task_idxs = []
    for start in tqdm(range(0, len(rows), args.eval_batch_size), desc=f"Evaluating {eval_split}", disable=(rank != 0), dynamic_ncols=True):
        sub = rows[start : start + args.eval_batch_size]
        batch = {
            "event_embs": torch.from_numpy(np.stack([x[0] for x in sub], axis=0)),
            "event_mask": torch.from_numpy(np.stack([x[1] for x in sub], axis=0)),
            "task_idxs": torch.tensor([x[2] for x in sub], dtype=torch.long),
            "labels": torch.tensor([x[3] for x in sub], dtype=torch.long),
        }
        logits, _, _, _ = raw_model(batch["event_embs"].to(device), batch["event_mask"].to(device), batch["task_idxs"].to(device), return_aux_logits=True)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())
        all_task_idxs.append(batch["task_idxs"].cpu())
    local_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, dtype=torch.float32)
    local_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    local_task_idxs = torch.cat(all_task_idxs, dim=0) if all_task_idxs else torch.empty(0, dtype=torch.long)
    if world_size > 1:
        gathered: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, (local_logits.numpy(), local_labels.numpy(), local_task_idxs.numpy()))
        logits_parts = [torch.from_numpy(x[0]) for x in gathered if x is not None and len(x[0]) > 0]
        labels_parts = [torch.from_numpy(x[1]) for x in gathered if x is not None and len(x[1]) > 0]
        task_parts = [torch.from_numpy(x[2]) for x in gathered if x is not None and len(x[2]) > 0]
        logits = torch.cat(logits_parts, dim=0) if logits_parts else torch.empty(0, dtype=torch.float32)
        labels = torch.cat(labels_parts, dim=0) if labels_parts else torch.empty(0, dtype=torch.long)
        task_idxs = torch.cat(task_parts, dim=0) if task_parts else torch.empty(0, dtype=torch.long)
    else:
        logits = local_logits
        labels = local_labels
        task_idxs = local_task_idxs
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    overall = {"auc": _binary_roc_auc(probs, labels), "accuracy": (preds == labels).float().mean().item(), "num_samples": float(labels.numel()), "num_pos": float((labels == 1).sum().item()), "num_neg": float((labels == 0).sum().item()), "pos_prob_mean": probs[labels == 1].mean().item() if (labels == 1).any() else float("nan"), "neg_prob_mean": probs[labels == 0].mean().item() if (labels == 0).any() else float("nan")}
    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    per_task: dict[str, dict[str, float]] = {}
    for t_idx in sorted(set(task_idxs.tolist())):
        mask = task_idxs == t_idx
        task_name = idx_to_name.get(int(t_idx), str(int(t_idx)))
        task_probs = probs[mask]
        task_labels = labels[mask]
        task_preds = preds[mask]
        per_task[task_name] = {"auc": _binary_roc_auc(task_probs, task_labels), "accuracy": (task_preds == task_labels).float().mean().item(), "num_samples": float(task_labels.numel()), "num_pos": float((task_labels == 1).sum().item()), "num_neg": float((task_labels == 0).sum().item()), "pos_prob_mean": task_probs[task_labels == 1].mean().item() if (task_labels == 1).any() else float("nan"), "neg_prob_mean": task_probs[task_labels == 0].mean().item() if (task_labels == 0).any() else float("nan")}
    overall["macro_auc"] = float(np.mean([x["auc"] for x in per_task.values()])) if per_task else float("nan")
    overall["macro_accuracy"] = float(np.mean([x["accuracy"] for x in per_task.values()])) if per_task else float("nan")
    raw_model.train()
    return overall, per_task


def parse_args():
    p = argparse.ArgumentParser(description="Soft-token classifier on pre-embedded task parquet files", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--train_data_dir", default="embedded_task_data")
    p.add_argument("--train_data_epochs", nargs="+", type=int, default=None)
    p.add_argument("--tasks", nargs="+", default=list(sorted(TASK_2_DISEASE_NAME.keys())))
    p.add_argument("--eval_data_dir", default="embedded_task_data")
    p.add_argument("--eval_split", default="val", choices=["train", "val", "test"])
    p.add_argument("--eval_data_epoch", type=int, default=0)
    p.add_argument("--event_embedding_dir", required=True)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=None)
    p.add_argument("--head_layers", type=int, default=1)
    p.add_argument("--max_positions", type=int, default=None)
    p.add_argument("--position_type", choices=["learned", "rotary"], default="learned")
    p.add_argument("--attention_type", choices=["bidirectional", "causal"], default="bidirectional")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--output_dir", default="output/disease-soft-token-classifier-preembedded")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--aux_loss_weight", type=float, default=0.0)
    p.add_argument("--align_loss_weight", type=float, default=0.0)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)
    return p.parse_args()


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
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run dir: %s", run_dir)
    first_task = sorted(args.tasks)[0]
    max_events, event_dim = infer_shape_from_embedded_parquet(args.train_data_dir, first_task, "train", 0)
    source_hint = load_embedding_source_hint(args.train_data_dir, first_task, "train")
    embedding_source = source_hint or args.event_embedding_dir
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    task_text_embs, disease_backend, disease_model_name = build_task_text_embs_auto(event_embedding_dir=embedding_source, event_dim=event_dim, device=device, rank=rank, is_ddp=is_ddp, local_files_only=args.local_files_only)
    disease_dim = int(task_text_embs.shape[1])
    if rank == 0:
        logger.info("Detected preembedded shape: max_events=%d event_dim=%d disease_backend=%s disease_model=%s disease_dim=%d", max_events, event_dim, disease_backend, disease_model_name, disease_dim)
        logger.info("Embedding source hint: %s", embedding_source)
    use_wandb = (args.wandb_project is not None) and (rank == 0)
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name or timestamp, tags=args.wandb_tags, config={**vars(args), "disease_backend": disease_backend, "disease_model_name": disease_model_name}, dir=str(run_dir), settings=wandb.Settings(console="wrap"))
    if args.checkpoint:
        model = DiseaseEventSoftTokenClassifier.load_checkpoint(Path(args.checkpoint), task_text_embs=task_text_embs, device=device, dtype=dtype)
    else:
        model = DiseaseEventSoftTokenClassifier(event_dim=event_dim, task_text_embs=task_text_embs, disease_dim=disease_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads, intermediate_size=args.intermediate_size, head_layers=args.head_layers, max_positions=args.max_positions or max_events, position_type=args.position_type, attention_type=args.attention_type, dropout=args.dropout, dtype=dtype).to(device)
    if is_ddp and not args.eval_only:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, static_graph=True)
    if args.eval_only:
        overall, per_task = evaluate_classifier(model, eval_data_dir=args.eval_data_dir, eval_split=args.eval_split, eval_epoch_idx=args.eval_data_epoch, tasks=args.tasks, args=args, device=device)
        if rank == 0:
            _log_eval("", overall, per_task)
        if is_ddp:
            dist.destroy_process_group()
        return
    task_dir = Path(args.train_data_dir) / first_task
    data_epoch_files = sorted(task_dir.glob("train_embedded_*.parquet"))
    available_epochs = sorted(int(p.stem.split("_")[-1]) for p in data_epoch_files)
    if not available_epochs:
        raise ValueError(f"No train_embedded_*.parquet found in {task_dir}")
    selected_epochs = list(dict.fromkeys(args.train_data_epochs)) if args.train_data_epochs is not None else available_epochs[: args.epochs]
    if rank == 0:
        logger.info("Selected train data epoch(s): %s", selected_epochs)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay, fused=True)
    schedule_lengths = []
    for schedule_idx, data_epoch in enumerate(selected_epochs):
        ds = PreEmbeddedBatchDataset(args.train_data_dir, "train", data_epoch, args.tasks, args.batch_size, schedule_idx, args.seed, world_size, rank)
        schedule_lengths.append(len(ds))
    total_steps = sum(math.ceil(n / args.grad_accum) for n in schedule_lengths)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    pos_weight = torch.tensor(args.pos_weight, device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    global_step = 0
    best_auc = float("-inf")
    for schedule_idx, data_epoch in enumerate(selected_epochs):
        if rank == 0:
            logger.info("Pass %d/%d: embedded data epoch %d …", schedule_idx + 1, len(selected_epochs), data_epoch)
        epoch_ds = PreEmbeddedBatchDataset(args.train_data_dir, "train", data_epoch, args.tasks, args.batch_size, schedule_idx, args.seed, world_size, rank)
        train_loader = DataLoader(epoch_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=args.num_workers, prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None, pin_memory=torch.cuda.is_available(), persistent_workers=False)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {schedule_idx + 1}/{len(selected_epochs)}", disable=(rank != 0), dynamic_ncols=True, total=len(epoch_ds))
        for batch_idx, batch in enumerate(pbar):
            labels = batch["labels"].to(device).float()
            logits, aux_logits, disease_hidden, event_pooled = model(batch["event_embs"].to(device), batch["event_mask"].to(device), batch["task_idxs"].to(device), return_aux_logits=True)
            main_loss = loss_fn(logits, labels)
            aux_loss = loss_fn(aux_logits, labels)
            align_loss = F.mse_loss(F.normalize(event_pooled.float(), dim=-1), F.normalize(disease_hidden.float(), dim=-1))
            loss = (main_loss + args.aux_loss_weight * aux_loss + args.align_loss_weight * align_loss) / args.grad_accum
            loss.backward()
            is_update_step = ((batch_idx + 1) % args.grad_accum == 0) or ((batch_idx + 1) == len(epoch_ds))
            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                optimizer.step()
                if global_step < warmup_steps:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            else:
                grad_norm = float("nan")
            epoch_loss += loss.item() * args.grad_accum
            if rank == 0:
                probs = torch.sigmoid(logits.float())
                pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}", lmain=f"{main_loss.item():.4f}", laux=f"{aux_loss.item():.4f}", lalign=f"{align_loss.item():.4f}", posp=f"{probs[labels == 1].mean().item():.3f}" if (labels == 1).any() else "nan", negp=f"{probs[labels == 0].mean().item():.3f}" if (labels == 0).any() else "nan", gnorm=f"{float(grad_norm):.3f}" if not isinstance(grad_norm, float) or not math.isnan(grad_norm) else "nan", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if rank == 0:
            logger.info("Epoch %d/%d avg_loss=%.4f", schedule_idx + 1, len(selected_epochs), epoch_loss / max(len(epoch_ds), 1))
        overall, per_task = evaluate_classifier(model, eval_data_dir=args.eval_data_dir, eval_split=args.eval_split, eval_epoch_idx=args.eval_data_epoch, tasks=args.tasks, args=args, device=device)
        if rank == 0:
            _log_eval("  ", overall, per_task)
            raw_model = model.module if isinstance(model, DDP) else model
            epoch_dir = run_dir / f"epoch_{schedule_idx + 1}"
            raw_model.save_checkpoint(epoch_dir)
            if overall["auc"] > best_auc:
                best_auc = overall["auc"]
                raw_model.save_checkpoint(run_dir / "best")
            if use_wandb:
                import wandb
                log_dict = {"epoch": schedule_idx + 1, "train/avg_loss": epoch_loss / max(len(epoch_ds), 1), "eval/auc": overall["auc"], "eval/accuracy": overall["accuracy"], "eval/pos_prob_mean": overall["pos_prob_mean"], "eval/neg_prob_mean": overall["neg_prob_mean"]}
                for task, stats in per_task.items():
                    log_dict[f"eval/{task}/auc"] = stats["auc"]
                    log_dict[f"eval/{task}/accuracy"] = stats["accuracy"]
                wandb.log(log_dict)
    if use_wandb:
        wandb_run.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
