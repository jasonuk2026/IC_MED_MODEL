#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
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

GEN_META_DIR = Path(__file__).resolve().parent / "01_gen_meta"
if str(GEN_META_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_META_DIR))

from encoders import ENCODER_REGISTRY, get_encoder

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


TASK_2_DISEASE_NAME = {
    "new_hypertension": "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan": "pancreatic cancer",
    "new_celiac": "celiac disease",
    "new_lupus": "systemic lupus erythematosus",
    "new_acutemi": "acute myocardial infarction",
}

TASK_2_DISEASE_QUERY_TEXT = {
    "new_hypertension": (
        "Disease query: hypertension, also called high blood pressure. "
        "Relevant clinical clues include persistently elevated systolic or "
        "diastolic blood pressure, antihypertensive medications, chronic "
        "cardiovascular risk, kidney disease, and repeated outpatient blood "
        "pressure measurements above the normal range."
    ),
    "new_hyperlipidemia": (
        "Disease query: hyperlipidemia, including high cholesterol, elevated "
        "LDL, elevated triglycerides, and dyslipidemia. Relevant clues include "
        "abnormal lipid panel results, statin or other lipid-lowering therapy, "
        "atherosclerotic cardiovascular risk, and follow-up laboratory testing "
        "for cholesterol and triglycerides."
    ),
    "new_pancan": (
        "Disease query: pancreatic cancer, pancreatic adenocarcinoma, or "
        "malignancy of the pancreas. Relevant clues include pancreatic mass or "
        "neoplasm, biliary obstruction or jaundice, weight loss, abdominal or "
        "back pain, elevated tumor markers, imaging findings, oncologic "
        "evaluation, surgery, chemotherapy, and hospital care related to "
        "pancreatic malignancy."
    ),
    "new_celiac": (
        "Disease query: celiac disease, gluten-sensitive enteropathy, or immune "
        "mediated small bowel disease triggered by gluten. Relevant clues include "
        "malabsorption, chronic diarrhea, abdominal pain, nutritional deficiency, "
        "positive serology such as tissue transglutaminase antibodies, small bowel "
        "biopsy findings, and dietary management with gluten restriction."
    ),
    "new_lupus": (
        "Disease query: systemic lupus erythematosus, also called lupus or SLE. "
        "Relevant clues include autoimmune disease, rash, arthritis, nephritis, "
        "hematologic abnormalities, positive ANA or double-stranded DNA tests, "
        "complement abnormalities, immunosuppressive therapy, and multisystem "
        "inflammatory manifestations."
    ),
    "new_acutemi": (
        "Disease query: acute myocardial infarction, heart attack, STEMI, NSTEMI, "
        "or acute coronary syndrome with myocardial injury. Relevant clues include "
        "chest pain, ischemia, elevated troponin, ECG changes, coronary "
        "catheterization or revascularization, antiplatelet or anticoagulant "
        "therapy, and hospitalization for acute cardiac ischemia."
    ),
}

TASK_2_IDX = {task: idx for idx, task in enumerate(sorted(TASK_2_DISEASE_NAME))}


class EmbeddingStore(object):
    """Memory-mapped event embedding array."""

    def __init__(self, embeddings_path):
        logger.info("Loading embeddings (mmap) from %s ...", embeddings_path)
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        logger.info("  Embeddings shape: %s dtype: %s", self.embeddings.shape, self.embeddings.dtype)
        self.dim = int(self.embeddings.shape[1])


def _collate_event_embs(
    eids_list,
    task_idxs,
    labels,
    embeddings,
    pad_to_num_events,
):
    bert_dim = embeddings.shape[1]
    embs_list = []
    for eids in eids_list:
        if pad_to_num_events is not None:
            eids = eids[:pad_to_num_events]
        if len(eids) == 0:
            embs_list.append(np.zeros((1, bert_dim), dtype=np.float32))
        else:
            embs_list.append(embeddings[eids].astype(np.float32))

    batch_size = len(embs_list)
    max_events = pad_to_num_events if pad_to_num_events is not None else max(emb.shape[0] for emb in embs_list)
    padded = np.zeros((batch_size, max_events, bert_dim), dtype=np.float32)
    mask = np.zeros((batch_size, max_events), dtype=np.int64)
    for i, emb in enumerate(embs_list):
        n_events = emb.shape[0]
        padded[i, :n_events] = emb
        mask[i, :n_events] = 1

    out = {
        "event_embs": torch.from_numpy(padded),
        "event_mask": torch.from_numpy(mask),
        "task_idxs": torch.tensor(task_idxs, dtype=torch.long),
    }
    if labels is not None:
        out["labels"] = torch.tensor(labels, dtype=torch.long)
    return out


class SampleLevelPreparedDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_epoch_idx,
        tasks,
        store,
        batch_size,
        training_epoch,
        seed,
        world_size,
        rank,
        pad_to_num_events,
    ):
        self.embeddings = store.embeddings
        self.pad_to_num_events = pad_to_num_events
        rows = []
        for task in sorted(tasks):
            parquet_path = Path(data_dir) / task / ("train_prepared_%03d.parquet" % data_epoch_idx)
            if not parquet_path.exists():
                raise FileNotFoundError("Missing prepared parquet: %s" % parquet_path)
            df = pd.read_parquet(parquet_path, columns=["task_idx", "label", "event_ids"])
            for row in df.itertuples(index=False):
                rows.append((
                    np.array(row.event_ids, dtype=np.int32),
                    int(row.task_idx),
                    int(row.label),
                ))

        rng = random.Random(seed + training_epoch * 1337)
        rng.shuffle(rows)
        if world_size > 1:
            rows = rows[rank::world_size]

        self.samples = rows
        self.batch_size = batch_size

    def __len__(self):
        return int(math.ceil(len(self.samples) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(len(self.samples), start + self.batch_size)
        batch_rows = self.samples[start:end]
        return _collate_event_embs(
            eids_list=[x[0] for x in batch_rows],
            task_idxs=[x[1] for x in batch_rows],
            labels=[x[2] for x in batch_rows],
            embeddings=self.embeddings,
            pad_to_num_events=self.pad_to_num_events,
        )


class EvalBatchDataset(Dataset):
    def __init__(self, entries, store, batch_size, pad_to_num_events=None):
        self.store = store
        self.pad_to_num_events = pad_to_num_events
        self._batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx):
        batch = self._batches[idx]
        return _collate_event_embs(
            eids_list=[entry[0] for entry in batch],
            task_idxs=[entry[1] for entry in batch],
            labels=None,
            embeddings=self.store.embeddings,
            pad_to_num_events=self.pad_to_num_events,
        )


def _binary_roc_auc(scores, labels):
    scores = scores.float().cpu()
    labels = labels.long().cpu()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    auc = (cmp > 0).float().mean() + 0.5 * (cmp == 0).float().mean()
    return auc.item()


@torch.inference_mode()
def evaluate_classifier(model, *, eval_data_paths, store, args, device):
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    rows = []
    for path in eval_data_paths:
        df = pd.read_parquet(path, columns=["task_idx", "label", "event_ids"])
        for row in df.itertuples(index=False):
            rows.append((np.array(row.event_ids, dtype=np.int32), int(row.task_idx), int(row.label)))

    entries = [(eids, task_idx) for eids, task_idx, _ in rows]
    labels = torch.tensor([label for _, _, label in rows], dtype=torch.long)
    task_idxs = torch.tensor([task_idx for _, task_idx, _ in rows], dtype=torch.long)
    eval_ds = EvalBatchDataset(entries, store, args.eval_batch_size, args.pad_to_num_events)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0], num_workers=0)

    all_logits = []
    for batch in tqdm(eval_dl, desc="Evaluating", dynamic_ncols=True):
        logits = raw_model(
            batch["event_embs"].to(device),
            batch["event_mask"].to(device),
            batch["task_idxs"].to(device),
        )
        all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    overall = {
        "auc": _binary_roc_auc(probs, labels),
        "accuracy": (preds == labels).float().mean().item(),
        "num_samples": float(labels.numel()),
        "num_pos": float((labels == 1).sum().item()),
        "num_neg": float((labels == 0).sum().item()),
        "pos_prob_mean": probs[labels == 1].mean().item() if (labels == 1).any() else float("nan"),
        "neg_prob_mean": probs[labels == 0].mean().item() if (labels == 0).any() else float("nan"),
    }

    idx_to_name = {v: k for k, v in TASK_2_IDX.items()}
    per_task = {}
    for task_idx in sorted(set(task_idxs.tolist())):
        mask = task_idxs == task_idx
        task_name = idx_to_name.get(int(task_idx), str(int(task_idx)))
        task_probs = probs[mask]
        task_labels = labels[mask]
        task_preds = preds[mask]
        per_task[task_name] = {
            "auc": _binary_roc_auc(task_probs, task_labels),
            "accuracy": (task_preds == task_labels).float().mean().item(),
            "num_samples": float(task_labels.numel()),
            "num_pos": float((task_labels == 1).sum().item()),
            "num_neg": float((task_labels == 0).sum().item()),
            "pos_prob_mean": task_probs[task_labels == 1].mean().item() if (task_labels == 1).any() else float("nan"),
            "neg_prob_mean": task_probs[task_labels == 0].mean().item() if (task_labels == 0).any() else float("nan"),
        }

    if per_task:
        overall["macro_auc"] = float(np.mean([stats["auc"] for stats in per_task.values()]))
        overall["macro_accuracy"] = float(np.mean([stats["accuracy"] for stats in per_task.values()]))
    else:
        overall["macro_auc"] = float("nan")
        overall["macro_accuracy"] = float("nan")

    raw_model.train()
    return overall, per_task


def _log_eval(prefix, overall, per_task):
    logger.info("%s val auc: %.4f", prefix, overall["auc"])
    logger.info("%s val macro_auc: %.4f", prefix, overall["macro_auc"])
    logger.info("%s val accuracy: %.4f", prefix, overall["accuracy"])
    logger.info("%s val macro_accuracy: %.4f", prefix, overall["macro_accuracy"])
    logger.info(
        "%s counts: n=%d  pos=%d  neg=%d",
        prefix,
        int(overall["num_samples"]),
        int(overall["num_pos"]),
        int(overall["num_neg"]),
    )
    logger.info(
        "%s mean prob: pos=%.4f  neg=%.4f",
        prefix,
        overall["pos_prob_mean"],
        overall["neg_prob_mean"],
    )
    for task, stats in per_task.items():
        logger.info("    %s: auc=%.4f  acc=%.4f", task, stats["auc"], stats["accuracy"])
        logger.info(
            "      n=%d  pos=%d  neg=%d  pos_prob=%.4f  neg_prob=%.4f",
            int(stats["num_samples"]),
            int(stats["num_pos"]),
            int(stats["num_neg"]),
            stats["pos_prob_mean"],
            stats["neg_prob_mean"],
        )


def _resolve_encoder_name_from_model(model_name):
    lowered = model_name.strip().lower()
    if lowered in ENCODER_REGISTRY:
        return lowered

    for encoder_name, encoder_cls in ENCODER_REGISTRY.items():
        if lowered == encoder_cls.MODEL.lower():
            return encoder_name

    available_models = ["%s (%s)" % (name, cls.MODEL) for name, cls in sorted(ENCODER_REGISTRY.items())]
    raise ValueError(
        "Unsupported disease_model_name=%r. Pass an encoder key or exact model name. Available: %s"
        % (model_name, ", ".join(available_models))
    )


def build_task_text_embs(args, device, rank, is_ddp, expected_dim):
    tasks_sorted = sorted(TASK_2_DISEASE_NAME)
    task_text_embs = torch.zeros(len(tasks_sorted), expected_dim, dtype=torch.float32, device=device)

    if rank == 0:
        encoder_name = _resolve_encoder_name_from_model(args.disease_model_name)
        encoder = get_encoder(encoder_name, model_name=args.disease_model_name)
        model_source = args.disease_model_path or encoder.model_name
        tokenizer_source = args.disease_tokenizer_name or encoder.model_name
        logger.info(
            "Loading disease text encoder: model=%s tokenizer=%s via encoder=%s add_special_tokens=%s",
            model_source,
            tokenizer_source,
            encoder_name,
            encoder.ADD_SPECIAL_TOKENS,
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
        model = AutoModel.from_pretrained(model_source, local_files_only=True).to(device)
        model.eval()

        texts = [TASK_2_DISEASE_QUERY_TEXT[task] for task in tasks_sorted]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=encoder.ADD_SPECIAL_TOKENS,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            model_output = model(**batch)
            task_text_embs = encoder.postprocess_embeddings(
                encoder.get_embeddings(model_output, batch, tokenizer)
            ).float()

        if task_text_embs.shape[1] != expected_dim:
            raise ValueError(
                "Disease embedding dim %d from %s does not match event embedding dim %d from %s"
                % (task_text_embs.shape[1], model_source, expected_dim, args.bert_embeddings)
            )

        logger.info("  Built disease embeddings for %d task queries", len(tasks_sorted))
        for task, text in zip(tasks_sorted, texts):
            logger.info("  Query text [%s]: %s", task, text)

    if is_ddp:
        dist.broadcast(task_text_embs, src=0)
    return task_text_embs.cpu()


def parse_args():
    p = argparse.ArgumentParser(
        description="Disease soft token + event token sequence -> bidirectional transformer -> classify from disease token",
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
    p.add_argument("--disease_model_path", default=None)
    p.add_argument("--disease_tokenizer_name", default=None)

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
    p.add_argument("--output_dir", default="output/disease-soft-token-classifier")
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
    p.add_argument("--pad_to_num_events", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--aux_loss_weight", type=float, default=0.0)
    p.add_argument("--align_loss_weight", type=float, default=0.0)

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

    if not args.eval_only and args.train_data_dir is None:
        raise ValueError("--train_data_dir is required unless --eval_only is set.")
    if args.eval_data_paths is None:
        raise ValueError("--eval_data_paths is required")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run dir: %s", run_dir)

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
    task_text_embs = build_task_text_embs(args, device, rank, is_ddp, expected_dim=store.dim)

    if args.checkpoint:
        model = DiseaseEventSoftTokenClassifier.load_checkpoint(
            Path(args.checkpoint),
            task_text_embs=task_text_embs,
            device=device,
            dtype=dtype,
        )
    else:
        model = DiseaseEventSoftTokenClassifier(
            bert_dim=store.dim,
            task_text_embs=task_text_embs,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            head_layers=args.head_layers,
            max_positions=args.max_positions or (args.pad_to_num_events + 1),
            position_type=args.position_type,
            attention_type=args.attention_type,
            dropout=args.dropout,
            dtype=dtype,
        ).to(device)

    if rank == 0:
        logger.info(
            "Soft-token classifier: hidden=%d num_layers=%d num_heads=%d head_layers=%d intermediate=%s max_positions=%d position_type=%s attention_type=%s dropout=%.3f dtype=%s",
            args.hidden_size,
            args.num_layers,
            args.num_heads,
            args.head_layers,
            args.intermediate_size or (args.hidden_size * 4),
            args.max_positions or (args.pad_to_num_events + 1),
            args.position_type,
            args.attention_type,
            args.dropout,
            dtype,
        )
        logger.info("Training batches use sample-level shuffle; a batch may mix multiple tasks.")

    if is_ddp and not args.eval_only:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, static_graph=True)

    if args.eval_only:
        overall, per_task = evaluate_classifier(
            model,
            eval_data_paths=args.eval_data_paths,
            store=store,
            args=args,
            device=device,
        )
        if rank == 0:
            _log_eval("", overall, per_task)
        if is_ddp:
            dist.destroy_process_group()
        return

    first_task = sorted(args.tasks)[0]
    task_dir = Path(args.train_data_dir) / first_task
    data_epoch_files = sorted(task_dir.glob("train_prepared_*.parquet"))
    available_epochs = sorted(int(p.stem.split("_")[-1]) for p in data_epoch_files)
    if not available_epochs:
        raise ValueError(f"No train_prepared_*.parquet found in {task_dir}")
    if args.train_data_epochs is not None:
        selected_epochs = list(dict.fromkeys(args.train_data_epochs))
        missing = [ep for ep in selected_epochs if ep not in available_epochs]
        if missing:
            raise ValueError(f"Requested epochs {missing}, available={available_epochs}")
    else:
        selected_epochs = available_epochs[: args.epochs]

    if rank == 0:
        logger.info("Selected train data epoch(s): %s", selected_epochs)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )

    schedule_lengths = []
    for schedule_idx, data_epoch in enumerate(selected_epochs):
        ds = SampleLevelPreparedDataset(
            args.train_data_dir, data_epoch, args.tasks, store,
            args.batch_size, schedule_idx, args.seed, world_size, rank,
            args.pad_to_num_events,
        )
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
            logger.info("Pass %d/%d: prepared data epoch %d …", schedule_idx + 1, len(selected_epochs), data_epoch)

        epoch_ds = SampleLevelPreparedDataset(
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
            persistent_workers=False,
        )

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {schedule_idx + 1}/{len(selected_epochs)}", disable=(rank != 0), dynamic_ncols=True, total=len(epoch_ds))

        for batch_idx, batch in enumerate(pbar):
            labels = batch["labels"].to(device).float()
            logits, aux_logits, disease_hidden, event_pooled = model(
                batch["event_embs"].to(device),
                batch["event_mask"].to(device),
                batch["task_idxs"].to(device),
                return_aux_logits=True,
            )
            main_loss = loss_fn(logits, labels)
            aux_loss = loss_fn(aux_logits, labels)
            align_loss = F.mse_loss(
                F.normalize(event_pooled.float(), dim=-1),
                F.normalize(disease_hidden.float(), dim=-1),
            )
            loss = (main_loss + args.aux_loss_weight * aux_loss + args.align_loss_weight * align_loss) / args.grad_accum
            loss.backward()

            is_update_step = ((batch_idx + 1) % args.grad_accum == 0) or ((batch_idx + 1) == len(epoch_ds))
            if is_update_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
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
                pbar.set_postfix(
                    loss=f"{loss.item() * args.grad_accum:.4f}",
                    lmain=f"{main_loss.item():.4f}",
                    laux=f"{aux_loss.item():.4f}",
                    lalign=f"{align_loss.item():.4f}",
                    posp=f"{probs[labels == 1].mean().item():.3f}" if (labels == 1).any() else "nan",
                    negp=f"{probs[labels == 0].mean().item():.3f}" if (labels == 0).any() else "nan",
                    gnorm=f"{float(grad_norm):.3f}" if not isinstance(grad_norm, float) or not math.isnan(grad_norm) else "nan",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if rank == 0:
            logger.info("Epoch %d/%d avg_loss=%.4f", schedule_idx + 1, len(selected_epochs), epoch_loss / max(len(epoch_ds), 1))
        overall, per_task = evaluate_classifier(
            model,
            eval_data_paths=args.eval_data_paths,
            store=store,
            args=args,
            device=device,
        )
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
                log_dict = {
                    "epoch": schedule_idx + 1,
                    "train/avg_loss": epoch_loss / max(len(epoch_ds), 1),
                    "eval/auc": overall["auc"],
                    "eval/accuracy": overall["accuracy"],
                    "eval/pos_prob_mean": overall["pos_prob_mean"],
                    "eval/neg_prob_mean": overall["neg_prob_mean"],
                }
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
