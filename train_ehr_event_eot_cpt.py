#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from jinja2 import Environment, StrictUndefined
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset, DistributedSampler, get_worker_info
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_warning()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


OMOP_TABLE_PREFIX = {
    "condition_occurrence": "Condition",
    "procedure_occurrence": "Procedure",
    "drug_exposure": "Drug",
    "measurement": "Measurement",
    "observation": "Observation",
    "visit_occurrence": "Visit",
    "device_exposure": "Device",
    "death": "Death",
    "note": "Note",
    "person": "Demographics",
}


def normalize_optional_str(x) -> str | None:
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    if pd.isna(x):
        return None
    return str(x).strip() or None


def unique_event_key(omop_table: object, code: object, value: object, unit: object) -> tuple[str, str, str, str]:
    return (
        normalize_optional_str(omop_table) or "",
        normalize_optional_str(code) or "",
        normalize_optional_str(value) or "",
        normalize_optional_str(unit) or "",
    )


def load_event_template(template_path: str):
    template_text = Path(template_path).read_text()
    env = Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return env.from_string(template_text)


def format_event_row(ev: pd.Series, include_condition_occurrence: bool, event_template) -> str | None:
    if (not include_condition_occurrence) and ev["omop_table"] == "condition_occurrence":
        return None

    code = str(ev["code"]).strip()
    event_type = normalize_optional_str(ev.get("event_type")) or OMOP_TABLE_PREFIX.get(ev["omop_table"], ev["omop_table"])
    description = normalize_optional_str(ev.get("description")) or ""
    value = normalize_optional_str(ev["value"])
    unit = normalize_optional_str(ev["unit"])
    rendered = event_template.render(
        event_type=event_type or "",
        description=description,
        code=code,
        value=value or "",
        unit=unit or "",
    ).strip()
    return rendered or None


def load_code_description_map(concept_csv: str) -> dict[str, str]:
    logger.info("Loading concept map from %s", concept_csv)
    concept_df = pd.read_csv(
        concept_csv,
        usecols=["concept_name", "vocabulary_id", "concept_code"],
        low_memory=False,
        dtype=str,
    ).fillna("")
    concept_df["code"] = concept_df["vocabulary_id"] + "/" + concept_df["concept_code"]
    filtered = concept_df[concept_df["code"] != concept_df["concept_name"]]
    code2desc = dict(zip(filtered["code"], filtered["concept_name"]))
    logger.info("Loaded %s code->description mappings", f"{len(code2desc):,}")
    return code2desc


def build_event_token_cache(
    df_ehr: pd.DataFrame,
    tokenizer,
    event_template,
    include_condition_occurrence: bool,
    append_eos_token_id: int,
    tokenize_batch_size: int,
) -> Dict[tuple[str, str, str, str], List[int]]:
    key_df = df_ehr[["omop_table", "event_type", "code", "description", "value", "unit"]].drop_duplicates(
        subset=["omop_table", "code", "value", "unit"],
        keep="first",
    ).reset_index(drop=True)
    if not include_condition_occurrence:
        key_df = key_df[key_df["omop_table"] != "condition_occurrence"].copy()
    logger.info("Unique events to tokenize in-memory: %s", f"{len(key_df):,}")

    unique_texts: List[str] = []
    unique_keys: List[tuple[str, str, str, str]] = []
    dropped = 0
    for _, row in tqdm(key_df.iterrows(), total=len(key_df), desc="render unique events", dynamic_ncols=True):
        text = format_event_row(row, include_condition_occurrence, event_template)
        if not text:
            dropped += 1
            continue
        unique_keys.append(unique_event_key(row["omop_table"], row["code"], row["value"], row["unit"]))
        unique_texts.append(text)
    logger.info("Renderable unique events: %s (dropped=%s)", f"{len(unique_keys):,}", f"{dropped:,}")

    event_token_map: Dict[tuple[str, str, str, str], List[int]] = {}
    for i in tqdm(range(0, len(unique_texts), tokenize_batch_size), desc="tokenize unique events", dynamic_ncols=True):
        batch_texts = unique_texts[i : i + tokenize_batch_size]
        batch_keys = unique_keys[i : i + tokenize_batch_size]
        enc = tokenizer(batch_texts, add_special_tokens=False, return_attention_mask=False)
        for key, token_ids in zip(batch_keys, enc["input_ids"]):
            ids = [int(x) for x in token_ids]
            ids.append(int(append_eos_token_id))
            event_token_map[key] = ids
    logger.info("Tokenized unique event cache size: %s", f"{len(event_token_map):,}")
    return event_token_map


class EHRNextTokenIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        patient_groups: Dict[int, pd.DataFrame],
        patient_ids: List[int],
        event_token_map: Dict[tuple[str, str, str, str], List[int]],
        include_condition_occurrence: bool,
        seq_len: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.patient_groups = patient_groups
        self.patient_ids = patient_ids
        self.event_token_map = event_token_map
        self.include_condition_occurrence = include_condition_occurrence
        self.seq_len = seq_len
        self.pad_token_id = int(pad_token_id)
        self.num_samples = self._estimate_num_samples()

    def _estimate_num_samples(self) -> int:
        n = 0
        for patient_id in self.patient_ids:
            pdf = self.patient_groups[int(patient_id)]
            total_tokens = 0
            for _, ev in pdf.iterrows():
                if (not self.include_condition_occurrence) and ev["omop_table"] == "condition_occurrence":
                    continue
                key = unique_event_key(ev["omop_table"], ev["code"], ev["value"], ev["unit"])
                token_ids = self.event_token_map.get(key)
                if token_ids is not None:
                    total_tokens += len(token_ids)
            if total_tokens > 0:
                n += math.ceil(total_tokens / self.seq_len)
        return n

    def __len__(self) -> int:
        return self.num_samples

    def _iter_patient_ids(self) -> Iterable[int]:
        worker = get_worker_info()
        if worker is None:
            yield from self.patient_ids
            return
        for idx, patient_id in enumerate(self.patient_ids):
            if idx % worker.num_workers == worker.id:
                yield patient_id

    def __iter__(self):
        for patient_id in self._iter_patient_ids():
            pdf = self.patient_groups[int(patient_id)]
            token_stream: List[int] = []
            event_stream: List[int] = []
            event_idx = 0
            for _, ev in pdf.iterrows():
                if (not self.include_condition_occurrence) and ev["omop_table"] == "condition_occurrence":
                    continue
                key = unique_event_key(ev["omop_table"], ev["code"], ev["value"], ev["unit"])
                token_ids = self.event_token_map.get(key)
                if token_ids is None:
                    continue
                token_stream.extend(token_ids)
                event_stream.extend([event_idx] * len(token_ids))
                event_idx += 1

            if not token_stream:
                continue

            for start in range(0, len(token_stream), self.seq_len):
                chunk_ids = token_stream[start:start + self.seq_len]
                chunk_events = event_stream[start:start + self.seq_len]
                n = len(chunk_ids)

                input_ids = torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((self.seq_len,), dtype=torch.long)
                event_ids = torch.full((self.seq_len,), -1, dtype=torch.long)
                labels = torch.full((self.seq_len,), -100, dtype=torch.long)

                input_ids[:n] = torch.tensor(chunk_ids, dtype=torch.long)
                attention_mask[:n] = 1
                event_ids[:n] = torch.tensor(chunk_events, dtype=torch.long)
                labels[:n] = torch.tensor(chunk_ids, dtype=torch.long)

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "event_ids": event_ids,
                    "labels": labels,
                }


class EHRNextTokenParquetDataset(Dataset):
    def __init__(self, parquet_path: str, max_samples: int | None = None):
        logger.info("Loading training parquet: %s", parquet_path)
        table = pq.read_table(
            parquet_path,
            columns=[
                "input_ids",
                "attention_mask",
                "event_ids",
                "labels",
            ],
            memory_map=True,
        )
        n = table.num_rows
        if max_samples is not None:
            n = min(n, max_samples)
            table = table.slice(0, n)
        self.input_ids = table["input_ids"]
        self.attention_mask = table["attention_mask"]
        self.event_ids = table["event_ids"]
        self.labels = table["labels"]
        self.num_samples = n
        logger.info(
            "Loaded %s offline training samples from parquet with Arrow-backed columns",
            f"{self.num_samples:,}",
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx].as_py(), dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx].as_py(), dtype=torch.long),
            "event_ids": torch.tensor(self.event_ids[idx].as_py(), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx].as_py(), dtype=torch.long),
        }


class EventEOTSummaryCPTModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        torch_dtype: torch.dtype,
        attn_implementation: str,
        local_files_only: bool,
    ):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        self.model = self.backbone.model
        self.vocab_size = int(self.backbone.config.vocab_size)

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        logger.info("Saved checkpoint -> %s", save_dir)

    def build_event_summary_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_ids: torch.Tensor,
        eos_token_id: int,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        valid = attention_mask.bool()
        pos = torch.arange(seq_len, device=device)
        causal = pos.view(1, 1, seq_len) <= pos.view(1, seq_len, 1)
        same_event = event_ids[:, :, None] == event_ids[:, None, :]
        eos_keys = ((input_ids == eos_token_id) & valid)[:, None, :]
        q_valid = valid[:, :, None]
        k_valid = valid[:, None, :]
        allowed = ((same_event & causal) | (eos_keys & causal)) & q_valid & k_valid

        eye = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
        allowed = allowed | ((~valid)[:, :, None] & eye)

        mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=self.model.embed_tokens.weight.dtype, device=device)
        mask = mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(mask.dtype).min)
        return mask

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_ids: torch.Tensor,
        labels: torch.Tensor,
        eos_token_id: int,
    ) -> dict[str, torch.Tensor]:
        attn_mask = self.build_event_summary_attention_mask(input_ids, attention_mask, event_ids, eos_token_id)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def parse_args():
    p = argparse.ArgumentParser(
        description="Continue-pretraining over EHR event text with event-EOT summary attention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data_dir", default="data/EHRSHOT_ASSETS")
    p.add_argument("--train_parquet", default=None, help="Offline parquet built by build_ehr_event_eot_cpt_parquet.py")
    p.add_argument("--ehrshot_csv", default=None)
    p.add_argument("--concept_csv", default=None)
    p.add_argument("--template_path", default="01_gen_meta/templates/biolinkbert_event.j2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--include_condition_occurrence", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--tokenize_batch_size", type=int, default=4096)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = maybe_init_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    ehrshot_csv = Path(args.ehrshot_csv) if args.ehrshot_csv else data_dir / "data" / "ehrshot.csv"
    concept_csv = Path(args.concept_csv) if args.concept_csv else data_dir / "femr" / "logs" / "omop_dir" / "concept.csv"

    logger.info("Rank %d/%d | device=%s", rank, world_size, device)
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(f"Tokenizer {args.model_name} has no pad_token_id; needed as end-of-event token.")
    logger.info("Using end-of-event token: %r (id=%d)", tokenizer.pad_token, pad_token_id)

    if args.train_parquet:
        dataset = EHRNextTokenParquetDataset(args.train_parquet, max_samples=args.max_samples)
    else:
        if world_size > 1:
            logger.warning("Online raw-EHR path with IterableDataset is not DDP-safe; prefer --train_parquet for multi-GPU runs.")
        logger.info("Loading and sorting raw EHR CSV: %s", ehrshot_csv)
        df_ehr = pd.read_csv(
            ehrshot_csv,
            usecols=["patient_id", "start", "omop_table", "code", "value", "unit"],
            low_memory=False,
            dtype={"value": str, "unit": str},
        )
        if df_ehr.columns[0] == "" or str(df_ehr.columns[0]).startswith("Unnamed"):
            df_ehr = df_ehr.drop(columns=[df_ehr.columns[0]])
        df_ehr["start"] = pd.to_datetime(df_ehr["start"])
        df_ehr = df_ehr.sort_values(["patient_id", "start"], ascending=[True, True]).reset_index(drop=True)
        logger.info("Loaded %s events across %s patients", f"{len(df_ehr):,}", f"{df_ehr['patient_id'].nunique():,}")

        if args.max_patients is not None:
            keep_ids = df_ehr["patient_id"].drop_duplicates().tolist()[: args.max_patients]
            df_ehr = df_ehr[df_ehr["patient_id"].isin(keep_ids)].copy()
            logger.info("Restricted to first %s patients", f"{len(keep_ids):,}")

        code2desc = load_code_description_map(str(concept_csv))
        for col in ["omop_table", "code", "value", "unit"]:
            df_ehr[col] = df_ehr[col].fillna("").astype(str).str.strip()
        df_ehr["description"] = df_ehr["code"].map(lambda code: normalize_optional_str(code2desc.get(code, "")) or "")
        df_ehr["event_type"] = df_ehr["omop_table"].map(lambda x: OMOP_TABLE_PREFIX.get(x, x))

        event_template = load_event_template(args.template_path)
        event_token_map = build_event_token_cache(
            df_ehr=df_ehr,
            tokenizer=tokenizer,
            event_template=event_template,
            include_condition_occurrence=args.include_condition_occurrence,
            append_eos_token_id=pad_token_id,
            tokenize_batch_size=args.tokenize_batch_size,
        )

        patient_ids = df_ehr["patient_id"].drop_duplicates().tolist()
        patient_groups = {int(pid): pdf for pid, pdf in df_ehr.groupby("patient_id", sort=False)}
        dataset = EHRNextTokenIterableDataset(
            patient_groups=patient_groups,
            patient_ids=patient_ids,
            event_token_map=event_token_map,
            include_condition_occurrence=args.include_condition_occurrence,
            seq_len=args.seq_len,
            pad_token_id=pad_token_id,
        )
    logger.info("Estimated training samples: %s", f"{len(dataset):,}")

    sampler = None
    if world_size > 1 and not isinstance(dataset, IterableDataset):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and not isinstance(dataset, IterableDataset)),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )

    torch_dtype = torch.float32
    autocast_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
        autocast_dtype = torch.float16

    logger.info("Loading model: %s", args.model_name)
    model = EventEOTSummaryCPTModel(
        model_name=args.model_name,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    ).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model params: %.1fM", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(dataset) / max(args.batch_size, 1) / max(args.grad_accum, 1)))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    wandb_run = None
    if args.wandb_project is not None and is_main_process():
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"ehr-event-eot-cpt-{datetime.now().strftime('%m%d-%H%M')}",
            tags=args.wandb_tags,
            config=vars(args),
        )

    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True, disable=not is_main_process())
        optimizer.zero_grad()
        run_loss = 0.0
        run_batches = 0
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            autocast_enabled = torch.cuda.is_available() and (autocast_dtype is not None)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    event_ids=batch["event_ids"],
                    labels=batch["labels"],
                    eos_token_id=pad_token_id,
                )
                loss = outputs["loss"] / args.grad_accum
            loss.backward()

            loss_value = float(outputs["loss"].item())
            run_loss += loss_value
            run_batches += 1
            epoch_loss_sum += loss_value
            epoch_batches += 1

            if ((batch_idx + 1) % args.grad_accum == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0 and is_main_process():
                    avg_loss = run_loss / max(run_batches, 1)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )
                    run_loss = 0.0
                    run_batches = 0

        if run_batches > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        epoch_avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        if dist.is_initialized():
            loss_tensor = torch.tensor([epoch_loss_sum, epoch_batches], dtype=torch.float64, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_avg_loss = float(loss_tensor[0].item() / max(loss_tensor[1].item(), 1.0))
        if is_main_process():
            logger.info("Epoch %d/%d summary: avg_loss=%.4f", epoch + 1, args.epochs, epoch_avg_loss)
        if args.save_every_epoch and is_main_process():
            unwrap_model(model).save_checkpoint(output_dir / f"epoch_{epoch + 1}")
        if epoch_avg_loss < best_loss and is_main_process():
            best_loss = epoch_avg_loss
            unwrap_model(model).save_checkpoint(output_dir / "best")

    if is_main_process():
        unwrap_model(model).save_checkpoint(output_dir / "final")
    if wandb_run is not None:
        wandb_run.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
