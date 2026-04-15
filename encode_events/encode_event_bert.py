#!/usr/bin/env python3
"""
Encode unique events from a parquet index into Bio/BERT-style embeddings.

This mirrors the overall flow of `extract_bio_emb.py`, but instead of scanning
ehrshot.csv directly, it reads the pre-extracted unique-event parquet from
`encode_events/extract_event_parquet.py` and renders each row into text with a
Jinja2 template.

Outputs:
  <output_dir>/embeddings.npy
      Float32 array where row i matches event_id == i.
  <output_dir>/<template_name>.j2
      Copy of the exact Jinja template used for rendering.

Distributed behavior:
  - world_size > 1: save rank-local shards, then merge on rank 0
  - world_size == 1: write embeddings.npy directly, no shard merge step
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@torch.inference_mode()
def encode_slice(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    special_ids: set[int],
    rank: int,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    desc: str = "Encoding",
) -> tuple[np.ndarray, dict[str, float]]:
    """Encode a list of texts to a float32 numpy array of shape (N, H)."""
    model.eval()
    all_embs: list[np.ndarray] = []
    total_steps = 0
    total_model_time = 0.0

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=desc,
        disable=(rank != 0),
        dynamic_ncols=True,
    ):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if use_amp and device.type in {"cuda", "cpu"} and amp_dtype is not None
            else nullcontext()
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        with amp_ctx:
            out = model(**enc)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.time()

        embs = mean_pool_no_special(
            out.last_hidden_state,
            enc["input_ids"],
            enc["attention_mask"],
            special_ids,
        )
        all_embs.append(embs.float().cpu().numpy())
        total_steps += 1
        total_model_time += (t1 - t0)

    if all_embs:
        arr = np.concatenate(all_embs, axis=0)
    else:
        arr = np.empty((0, model.config.hidden_size), dtype=np.float32)

    stats = {
        "total_steps": float(total_steps),
        "total_model_time": float(total_model_time),
    }
    return arr, stats


def mean_pool_no_special(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: set[int],
) -> torch.Tensor:
    """Mean pool over non-padding, non-special tokens."""
    special = torch.zeros_like(attention_mask, dtype=torch.bool)
    for sid in special_token_ids:
        special |= (input_ids == sid)
    pool_mask = attention_mask.bool() & ~special
    pool_mask_f = pool_mask.float().unsqueeze(-1)
    sum_emb = (last_hidden_state * pool_mask_f).sum(dim=1)
    count = pool_mask_f.sum(dim=1).clamp(min=1e-9)
    return sum_emb / count


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--event_parquet",
        default="EHRSHOT_ASSETS/features/unique_event_rows_plus_cond_fast.parquet",
        help="Unique-event parquet from encode_events/extract_event_parquet.py",
    )
    p.add_argument(
        "--template_path",
        default="encode_events/event_to_text.j2",
        help="Jinja2 template used to render one event row into text.",
    )
    p.add_argument(
        "--model_name",
        default="michiyasunaga/BioLinkBERT-base",
        help="HuggingFace model name or local path.",
    )
    p.add_argument("--output_dir", default="encode_events_result")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile for the model.")
    p.add_argument("--amp", action="store_true", help="Enable autocast during model forward.")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def load_template(template_path: str):
    template_file = Path(template_path)
    env = Environment(
        loader=FileSystemLoader(str(template_file.parent)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    return env.get_template(template_file.name)


def normalise_optional(x: object) -> str:
    if isinstance(x, str):
        return x.strip()
    if pd.isna(x):
        return ""
    return str(x).strip()


def get_num_events(event_parquet: str) -> int:
    pf = pq.ParquetFile(event_parquet)
    return pf.metadata.num_rows


def get_rank_bounds(n_total: int, rank: int, world_size: int) -> tuple[int, int]:
    start = (n_total * rank) // world_size
    end = (n_total * (rank + 1)) // world_size
    return start, end


def read_event_slice(event_parquet: str, start: int, end: int) -> pd.DataFrame:
    columns = ["event_id", "omop_table", "event_type", "code", "description", "value", "unit"]
    if end <= start:
        return pd.DataFrame(columns=columns)

    pf = pq.ParquetFile(event_parquet)
    chunks = []
    row_cursor = 0
    for rg_idx in range(pf.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        rg_start = row_cursor
        rg_end = row_cursor + rg_rows
        row_cursor = rg_end
        if rg_end <= start or rg_start >= end:
            continue

        table = pf.read_row_group(rg_idx, columns=columns)
        slice_start = max(start - rg_start, 0)
        slice_len = min(end, rg_end) - max(start, rg_start)
        chunks.append(table.slice(slice_start, slice_len).to_pandas())

    if not chunks:
        return pd.DataFrame(columns=columns)
    event_df = pd.concat(chunks, ignore_index=True)
    return event_df


def load_and_render_events(event_parquet: str, template_path: str, start: int, end: int) -> pd.DataFrame:
    logger.info("[Phase 1] Loading unique-event parquet rows [%s, %s) ...", start, end)
    event_df = read_event_slice(event_parquet, start, end)
    template = load_template(template_path)
    logger.info("[Phase 1] Rendering event text with template %s ...", template_path)

    records = event_df.to_dict(orient="records")
    texts: list[str] = []
    empty_count = 0
    for row in tqdm(records, desc="Rendering", dynamic_ncols=True):
        ctx = {k: normalise_optional(v) for k, v in row.items()}
        text = template.render(**ctx).strip()
        if not text:
            empty_count += 1
        texts.append(text)

    if empty_count:
        raise ValueError(f"Template rendered {empty_count} empty event texts.")

    expected_ids = np.arange(start, end)
    actual_ids = event_df["event_id"].to_numpy()
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError(
            f"Input parquet slice has unexpected event_id values for range [{start}, {end})."
        )

    event_df = event_df.copy()
    event_df["event_text"] = texts
    return event_df


def merge_shards(shard_dir: Path, n_total: int, hidden_size: int, output_dir: Path):
    out_path = output_dir / "embeddings.npy"
    if out_path.exists():
        logger.info("[Phase 3] embeddings.npy already exists — skipping merge.")
        return

    logger.info("[Phase 3] Merging shards ...")
    meta_files = sorted(shard_dir.glob("shard_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No shard metadata files found in {shard_dir}")

    embeddings = np.zeros((n_total, hidden_size), dtype=np.float32)
    for mf in meta_files:
        with open(mf) as f:
            meta = json.load(f)
        shard_npy = shard_dir / mf.name.replace(".json", ".npy")
        emb = np.load(shard_npy)
        start = int(meta["start"])
        end = int(meta["end"])
        if emb.shape[0] != (end - start):
            raise ValueError(
                f"Shard {shard_npy} has {emb.shape[0]} rows, expected {end - start} for [{start}, {end})."
            )
        embeddings[start:end] = emb
        logger.info("  Loaded %s: rows [%s, %s)", shard_npy.name, f"{start:,}", f"{end:,}")

    np.save(out_path, embeddings)
    logger.info("[Phase 3] Saved embeddings.npy → %s  shape=%s", out_path, embeddings.shape)


def resolve_amp_dtype(args, device: torch.device) -> torch.dtype | None:
    if not args.amp:
        return None
    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one of --fp16 or --bf16.")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return None


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if is_dist:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_dist:
            shard_dir.mkdir(parents=True, exist_ok=True)
        template_src = Path(args.template_path)
        shutil.copy2(template_src, output_dir / template_src.name)

    if is_dist:
        dist.barrier()

    n_total = get_num_events(args.event_parquet)
    my_start, my_end = get_rank_bounds(n_total, rank, world_size)
    event_df = load_and_render_events(args.event_parquet, args.template_path, my_start, my_end)

    texts = event_df["event_text"].tolist()
    if rank == 0:
        logger.info("Total unique event texts to encode: %s", f"{n_total:,}")

    logger.info("[Rank %d] Loading %s ...", rank, args.model_name)
    model_kwargs: dict[str, object] = {}
    if args.fp16:
        model_kwargs["dtype"] = torch.float16
    elif args.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    if args.local_files_only:
        model_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        **({} if not args.local_files_only else {"local_files_only": True}),
    )
    model = AutoModel.from_pretrained(args.model_name, **model_kwargs).to(device)
    model.eval()
    if args.compile:
        model = torch.compile(model, dynamic=False)
    special_ids = set(tokenizer.all_special_ids)
    amp_dtype = resolve_amp_dtype(args, device)

    logger.info(
        "[Rank %d] Encoding %s texts (of %s total, world_size=%d, amp=%s, amp_dtype=%s, compile=%s) ...",
        rank,
        f"{len(texts):,}",
        f"{n_total:,}",
        world_size,
        "on" if args.amp else "off",
        str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "n/a",
        "on" if args.compile else "off",
    )
    embs, local_stats = encode_slice(
        texts,
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        special_ids=special_ids,
        rank=rank,
        use_amp=args.amp,
        amp_dtype=amp_dtype,
        desc=f"[Rank {rank}]",
    )

    local_model_time = float(local_stats["total_model_time"])
    local_steps = int(local_stats["total_steps"])
    local_ms_step = (1000.0 * local_model_time / local_steps) if local_steps > 0 else 0.0
    logger.info(
        "[Rank %d] Timing: %.1f ms/step across %d steps",
        rank,
        local_ms_step,
        local_steps,
    )

    if is_dist:
        shard_npy = shard_dir / f"shard_{rank:04d}.npy"
        shard_json = shard_dir / f"shard_{rank:04d}.json"
        np.save(shard_npy, embs)
        with open(shard_json, "w") as f:
            json.dump({"rank": rank, "start": my_start, "end": my_end}, f)
        logger.info("[Rank %d] Saved shard → %s  shape=%s", rank, shard_npy, embs.shape)
        dist.barrier()

        time_tensor = torch.tensor([local_model_time], device=device, dtype=torch.float64)
        steps_tensor = torch.tensor([float(local_steps)], device=device, dtype=torch.float64)
        dist.reduce(time_tensor, dst=0, op=dist.ReduceOp.MAX)
        dist.reduce(steps_tensor, dst=0, op=dist.ReduceOp.MAX)

        if rank == 0:
            first_shard = sorted(shard_dir.glob("shard_*.npy"))[0]
            hidden_size = np.load(first_shard, mmap_mode="r").shape[1]
            merge_shards(shard_dir, n_total, hidden_size, output_dir)
            total_model_time = float(time_tensor[0].item())
            total_steps = int(steps_tensor[0].item())
            agg_ms_step = (1000.0 * total_model_time / total_steps) if total_steps > 0 else 0.0
            logger.info("Done.")
            logger.info("Outputs:")
            logger.info("  %s/embeddings.npy       — shape (%s, %s), float32", output_dir, n_total, hidden_size)
            logger.info("  %s/%s            — copied Jinja template", output_dir, Path(args.template_path).name)
            logger.info("Aggregate timing: %.1f ms/step", agg_ms_step)
        dist.destroy_process_group()
        return

    out_path = output_dir / "embeddings.npy"
    np.save(out_path, embs)
    logger.info("Done.")
    logger.info("Outputs:")
    logger.info("  %s/embeddings.npy       — shape %s, float32", out_path, embs.shape)
    logger.info("  %s/%s            — copied Jinja template", output_dir, Path(args.template_path).name)
    logger.info("Aggregate timing: %.1f ms/step", local_ms_step)


if __name__ == "__main__":
    main()
