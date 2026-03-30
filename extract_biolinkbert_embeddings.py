#!/usr/bin/env python3
"""
extract_biolinkbert_embeddings.py

Offline extraction of per-event BioLinkBERT embeddings from ehrshot.csv.
Each unique event text (derived from code + description + value + unit) is
embedded exactly once, enabling downstream training to look up pre-computed
embeddings rather than re-tokenising full patient histories.

Pooling: mean pooling over non-padding, non-special tokens (no [CLS]/[SEP]).

─── Pipeline ───────────────────────────────────────────────────────────────
Phase 1  (rank 0): scan ehrshot.csv → collect unique events →
             save  <output_dir>/event_index.parquet
Phase 2  (all ranks): each rank encodes its own slice → saves
             <output_dir>/shards/shard_{rank:04d}.npy
Phase 3  (rank 0): merge shards → <output_dir>/embeddings.npy
             (row i  ↔  event_index.parquet row i)

─── Usage ──────────────────────────────────────────────────────────────────
Single-node, 4 GPUs:
    torchrun --nproc_per_node=4 extract_biolinkbert_embeddings.py

Multi-node (2 × 4 GPUs):
    torchrun --nnodes=2 --nproc_per_node=4 \\
        --rdzv_id=job1 --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
        extract_biolinkbert_embeddings.py [args]

Single GPU / CPU (debug):
    python extract_biolinkbert_embeddings.py --batch_size 32
"""

import os
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Event key / text formatting ───────────────────────────────────────────────

def normalise_event_key(code, value, unit) -> tuple[str, str, str]:
    """Canonical (code, value, unit) key used to index and look up embeddings.

    Called both during extraction (building the index) and during training
    (looking up pre-computed embeddings), so any change here applies everywhere.
    """
    return (
        str(code  if code  is not None else "").strip(),
        str(value if value is not None else "").strip(),
        str(unit  if unit  is not None else "").strip(),
    )


def format_event_text(code: str, description: str, value: str, unit: str) -> str | None:
    """Format a single EHR event row into a text string.

    Returns None if there is nothing meaningful to encode.
    """
    desc  = (description or "").strip()
    code  = (code        or "").strip()
    value = (value       or "").strip()
    unit  = (unit        or "").strip()

    if desc and code:
        text = f"{desc} [{code}]"
    elif code:
        text = f"[{code}]"
    elif desc:
        text = desc
    else:
        return None

    if value:
        text += f" | value={value}"
    if unit:
        text += f" | unit={unit}"
    return text


# ── Mean pooling (exclude [CLS], [SEP], and padding) ──────────────────────────

def mean_pool_no_special(
    last_hidden_state: torch.Tensor,   # (B, L, H)
    input_ids:         torch.Tensor,   # (B, L)
    attention_mask:    torch.Tensor,   # (B, L)
    special_token_ids: set[int],
) -> torch.Tensor:                     # (B, H)
    """Mean pool over real tokens, excluding padding and special tokens."""
    # Build mask: True where token is real AND not a special token
    special = torch.zeros_like(attention_mask, dtype=torch.bool)
    for sid in special_token_ids:
        special |= (input_ids == sid)
    pool_mask = attention_mask.bool() & ~special          # (B, L)
    pool_mask_f = pool_mask.float().unsqueeze(-1)          # (B, L, 1)
    sum_emb = (last_hidden_state * pool_mask_f).sum(dim=1) # (B, H)
    count   = pool_mask_f.sum(dim=1).clamp(min=1e-9)       # (B, 1)
    return sum_emb / count                                  # (B, H)


# ── Phase 1: collect unique event texts ───────────────────────────────────────

def collect_unique_events(
    ehrshot_csv: str,
    concept_csv: str,
    output_dir: Path,
    chunksize:  int = 500_000,
) -> pd.DataFrame:
    """Scan ehrshot.csv, build unique (code, value, unit) → event_text mapping.

    Saves <output_dir>/event_index.parquet and returns the DataFrame.
    Columns: event_id (int), event_text (str), code, value, unit.
    """
    index_path = output_dir / "event_index.parquet"
    if index_path.exists():
        logger.info(f"[Phase 1] Found existing event_index.parquet — skipping scan.")
        return pd.read_parquet(index_path)

    # Build code → description lookup from concept.csv
    logger.info("[Phase 1] Loading concept.csv ...")
    concept_df = pd.read_csv(
        concept_csv,
        usecols=["concept_name", "vocabulary_id", "concept_code"],
        low_memory=False,
        dtype=str,
    ).fillna("")
    # Construct the OMOP-style key used in ehrshot.csv: "vocab_id/concept_code"
    concept_df["code"] = concept_df["vocabulary_id"] + "/" + concept_df["concept_code"]
    filtered = concept_df[concept_df["code"] != concept_df["concept_name"]]
    code2desc: dict[str, str] = dict(
        zip(filtered["code"], filtered["concept_name"])
    )
    logger.info(f"[Phase 1] Loaded {len(code2desc):,} code→description mappings.")

    # Stream ehrshot.csv, collect unique (code, value, unit) tuples
    logger.info("[Phase 1] Scanning ehrshot.csv for unique events ...")
    seen:  set[tuple]    = set()
    rows:  list[dict]    = []
    total_lines          = 0
    no_desc_count        = 0

    for chunk in pd.read_csv(
        ehrshot_csv,
        usecols=["code", "value", "unit"],
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    ):
        total_lines += len(chunk)
        for code, value, unit in zip(chunk["code"], chunk["value"], chunk["unit"]):
            key = normalise_event_key(code, value, unit)
            if key in seen:
                continue
            seen.add(key)
            norm_code, norm_value, norm_unit = key
            desc = code2desc.get(norm_code, "")
            if not desc:
                no_desc_count += 1
            text = format_event_text(norm_code, desc, norm_value, norm_unit)
            if text is None:
                continue
            rows.append({"code": norm_code, "value": norm_value, "unit": norm_unit, "event_text": text})

    n_unique = len(seen)
    logger.info(f"[Phase 1] Scanned {total_lines:,} rows → {n_unique:,} unique events → {len(rows):,} encodable.")
    logger.info(f"[Phase 1] No-description events: {no_desc_count:,} / {n_unique:,} ({100*no_desc_count/max(n_unique,1):.1f}%)")

    event_df = pd.DataFrame(rows).reset_index(drop=True)
    event_df.index.name = "event_id"
    event_df = event_df.reset_index()   # event_id becomes a column

    event_df.to_parquet(index_path, index=False)
    logger.info(f"[Phase 1] Saved event_index.parquet → {index_path}")
    return event_df


# ── Phase 2: encode one rank's slice ──────────────────────────────────────────

@torch.inference_mode()
def encode_slice(
    texts:      list[str],
    model:      AutoModel,
    tokenizer:  AutoTokenizer,
    device:     torch.device,
    batch_size: int,
    max_length: int,
    special_ids: set[int],
    rank:       int,
    desc:       str = "Encoding",
) -> np.ndarray:
    """Encode a list of texts → numpy array (N, hidden_size), float32."""
    model.eval()
    all_embs: list[np.ndarray] = []

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
            truncation=False,
            return_tensors="pt",
        ).to(device)

        assert enc.input_ids[0].size(0) < max_length, f"Shouldn't exist sequence exceeding {max_length}"

        out  = model(**enc)
        embs = mean_pool_no_special(
            out.last_hidden_state,
            enc["input_ids"],
            enc["attention_mask"],
            special_ids,
        )
        embs = embs.float()
        all_embs.append(embs.cpu().numpy())

    return np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, model.config.hidden_size), dtype=np.float32)


# ── Phase 3: merge shards ─────────────────────────────────────────────────────

def merge_shards(shard_dir: Path, n_total: int, hidden_size: int, output_dir: Path):
    """Concatenate per-rank shard files into a single embeddings.npy."""
    out_path = output_dir / "embeddings.npy"
    if out_path.exists():
        logger.info(f"[Phase 3] embeddings.npy already exists — skipping merge.")
        return

    logger.info("[Phase 3] Merging shards ...")
    # Read shard metadata to know which indices each shard covers
    meta_files = sorted(shard_dir.glob("shard_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No shard metadata files found in {shard_dir}")

    embeddings = np.zeros((n_total, hidden_size), dtype=np.float32)
    for mf in meta_files:
        with open(mf) as f:
            meta = json.load(f)
        shard_npy = shard_dir / mf.name.replace(".json", ".npy")
        emb = np.load(shard_npy)
        for local_i, global_i in enumerate(meta["indices"]):
            embeddings[global_i] = emb[local_i]
        logger.info(f"  Loaded {shard_npy.name}: {len(meta['indices'])} vectors")

    np.save(out_path, embeddings)
    logger.info(f"[Phase 3] Saved embeddings.npy → {out_path}  shape={embeddings.shape}")


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ehrshot_csv",  default="EHRSHOT_ASSETS/data/ehrshot.csv")
    p.add_argument("--concept_csv",  default="EHRSHOT_ASSETS/femr/logs/omop_dir/concept.csv")
    p.add_argument("--model_name",   default="michiyasunaga/BioLinkBERT-base",
                   help="HuggingFace model name or local path.")
    p.add_argument("--output_dir",   default="data/biolinkbert_embeddings")
    p.add_argument("--batch_size",   type=int, default=256,
                   help="Tokenisation + forward-pass batch size per rank.")
    p.add_argument("--max_length",   type=int, default=256,
                   help="Max token length. BioLinkBERT-base supports up to 512.")
    p.add_argument("--fp16",         action="store_true",
                   help="Run model in float16 (saves GPU memory, minor precision loss).")
    p.add_argument("--bf16",         action="store_true",
                   help="Run model in bfloat16.")
    p.add_argument("--local_files_only", action="store_true",
                   help="Load model from local cache only (no HuggingFace download).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Distributed setup ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist    = world_size > 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if is_dist:
        dist.init_process_group(backend="nccl", device_id=device)

    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    shard_dir  = output_dir / "shards"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)

    if is_dist:
        dist.barrier()   # ensure dirs exist before all ranks proceed

    # ── Phase 1: collect unique events (rank 0 only) ───────────────────────────
    if rank == 0:
        event_df = collect_unique_events(
            args.ehrshot_csv, args.concept_csv, output_dir
        )
    if is_dist:
        dist.barrier()   # all ranks wait for Phase 1 to finish

    if rank != 0:
        event_df = pd.read_parquet(output_dir / "event_index.parquet")

    texts     = event_df["event_text"].tolist()
    n_total   = len(texts)
    if rank == 0:
        logger.info(f"Total unique event texts to encode: {n_total:,}")

    # ── Phase 2: distributed encoding ─────────────────────────────────────────
    # Each rank handles round-robin slice: rank, rank+world_size, rank+2*world_size, ...
    my_indices = list(range(rank, n_total, world_size))
    my_texts   = [texts[i] for i in my_indices]

    shard_npy  = shard_dir / f"shard_{rank:04d}.npy"
    shard_json = shard_dir / f"shard_{rank:04d}.json"

    if shard_npy.exists() and shard_json.exists():
        logger.info(f"[Rank {rank}] Shard already exists — skipping encoding.")
    else:
        logger.info(f"[Rank {rank}] Loading {args.model_name} ...")
        model_kwargs = {}
        if args.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        elif args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        if args.local_files_only:
            model_kwargs["local_files_only"] = True

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, **({} if not args.local_files_only else {"local_files_only": True})
        )
        model = AutoModel.from_pretrained(args.model_name, **model_kwargs).to(device)
        model.eval()

        # Identify special token IDs ([CLS]=101, [SEP]=102 for BERT-family)
        special_ids: set[int] = set(tokenizer.all_special_ids)

        logger.info(
            f"[Rank {rank}] Encoding {len(my_texts):,} texts "
            f"(of {n_total:,} total, world_size={world_size}) ..."
        )
        embs = encode_slice(
            my_texts, model, tokenizer, device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            special_ids=special_ids,
            rank=rank,
            desc=f"[Rank {rank}]",
        )

        np.save(shard_npy, embs)
        with open(shard_json, "w") as f:
            json.dump({"rank": rank, "indices": my_indices}, f)
        logger.info(
            f"[Rank {rank}] Saved shard → {shard_npy}  "
            f"shape={embs.shape}  dtype={embs.dtype}"
        )

    if is_dist:
        dist.barrier()   # wait for all ranks to finish encoding

    # ── Phase 3: merge shards (rank 0 only) ───────────────────────────────────
    if rank == 0:
        # Infer hidden_size from first shard
        first_shard = sorted(shard_dir.glob("shard_*.npy"))[0]
        hidden_size = np.load(first_shard, mmap_mode="r").shape[1]
        merge_shards(shard_dir, n_total, hidden_size, output_dir)
        logger.info("Done.")
        logger.info(f"Outputs:")
        logger.info(f"  {output_dir}/event_index.parquet  — event_id, event_text, code, value, unit")
        logger.info(f"  {output_dir}/embeddings.npy       — shape ({n_total}, {hidden_size}), float32")
        logger.info(f"  Lookup: embeddings[event_index[event_index.event_text == text].event_id[0]]")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
