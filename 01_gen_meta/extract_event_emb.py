#!/usr/bin/env python3
"""
Offline extraction of per-event embeddings from a unique-events parquet.

The unique semantic events are expected to come from
`01_gen_meta/extract_unique_events.py`, which writes a parquet with one row per
deduplicated event and preserves structured columns for Jinja-based text
templating.

Pooling: mean pooling over non-padding, non-special tokens (no [CLS]/[SEP]).

─── Pipeline ───────────────────────────────────────────────────────────────
Phase 1  (rank 0): load unique-events parquet → render event_text with Jinja →
             save  <output_dir>/event_index.parquet
Phase 2  (all ranks): each rank encodes its own slice → saves
             <output_dir>/shards/shard_{rank:04d}.npy
Phase 3  (rank 0): merge shards → <output_dir>/embeddings.npy
             (row i  ↔  event_index.parquet row i)

─── Usage ──────────────────────────────────────────────────────────────────
Single-node, 4 GPUs:
    torchrun --nproc_per_node=4 extract_event_emb.py

Multi-node (2 × 4 GPUs):
    torchrun --nnodes=2 --nproc_per_node=4 \\
        --rdzv_id=job1 --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
        extract_event_emb.py [args]

Single GPU / CPU (debug):
    python extract_event_emb.py --batch_size 32
"""

import os
import json
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from encoders import get_encoder

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


# ── Phase 1: render unique event texts from parquet ───────────────────────────

def build_event_index(
    unique_events_path: str,
    output_dir: Path,
    encoder,
) -> pd.DataFrame:
    """Load unique events parquet, render Jinja event_text, write event_index."""
    index_path = output_dir / "event_index.parquet"
    if index_path.exists():
        logger.info(f"[Phase 1] Found existing event_index.parquet — skipping scan.")
        return pd.read_parquet(index_path)

    logger.info("[Phase 1] Loading unique events from %s ...", unique_events_path)
    event_df = pd.read_parquet(unique_events_path)
    required_cols = {
        "event_id", "omop_table", "event_type", "code", "description", "value", "unit",
    }
    missing = sorted(required_cols - set(event_df.columns))
    if missing:
        raise ValueError(
            "Unique events parquet is missing required columns: {}".format(", ".join(missing))
        )

    rows = []
    n_no_text = 0
    for row in event_df.itertuples(index=False):
        code = str(row.code or "").strip()
        description = str(row.description or "").strip()
        value = str(row.value or "").strip()
        unit = str(row.unit or "").strip()
        omop_table = str(row.omop_table or "").strip()
        event_type = str(row.event_type or "").strip()
        text = encoder.format_event_text(
            code=code,
            description=description,
            value=value,
            unit=unit,
            omop_table=omop_table,
            event_type=event_type,
        )
        if text is None:
            n_no_text += 1
            continue
        rows.append({
            "event_id": int(row.event_id),
            "omop_table": omop_table,
            "event_type": event_type,
            "code": code,
            "description": description,
            "value": value,
            "unit": unit,
            "event_text": text,
        })

    logger.info(
        "[Phase 1] Loaded %s unique events → %s encodable (%s dropped by template).",
        "{:,}".format(len(event_df)),
        "{:,}".format(len(rows)),
        "{:,}".format(n_no_text),
    )

    event_df = pd.DataFrame(rows)
    event_df = event_df.sort_values("event_id").reset_index(drop=True)

    event_df.to_parquet(index_path, index=False)
    logger.info(f"[Phase 1] Saved event_index.parquet → {index_path}")
    return event_df


# ── Phase 2: encode one rank's slice ──────────────────────────────────────────

@torch.inference_mode()
def encode_slice(
    texts:      list[str],
    model:      AutoModel,
    tokenizer:  AutoTokenizer,
    encoder,
    device:     torch.device,
    batch_size: int,
    max_length: int,
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
            add_special_tokens=encoder.ADD_SPECIAL_TOKENS,
            return_tensors="pt",
        ).to(device)

        assert enc.input_ids[0].size(0) < max_length, f"Shouldn't exist sequence exceeding {max_length}"

        out = model(**enc)
        embs = encoder.get_embeddings(out, enc, tokenizer)
        embs = encoder.postprocess_embeddings(embs)
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


def log_tokenization_preview(texts, tokenizer, encoder, n_examples=5, seed=0):
    if not texts or n_examples <= 0:
        return

    n_examples = min(n_examples, len(texts))
    rng = random.Random(seed)
    sample_indices = sorted(rng.sample(range(len(texts)), n_examples))

    logger.info("=" * 80)
    logger.info(
        "Tokenization preview: %s random event(s), add_special_tokens=%s",
        n_examples,
        encoder.ADD_SPECIAL_TOKENS,
    )
    logger.info("Pooling mode: %s", encoder.pooling_mode)
    logger.info("=" * 80)

    for preview_idx, text_idx in enumerate(sample_indices, start=1):
        text = texts[text_idx]
        encoded = tokenizer(
            text,
            padding=False,
            truncation=False,
            add_special_tokens=encoder.ADD_SPECIAL_TOKENS,
            return_tensors="pt",
        )
        input_ids_tensor = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        input_ids = input_ids_tensor[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        decoded_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
        decoded_without_special = tokenizer.decode(input_ids, skip_special_tokens=True)
        special_token_ids = set(tokenizer.all_special_ids) if encoder.ADD_SPECIAL_TOKENS else None
        pool_mask = encoder.build_pool_mask(
            attention_mask=attention_mask,
            input_ids=input_ids_tensor,
            special_token_ids=special_token_ids,
        )[0].tolist()

        logger.info("[Preview %s] event_index=%s", preview_idx, text_idx)
        logger.info("  original_text: %s", text)
        logger.info("  input_ids: %s", input_ids)
        logger.info("  tokens: %s", tokens)
        logger.info("  decoded(skip_special_tokens=False): %s", decoded_with_special)
        logger.info("  decoded(skip_special_tokens=True):  %s", decoded_without_special)
        logger.info("  token_contributes_to_embedding:")
        for token_pos, (token_id, token_text, keep) in enumerate(zip(input_ids, tokens, pool_mask)):
            logger.info(
                "    [%02d] id=%s keep=%s token=%r",
                token_pos,
                token_id,
                bool(keep),
                token_text,
            )
        logger.info("-" * 80)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--unique_events_path",
        default="data/01_outputs/unique_events.parquet",
        help="Parquet produced by extract_unique_events.py.",
    )
    p.add_argument("--model_name",   default="michiyasunaga/BioLinkBERT-base",
                   help="Base HuggingFace model name. Also used as the default tokenizer source.")
    p.add_argument("--model_path",   default=None,
                   help="Optional local checkpoint/model path for AutoModel.from_pretrained. Defaults to --model_name.")
    p.add_argument("--tokenizer_name", default=None,
                   help="Optional tokenizer source for AutoTokenizer.from_pretrained. Defaults to --model_name.")
    p.add_argument("--encoder",      default="biolinkbert",
                   help="Encoder backend name. Controls model defaults, Jinja template, and pooling.")
    p.add_argument("--template_path", default=None,
                   help="Optional custom Jinja template path for event text formatting.")
    p.add_argument("--append_token_text", default=None,
                   help="Optional token text appended to the end of every rendered event text.")
    p.add_argument("--append_token_name", default=None,
                   help="Optional existing tokenizer token attribute to append, e.g. pad_token or eos_token.")
    p.add_argument("--pool_max_tokens", type=int, default=None,
                   help="Optional cap: mean-pool over only the first N valid tokens after masking.")
    p.add_argument("--pooling_mode", default="mean", choices=["mean", "suffix_only"],
                   help="How to collapse token hidden states into one event embedding.")
    p.add_argument("--output_dir",   default="data/biolinkbert_embeddings")
    p.add_argument("--batch_size",   type=int, default=256,
                   help="Tokenisation + forward-pass batch size per rank.")
    p.add_argument("--max_length",   type=int, default=256,
                   help="Max token length. BioLinkBERT-base supports up to 512.")
    p.add_argument("--preview_tokenization_n", type=int, default=5,
                   help="On rank 0, log this many random tokenization/decode examples before encoding. Use 0 to disable.")
    p.add_argument("--preview_seed", type=int, default=0,
                   help="Random seed for tokenization preview sampling.")
    p.add_argument("--fp16",         action="store_true",
                   help="Run model in float16 (saves GPU memory, minor precision loss).")
    p.add_argument("--bf16",         action="store_true",
                   help="Run model in bfloat16.")
    p.add_argument(
        "--attn_implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional attention implementation passed to AutoModel.from_pretrained. "
             "Useful for decoder-only models such as Qwen when matching training-time inference behavior.",
    )
    p.add_argument("--local_files_only", action="store_true",
                   help="Load model from local cache only (no HuggingFace download).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    encoder = get_encoder(
        args.encoder,
        model_name=args.model_name,
        template_path=args.template_path,
        append_token_text=args.append_token_text,
        append_token_name=args.append_token_name,
        pool_max_tokens=args.pool_max_tokens,
        pooling_mode=args.pooling_mode,
    )
    model_source = args.model_path or encoder.model_name
    tokenizer_source = args.tokenizer_name or encoder.model_name

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

    if rank == 0 and args.append_token_name and not encoder.append_token_text:
        tokenizer_preview = AutoTokenizer.from_pretrained(
            tokenizer_source, **({} if not args.local_files_only else {"local_files_only": True})
        )
        encoder.resolve_append_token(tokenizer_preview)
        logger.info(
            "Resolved append_token_name=%s to token=%r using tokenizer=%s",
            args.append_token_name,
            encoder.append_token_text,
            tokenizer_source,
        )

    # ── Phase 1: render unique event texts (rank 0 only) ──────────────────────
    if rank == 0:
        event_df = build_event_index(
            args.unique_events_path, output_dir, encoder
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
        logger.info(
            f"[Rank {rank}] Loading encoder={args.encoder} "
            f"model={model_source} tokenizer={tokenizer_source} ..."
        )
        model_kwargs = {}
        if args.fp16:
            model_kwargs["dtype"] = torch.float16
        elif args.bf16:
            model_kwargs["dtype"] = torch.bfloat16
        if args.attn_implementation is not None:
            model_kwargs["attn_implementation"] = args.attn_implementation
        if args.local_files_only:
            model_kwargs["local_files_only"] = True

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, **({} if not args.local_files_only else {"local_files_only": True})
        )
        model = AutoModel.from_pretrained(model_source, **model_kwargs).to(device)
        tokenizer, model = encoder.configure_tokenizer_and_model(tokenizer, model)
        model.eval()
        logger.info(
            "[Rank %s] Loaded model with dtype=%s attn_implementation=%s",
            rank,
            str(model.dtype),
            args.attn_implementation or "default",
        )

        if rank == 0 and args.preview_tokenization_n > 0:
            log_tokenization_preview(
                texts=texts,
                tokenizer=tokenizer,
                encoder=encoder,
                n_examples=args.preview_tokenization_n,
                seed=args.preview_seed,
            )

        logger.info(
            f"[Rank {rank}] Encoding {len(my_texts):,} texts "
            f"(of {n_total:,} total, world_size={world_size}) ..."
        )
        embs = encode_slice(
            my_texts, model, tokenizer, encoder, device,
            batch_size=args.batch_size,
            max_length=args.max_length,
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
        logger.info(
            f"  encoder={args.encoder}  model={model_source}  tokenizer={tokenizer_source}  "
            f"attn_implementation={args.attn_implementation or 'default'}  "
            f"base_model={encoder.model_name}  template={encoder.template_path}  "
            f"append_token={encoder.append_token_text}  append_token_name={encoder.append_token_name}  "
            f"pool_max_tokens={encoder.pool_max_tokens}  pooling_mode={encoder.pooling_mode}"
        )
        logger.info(f"  Lookup: embeddings[event_index[event_index.event_text == text].event_id[0]]")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
