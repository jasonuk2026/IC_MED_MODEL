#!/usr/bin/env python3
"""
evaluate_embedding.py — Standalone evaluation for trained EHR embedding models.

Supports both:
  - Base model (no fine-tuning, for baseline comparison)
  - LoRA checkpoint saved by train_embedding_custom.py

Metrics reported:
  - Triplet accuracy (per task and overall): fraction where d(a,p) < d(a,n)
  - Mean intra-class cosine similarity (pos-pos pairs within same task)
  - Mean inter-class cosine similarity (pos-neg pairs within same task)
  - Gap = intra - inter (higher is better)

Usage:
  # Evaluate fine-tuned checkpoint
  python evaluate_embedding.py \\
      --checkpoint output/medical-embedding-custom/20250325_120000/epoch_1 \\
      --base_model Qwen/Qwen3-Embedding-0.6B \\
      --data_paths data/embedding_inputs/new_diagnosis/new_hypertension_all_p0.5_m500_n1000.parquet \\
      --split val --bf16

  # Evaluate base model (no LoRA)
  python evaluate_embedding.py \\
      --base_model Qwen/Qwen3-Embedding-0.6B \\
      --data_paths data/embedding_inputs/new_diagnosis/new_hypertension_all_p0.5_m500_n1000.parquet \\
      --split val --bf16
"""

import argparse
import logging
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Constants (must match training script) ────────────────────────────────────

SPECIAL_TOKENS = ["<|disease_start|>", "<|disease_end|>", "<|event_start|>", "<|event_end|>"]

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}


# ── Text formatting ────────────────────────────────────────────────────────────

def format_events(events: list) -> str:
    lines = []
    for e in events:
        desc  = e.get("description") or ""
        code  = e.get("code") or ""
        value = e.get("value") or ""
        unit  = e.get("unit") or ""
        if desc and code:
            line = f"{desc} [{code}]"
        elif code:
            line = f"[{code}]"
        elif desc:
            line = desc
        else:
            continue
        if value:
            line += f" | value={value}"
        if unit:
            line += f" | unit={unit}"
        lines.append(line)
    return "\n".join(lines)


def build_prompt(disease_name: str, events: list) -> str:
    return (
        f"<|disease_start|>{disease_name}<|disease_end|>"
        f"<|event_start|>{format_events(events)}<|event_end|>"
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(args):
    """Load base model + optionally apply LoRA checkpoint."""
    model_kwargs = {}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if args.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["dtype"] = torch.float16

    logger.info(f"Loading base model: {args.base_model}")
    model     = AutoModel.from_pretrained(args.base_model, local_files_only=True, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True,
                                              padding_side="right")

    # Add special tokens (must match training setup)
    tokenizer.add_special_tokens({"extra_special_tokens": SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    for tok, phrase in [
        ("<|disease_start|>", "disease name"),
        ("<|disease_end|>",   "disease name end"),
        ("<|event_start|>",   "medical event start"),
        ("<|event_end|>",     "<|endoftext|>"),
    ]:
        ids      = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        mean_emb = model.embed_tokens.weight.data[ids].mean(dim=0)
        tok_id   = tokenizer.convert_tokens_to_ids(tok)
        model.embed_tokens.weight.data[tok_id] = mean_emb
    pad_id = tokenizer.convert_tokens_to_ids("<|event_end|>")
    tokenizer.pad_token_id = pad_id

    if args.checkpoint:
        logger.info(f"Loading LoRA checkpoint: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()   # fold LoRA weights into base for faster inference
        logger.info("LoRA weights merged into base model.")

    model.eval()
    return model, tokenizer


# ── Encoding ──────────────────────────────────────────────────────────────────

def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx   = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, seq_lengths]


@torch.inference_mode()
def encode_texts(model, tokenizer, texts: list, device, batch_size: int,
                 show_progress: bool = True) -> torch.Tensor:
    all_emb = []
    batches = range(0, len(texts), batch_size)
    if show_progress:
        batches = tqdm(batches, desc="Encoding", dynamic_ncols=True)
    for i in batches:
        chunk = texts[i : i + batch_size]
        enc   = tokenizer(chunk, padding=True, return_tensors="pt", add_special_tokens=False)
        enc   = {k: v.to(device) for k, v in enc.items()}
        out   = model(**enc, use_cache=False)
        emb   = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb   = F.normalize(emb.float(), p=2, dim=-1)
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_paths: list, split: str) -> pd.DataFrame:
    frames = []
    for p in data_paths:
        df = pd.read_parquet(p)
        frames.append(df[df["split"] == split])
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows for split='{split}'")
    return df


def build_text_groups(df: pd.DataFrame) -> dict:
    """Build {(task, label): [prompt, ...]} mapping."""
    groups = defaultdict(list)
    for task, events, label in zip(df["task"], df["events"], df["label"]):
        if not format_events(events).strip():
            continue
        disease_name = TASK_2_DISEASE_NAME.get(task, task)
        groups[(task, bool(label))].append(build_prompt(disease_name, events))
    return groups


# ── Metrics ───────────────────────────────────────────────────────────────────

def triplet_accuracy(a_emb, p_emb, n_emb) -> float:
    d_ap = (a_emb - p_emb).norm(dim=1)
    d_an = (a_emb - n_emb).norm(dim=1)
    return (d_ap < d_an).float().mean().item()


def cosine_similarity_stats(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """Mean cosine similarity between two sets of embeddings (paired)."""
    return (emb_a * emb_b).sum(dim=1).mean().item()


def build_triplets(groups: dict, n_per_task: int, seed: int):
    rng = random.Random(seed)
    anchors, positives, negatives, task_ids = [], [], [], []
    for task in sorted({t for t, _ in groups}):
        pos = groups.get((task, True), [])
        neg_all = [t for (tsk, pol), texts in groups.items()
                   if not (tsk == task and pol) for t in texts]
        if len(pos) < 2 or not neg_all:
            logger.warning(f"  Skipping task '{task}': pos={len(pos)}, neg={len(neg_all)}")
            continue
        n    = min(n_per_task, len(pos))
        pool = list(range(len(pos)))
        rng.shuffle(pool)
        for i in range(n):
            ai = pool[i % len(pos)]
            pi = rng.choice([j for j in pool if j != ai])
            ni = rng.randint(0, len(neg_all) - 1)
            anchors.append(pos[ai])
            positives.append(pos[pi])
            negatives.append(neg_all[ni])
            task_ids.append(task)
    return anchors, positives, negatives, task_ids


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(model, tokenizer, df: pd.DataFrame, device, args):
    groups = build_text_groups(df)

    # ── Collect all unique texts and encode once ───────────────────────────────
    all_tasks = sorted({t for t, _ in groups})
    logger.info(f"Tasks found: {all_tasks}")

    anchors, positives, negatives, task_ids = build_triplets(
        groups, n_per_task=args.n_triplets_per_task, seed=args.seed,
    )
    if not anchors:
        logger.error("No triplets could be built.")
        return

    logger.info(f"Built {len(anchors)} triplets across {len(set(task_ids))} tasks")

    # Encode all texts in one pass (deduplicated)
    all_texts   = anchors + positives + negatives
    all_embs    = encode_texts(model, tokenizer, all_texts, device,
                               batch_size=args.batch_size)
    n           = len(anchors)
    a_emb       = all_embs[:n]
    p_emb       = all_embs[n:2*n]
    neg_emb     = all_embs[2*n:]
    task_ids_t  = task_ids

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall_acc    = triplet_accuracy(a_emb, p_emb, neg_emb)
    sim_ap         = cosine_similarity_stats(a_emb, p_emb)
    sim_an         = cosine_similarity_stats(a_emb, neg_emb)

    print("\n" + "="*60)
    print(f"  Split: {args.split}   Triplets: {n}")
    print("="*60)
    print(f"  Overall triplet accuracy : {overall_acc:.4f}")
    print(f"  Mean cosine sim (a,p)    : {sim_ap:.4f}")
    print(f"  Mean cosine sim (a,n)    : {sim_an:.4f}")
    print(f"  Gap (a,p) - (a,n)        : {sim_ap - sim_an:.4f}  (higher = better)")
    print("="*60)

    # ── Per-task metrics ──────────────────────────────────────────────────────
    print("\nPer-task breakdown:")
    print(f"  {'Task':<25} {'Acc':>6}  {'sim(a,p)':>9}  {'sim(a,n)':>9}  {'Gap':>7}  {'N':>5}")
    print("  " + "-"*65)

    task_indices = defaultdict(list)
    for i, t in enumerate(task_ids_t):
        task_indices[t].append(i)

    for task in sorted(task_indices):
        idx     = task_indices[task]
        ta_emb  = a_emb[idx]
        tp_emb  = p_emb[idx]
        tn_emb  = neg_emb[idx]
        acc     = triplet_accuracy(ta_emb, tp_emb, tn_emb)
        sap     = cosine_similarity_stats(ta_emb, tp_emb)
        san     = cosine_similarity_stats(ta_emb, tn_emb)
        print(f"  {task:<25} {acc:>6.4f}  {sap:>9.4f}  {san:>9.4f}  {sap-san:>7.4f}  {len(idx):>5}")

    # ── Intra-class similarity (pos-pos within each task) ─────────────────────
    print("\nPos-pos intra-class cosine similarity (all pairs, not just triplets):")
    print(f"  {'Task':<25} {'mean':>7}  {'N pairs':>8}")
    print("  " + "-"*45)
    for task in sorted(all_tasks):
        pos_texts = groups.get((task, True), [])
        if len(pos_texts) < 2:
            continue
        # Sample up to 200 texts to avoid OOM
        sample = pos_texts[:200]
        embs   = encode_texts(model, tokenizer, sample, device,
                              batch_size=args.batch_size, show_progress=False)
        # All unique pairs
        n_s = embs.size(0)
        sim_matrix = embs @ embs.T   # (n, n) cosine sim (already normalized)
        mask = torch.triu(torch.ones(n_s, n_s, dtype=torch.bool), diagonal=1)
        mean_sim = sim_matrix[mask].mean().item()
        n_pairs  = mask.sum().item()
        print(f"  {task:<25} {mean_sim:>7.4f}  {n_pairs:>8}")

    print()


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base_model",  required=True,
                   help="Base model name or path (e.g. Qwen/Qwen3-Embedding-0.6B).")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to LoRA checkpoint dir saved by train_embedding_custom.py. "
                        "If not set, evaluates the base model without fine-tuning.")
    p.add_argument("--data_paths",  nargs="+", required=True)
    p.add_argument("--split",       default="val", choices=["train", "val", "test"])
    p.add_argument("--bf16",        action="store_true")
    p.add_argument("--fp16",        action="store_true")
    p.add_argument("--flash_attn",  action="store_true")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--n_triplets_per_task", type=int, default=200)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    model, tokenizer = load_model(args)
    model = model.to(device)

    df = load_data(args.data_paths, split=args.split)
    run_evaluation(model, tokenizer, df, device, args)


if __name__ == "__main__":
    main()
