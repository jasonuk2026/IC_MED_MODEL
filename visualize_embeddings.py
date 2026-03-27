#!/usr/bin/env python3
"""
visualize_embeddings.py

UMAP visualisation of patient embeddings: base model vs fine-tuned side-by-side.

Usage:
    conda run -n torch python visualize_embeddings.py \
        --checkpoint output/ehrshot_embed/<run_dir>/best \
        --output_png figures/embedding_umap.png
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from peft import PeftModel
import umap

hf_logging.set_verbosity_error()

# ── Config ────────────────────────────────────────────────────────────────────

DISEASES = {
    "new_hypertension":   "Hypertension",
    "new_hyperlipidemia": "Hyperlipidemia",
    "new_pancan":         "Pancreatic Cancer",
    "new_celiac":         "Celiac Disease",
    "new_lupus":          "Lupus",
    "new_acutemi":        "Acute MI",
}

DISEASE_COLORS = {
    "new_hypertension":   "#2196F3",   # blue
    "new_hyperlipidemia": "#9C27B0",   # purple
    "new_pancan":         "#E53935",   # red
    "new_celiac":         "#00ACC1",   # cyan
    "new_lupus":          "#43A047",   # green
    "new_acutemi":        "#FB8C00",   # orange
}

SEED = 42


# ── Text formatting (mirrors train_embedding_custom.py exactly) ───────────────

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}


def format_events(events) -> str:
    lines = []
    for e in events:
        desc  = e.get("description") or ""
        code  = e.get("code")        or ""
        value = e.get("value")       or ""
        unit  = e.get("unit")        or ""
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


def build_prompt(disease_name: str, events) -> str:
    return (
        f"Please predict disease {disease_name} based on the following events.\n"
        f"Start of medical events:{format_events(events)}<|endoftext|>"
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples(data_dir: str, n_per_class: int, seed: int) -> pd.DataFrame:
    """Sample n_per_class pos + neg patients per disease from the test split."""
    rng = random.Random(seed)
    data_dir = Path(data_dir)
    chunks = []

    for task in DISEASES:
        # Prefer m500; fall back to any parquet for this task
        candidates = sorted(data_dir.glob(f"{task}_all_*m500*ntrain50000*.parquet"))
        if not candidates:
            candidates = sorted(data_dir.glob(f"{task}_all_*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet found for {task} in {data_dir}")
        path = candidates[0]
        print(f"  {task}: {path.name}")

        df = pd.read_parquet(path, columns=["task", "split", "label", "events"])
        df = df[df["split"] == "test"].reset_index(drop=True)

        for label_val in [True, False]:
            sub = df[df["label"] == label_val]
            idx = rng.sample(range(len(sub)), min(n_per_class, len(sub)))
            chunks.append(sub.iloc[idx])

    result = pd.concat(chunks, ignore_index=True)
    print(f"\n  Total: {len(result)} samples")
    print(result.groupby(["task", "label"]).size().rename("n").to_string())
    return result


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def encode(model, tokenizer, texts: list[str], device, batch_size: int) -> np.ndarray:
    model.eval()
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), leave=False):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk, padding=True, truncation=True, max_length=512,
            add_special_tokens=False, return_tensors="pt",
        ).to(device)
        out  = model(**enc)
        mask = enc["attention_mask"]
        # last-token pool (same as training)
        seq_len = mask.sum(dim=1) - 1
        bidx = torch.arange(out.last_hidden_state.size(0), device=device)
        emb  = out.last_hidden_state[bidx, seq_len]
        emb  = F.normalize(emb.float(), p=2, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(embs: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",        # embeddings are L2-normalised → cosine makes sense
        random_state=SEED,
        verbose=False,
    )
    return reducer.fit_transform(embs)


# ── Plot ──────────────────────────────────────────────────────────────────────

def draw_panel(ax, xy: np.ndarray, df: pd.DataFrame, title: str, annotate: bool = False):
    ax.set_facecolor("#F7F8FA")
    ax.grid(True, color="white", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    tasks   = df["task"].values
    labels  = df["label"].values

    for task in DISEASES:
        color = DISEASE_COLORS[task]
        # positives: filled circle
        mask_pos = (tasks == task) & (labels == True)
        if mask_pos.any():
            ax.scatter(xy[mask_pos, 0], xy[mask_pos, 1],
                       c=color, marker="o", s=40, alpha=0.85,
                       linewidths=0.3, edgecolors="white", zorder=3)
        # negatives: hollow triangle
        mask_neg = (tasks == task) & (labels == False)
        if mask_neg.any():
            ax.scatter(xy[mask_neg, 0], xy[mask_neg, 1],
                       facecolors="none", edgecolors=color,
                       marker="^", s=28, alpha=0.65,
                       linewidths=0.8, zorder=2)

    if annotate:
        for task, label in DISEASES.items():
            mask = tasks == task
            if not mask.any():
                continue
            cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
            ax.annotate(
                label, xy=(cx, cy),
                fontsize=8.5, fontweight="bold",
                color=DISEASE_COLORS[task], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          alpha=0.80, ec=DISEASE_COLORS[task], lw=1.2),
                zorder=5,
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#222")
    ax.set_xlabel("UMAP 1", fontsize=9, color="#666")
    ax.set_ylabel("UMAP 2", fontsize=9, color="#666")
    ax.tick_params(colors="#aaa", labelsize=7)


def build_legend():
    handles = []
    for task, label in DISEASES.items():
        handles.append(mpatches.Patch(color=DISEASE_COLORS[task], label=label))
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="grey", markersize=8,
                               label="Positive (new Dx within 1 yr)"))
    handles.append(plt.Line2D([0], [0], marker="^", color="w",
                               markerfacecolor="none", markeredgecolor="grey",
                               markersize=8, label="Negative"))
    return handles


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",   required=True,
                   help="Path to fine-tuned PEFT checkpoint directory.")
    p.add_argument("--model_name",   default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--data_dir",     default="data/embedding_inputs/new_diagnosis")
    p.add_argument("--output_png",   default="figures/embedding_umap.png")
    p.add_argument("--n_samples",    type=int, default=150,
                   help="Samples per (disease × pos/neg) class.")
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--device",       default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}\n")

    # ── Load data & build prompts ──────────────────────────────────────────────
    print("Loading test samples ...")
    df = load_samples(args.data_dir, args.n_samples, SEED)
    texts = [
        build_prompt(TASK_2_DISEASE_NAME[row.task], row.events)
        for row in df.itertuples()
    ]

    # ── Base model embeddings ──────────────────────────────────────────────────
    print(f"\n[1/4] Loading base model ...")
    tokenizer  = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=True, padding_side="right"
    )
    base_model = AutoModel.from_pretrained(
        args.model_name, local_files_only=True, torch_dtype=dtype,
    ).to(device)

    print("[2/4] Encoding with BASE model ...")
    embs_base = encode(base_model, tokenizer, texts, device, args.batch_size)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Fine-tuned model embeddings ────────────────────────────────────────────
    print(f"\n[2/4] Loading fine-tuned model from {args.checkpoint} ...")
    ft_base  = AutoModel.from_pretrained(
        args.model_name, local_files_only=True, torch_dtype=dtype,
    )
    ft_base.config.use_cache = False
    ft_model = PeftModel.from_pretrained(ft_base, args.checkpoint).to(device)

    print("[3/4] Encoding with FINE-TUNED model ...")
    embs_ft = encode(ft_model, tokenizer, texts, device, args.batch_size)
    del ft_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Running UMAP ...")
    print("  base model ...", end=" ", flush=True)
    xy_base = run_umap(embs_base)
    print("done")
    print("  fine-tuned ...", end=" ", flush=True)
    xy_ft = run_umap(embs_ft)
    print("done")

    # ── Plot ──────────────────────────────────────────────────────────────────
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    draw_panel(axes[0], xy_base, df,
               title="Base Model  (Qwen3-Embedding-0.6B, no fine-tuning)",
               annotate=False)
    draw_panel(axes[1], xy_ft, df,
               title="Fine-tuned  (LoRA + BatchAllTripletLoss)",
               annotate=True)

    fig.legend(
        handles=build_legend(),
        loc="lower center", ncol=6,
        fontsize=10, frameon=True, framealpha=0.95,
        edgecolor="#ddd", bbox_to_anchor=(0.5, -0.05),
    )

    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    fig.suptitle(
        f"Patient Embedding Space — 4 Diseases, Test Split  "
        f"(n={len(df)}: {n_pos} positive, {n_neg} negative)",
        fontsize=13, y=1.01, color="#222",
    )

    plt.tight_layout()
    plt.savefig(args.output_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved → {args.output_png}")


if __name__ == "__main__":
    main()
