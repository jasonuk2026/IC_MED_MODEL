#!/usr/bin/env python3
"""
visualize_embeddings.py

Per-disease UMAP: 2 rows (base / fine-tuned) × 6 columns (diseases).
Each panel shows only that disease's patients, coloured by pos (red) vs neg (blue).
This directly answers "can the model separate positive from negative for this disease?"

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
from sklearn.metrics import roc_auc_score
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

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}

POS_COLOR = "#E53935"   # red
NEG_COLOR = "#1E88E5"   # blue

N_PER_CLASS = 200   # pos + neg samples per disease
SEED = 42


# ── Text formatting (mirrors train_embedding_custom.py exactly) ───────────────

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

def load_samples(data_dir: str, n_per_class: int, seed: int) -> dict[str, pd.DataFrame]:
    """Returns {task: DataFrame} with n_per_class pos + neg from test split."""
    rng = random.Random(seed)
    data_dir = Path(data_dir)
    result = {}

    for task in DISEASES:
        candidates = sorted(data_dir.glob(f"{task}_all_*m500*ntrain50000*.parquet"))
        if not candidates:
            candidates = sorted(data_dir.glob(f"{task}_all_*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet found for {task} in {data_dir}")

        df = pd.read_parquet(candidates[0], columns=["task", "split", "label", "events"])
        df = df[df["split"] == "test"].reset_index(drop=True)

        chunks = []
        for label_val in [True, False]:
            sub = df[df["label"] == label_val]
            idx = rng.sample(range(len(sub)), min(n_per_class, len(sub)))
            chunks.append(sub.iloc[idx])
        result[task] = pd.concat(chunks, ignore_index=True)

    return result


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_disease(
    model, tokenizer, df: pd.DataFrame,
    disease_name: str, device, batch_size: int,
) -> np.ndarray:
    """Encode all patients for one disease → (N, H) float32 L2-normalised."""
    model.eval()
    texts = [build_prompt(disease_name, row.events) for row in df.itertuples()]
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk, padding=True, truncation=True, max_length=512,
            add_special_tokens=False, return_tensors="pt",
        ).to(device)
        out  = model(**enc)
        mask = enc["attention_mask"]
        seq_len = mask.sum(dim=1) - 1
        bidx = torch.arange(out.last_hidden_state.size(0), device=device)
        emb  = out.last_hidden_state[bidx, seq_len]
        emb  = F.normalize(emb.float(), p=2, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ── UMAP (per-disease) ────────────────────────────────────────────────────────

def run_umap(embs: np.ndarray) -> np.ndarray:
    np.random.seed(SEED)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.05,
        metric="cosine",
        verbose=False,
    )
    return reducer.fit_transform(embs)


# ── Metric: cosine-sim AUC (pos closer together than neg?) ────────────────────

def separation_auc(embs: np.ndarray, labels: np.ndarray) -> float:
    """
    Score each sample by its mean cosine similarity to same-class neighbours
    minus mean cosine similarity to opposite-class neighbours.
    Then compute AUC of pos vs neg scores — 0.5 = random, 1.0 = perfect.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return float("nan")

    sim = embs @ embs.T   # (N, N), already L2-normalised → cosine sim

    scores = np.zeros(len(labels))
    for i in range(len(labels)):
        same = pos_mask if labels[i] == 1 else neg_mask
        diff = neg_mask if labels[i] == 1 else pos_mask
        same_idx = np.where(same)[0]
        diff_idx  = np.where(diff)[0]
        same_idx = same_idx[same_idx != i]
        scores[i] = sim[i, same_idx].mean() - sim[i, diff_idx].mean()

    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return float("nan")


# ── Draw one panel ────────────────────────────────────────────────────────────

def draw_panel(ax, xy: np.ndarray, labels: np.ndarray,
               title: str, auc: float):
    ax.set_facecolor("#F7F8FA")
    ax.grid(True, color="white", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    pos = labels == 1
    neg = labels == 0

    ax.scatter(xy[neg, 0], xy[neg, 1],
               c=NEG_COLOR, marker="o", s=22, alpha=0.55,
               linewidths=0, zorder=2, label="Negative")
    ax.scatter(xy[pos, 0], xy[pos, 1],
               c=POS_COLOR, marker="o", s=22, alpha=0.80,
               linewidths=0, zorder=3, label="Positive")

    auc_str = f"AUC={auc:.3f}" if not np.isnan(auc) else ""
    ax.set_title(f"{title}\n{auc_str}", fontsize=8.5, pad=4, color="#222")
    ax.set_xticks([])
    ax.set_yticks([])


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",  required=True,
                   help="Path to fine-tuned PEFT checkpoint directory.")
    p.add_argument("--model_name",  default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--data_dir",    default="data/embedding_inputs/new_diagnosis")
    p.add_argument("--output_png",  default="figures/embedding_umap.png")
    p.add_argument("--n_samples",   type=int, default=N_PER_CLASS,
                   help="Samples per pos/neg class per disease.")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--bf16",        action="store_true")
    p.add_argument("--device",      default=None)
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

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading test samples ...")
    disease_dfs = load_samples(args.data_dir, args.n_samples, SEED)
    for task, df in disease_dfs.items():
        n_pos = int(df["label"].sum())
        print(f"  {task}: {len(df)} samples ({n_pos} pos, {len(df)-n_pos} neg)")

    task_list = list(DISEASES.keys())
    n_diseases = len(task_list)

    # ── Load tokenizer once ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=True, padding_side="right"
    )

    # ── Encode with both models ────────────────────────────────────────────────
    all_embs = {}   # {("base"|"ft", task): np.ndarray}

    for model_tag, load_fn in [
        ("base", lambda: AutoModel.from_pretrained(
            args.model_name, local_files_only=True, torch_dtype=dtype).to(device)),
        ("ft", lambda: PeftModel.from_pretrained(
            AutoModel.from_pretrained(
                args.model_name, local_files_only=True, torch_dtype=dtype
            ).eval(),
            args.checkpoint,
        ).to(device)),
    ]:
        print(f"\nEncoding with {model_tag} model ...")
        model = load_fn()
        model.config.use_cache = False

        for task in tqdm(task_list, desc=f"  [{model_tag}]"):
            df = disease_dfs[task]
            disease_name = TASK_2_DISEASE_NAME[task]
            embs = encode_disease(model, tokenizer, df, disease_name, device, args.batch_size)
            all_embs[(model_tag, task)] = embs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── UMAP + AUC per (model, disease) ───────────────────────────────────────
    print("\nRunning UMAP ...")
    xy     = {}
    aucs   = {}
    for model_tag in ("base", "ft"):
        for task in tqdm(task_list, desc=f"  [{model_tag}] UMAP"):
            embs   = all_embs[(model_tag, task)]
            labels = disease_dfs[task]["label"].astype(int).values
            xy[(model_tag, task)]   = run_umap(embs)
            aucs[(model_tag, task)] = separation_auc(embs, labels)

    # ── Plot: 2 rows × 6 cols ─────────────────────────────────────────────────
    print("\nPlotting ...")
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, n_diseases, figsize=(n_diseases * 3.2, 7))
    fig.patch.set_facecolor("white")

    row_labels = ["Base Model", "Fine-tuned"]
    for row_idx, model_tag in enumerate(("base", "ft")):
        for col_idx, task in enumerate(task_list):
            ax     = axes[row_idx, col_idx]
            labels = disease_dfs[task]["label"].astype(int).values
            draw_panel(
                ax,
                xy[(model_tag, task)],
                labels,
                title=DISEASES[task],
                auc=aucs[(model_tag, task)],
            )
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=10,
                              fontweight="bold", color="#333", labelpad=6)

    # Row divider line
    fig.add_artist(plt.Line2D(
        [0.02, 0.98], [0.505, 0.505],
        transform=fig.transFigure,
        color="#ccc", linewidth=1,
    ))

    # Legend
    legend_handles = [
        mpatches.Patch(color=POS_COLOR, label="Positive (new diagnosis within 1 yr)"),
        mpatches.Patch(color=NEG_COLOR, label="Negative"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=10, frameon=True, framealpha=0.95,
               edgecolor="#ddd", bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Per-Disease Patient Embedding Space  —  Base vs Fine-tuned  (Test Split)\n"
        "AUC: same-class cosine similarity vs cross-class (higher = better separation)",
        fontsize=11, y=1.02, color="#222",
    )

    plt.tight_layout(h_pad=1.5, w_pad=0.5)
    plt.savefig(args.output_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved → {args.output_png}")

    # ── Print AUC summary ─────────────────────────────────────────────────────
    print(f"\n{'Disease':<22} {'Base AUC':>10} {'FT AUC':>10} {'Δ':>8}")
    print("-" * 52)
    for task in task_list:
        b = aucs[("base", task)]
        f = aucs[("ft",   task)]
        delta = f - b if not (np.isnan(b) or np.isnan(f)) else float("nan")
        print(f"{DISEASES[task]:<22} {b:>10.3f} {f:>10.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
