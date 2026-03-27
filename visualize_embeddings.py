#!/usr/bin/env python3
"""
visualize_embeddings.py

Per-disease KDE plot: 2 rows (base / fine-tuned) × 6 columns (diseases).
Each panel shows the distribution of cosine similarity to the positive centroid,
separately for positive (red) and negative (blue) patients.

If the model works: red curve shifts right relative to blue.
AUC = roc_auc_score(labels, sim_to_pos_centroid) — standard classification AUC.

Usage:
    conda run -n torch python visualize_embeddings.py \
        --checkpoint output/ehrshot_embed/<run_dir>/best \
        --output_png figures/embedding_kde.png
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
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from peft import PeftModel

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

POS_COLOR = "#E53935"
NEG_COLOR = "#1E88E5"
N_PER_CLASS = 200
SEED = 42


# ── Text formatting ───────────────────────────────────────────────────────────

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
    rng = random.Random(seed)
    data_dir = Path(data_dir)
    result = {}
    for task in DISEASES:
        candidates = sorted(data_dir.glob(f"{task}_all_*m500*ntrain50000*.parquet"))
        if not candidates:
            candidates = sorted(data_dir.glob(f"{task}_all_*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet for {task} in {data_dir}")
        df = pd.read_parquet(candidates[0], columns=["task", "split", "label", "events"])
        df = df[df["split"] == "test"].reset_index(drop=True)
        chunks = []
        for label_val in [True, False]:
            sub = df[df["label"] == label_val]
            idx = rng.sample(range(len(sub)), min(n_per_class, len(sub)))
            chunks.append(sub.iloc[idx])
        result[task] = pd.concat(chunks, ignore_index=True)
    return result


# ── Encoding ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_disease(model, tokenizer, df, disease_name, device, batch_size):
    model.eval()
    texts = [build_prompt(disease_name, row.events) for row in df.itertuples()]
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=512,
                        add_special_tokens=False, return_tensors="pt").to(device)
        out = model(**enc)
        mask = enc["attention_mask"]
        seq_len = mask.sum(dim=1) - 1
        bidx = torch.arange(out.last_hidden_state.size(0), device=device)
        emb  = out.last_hidden_state[bidx, seq_len]
        emb  = F.normalize(emb.float(), p=2, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ── KDE panel ─────────────────────────────────────────────────────────────────

def draw_kde_panel(ax, embs: np.ndarray, labels: np.ndarray, title: str):
    """Cosine similarity to positive centroid, KDE for pos vs neg."""
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_centroid = embs[pos_mask].mean(axis=0)
    pos_centroid /= np.linalg.norm(pos_centroid)

    scores = embs @ pos_centroid   # cosine sim (embeddings are L2-normalised)

    x = np.linspace(scores.min() - 0.02, scores.max() + 0.02, 300)

    for mask, color, label in [
        (neg_mask, NEG_COLOR, "Negative"),
        (pos_mask, POS_COLOR, "Positive"),
    ]:
        vals = scores[mask]
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method="scott")
        y   = kde(x)
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.fill_between(x, y, alpha=0.15, color=color)

    try:
        auc = roc_auc_score(labels, scores)
        auc_str = f"AUC = {auc:.3f}"
        auc_color = "#2E7D32" if auc >= 0.6 else ("#E65100" if auc >= 0.52 else "#555")
    except Exception:
        auc_str, auc_color = "", "#555"

    ax.set_facecolor("#F7F8FA")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="white", linewidth=0.8)
    ax.set_title(f"{title}\n{auc_str}", fontsize=9,
                 color=auc_color, fontweight="bold", pad=4)
    ax.set_xlabel("Cosine sim to pos centroid", fontsize=7.5, color="#666")
    ax.set_yticks([])
    ax.tick_params(labelsize=7, colors="#888")

    return float(auc_str.split()[-1]) if auc_str else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--model_name",  default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--data_dir",    default="data/embedding_inputs/new_diagnosis")
    p.add_argument("--output_png",  default="figures/embedding_kde.png")
    p.add_argument("--n_samples",   type=int, default=N_PER_CLASS)
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

    print("Loading test samples ...")
    disease_dfs = load_samples(args.data_dir, args.n_samples, SEED)
    task_list = list(DISEASES.keys())

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=True, padding_side="right"
    )

    all_embs = {}
    for model_tag, load_fn in [
        ("base", lambda: AutoModel.from_pretrained(
            args.model_name, local_files_only=True, torch_dtype=dtype).to(device)),
        ("ft", lambda: PeftModel.from_pretrained(
            AutoModel.from_pretrained(
                args.model_name, local_files_only=True, torch_dtype=dtype),
            args.checkpoint,
        ).to(device)),
    ]:
        print(f"Encoding with {model_tag} model ...")
        model = load_fn()
        model.config.use_cache = False
        for task in tqdm(task_list, desc=f"  [{model_tag}]"):
            df = disease_dfs[task]
            all_embs[(model_tag, task)] = encode_disease(
                model, tokenizer, df,
                TASK_2_DISEASE_NAME[task], device, args.batch_size,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nPlotting ...")
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(task_list), figsize=(len(task_list) * 3.2, 6.5))
    fig.patch.set_facecolor("white")

    aucs = {}
    row_labels = {"base": "Base Model", "ft": "Fine-tuned"}
    for row_idx, model_tag in enumerate(("base", "ft")):
        for col_idx, task in enumerate(task_list):
            ax     = axes[row_idx, col_idx]
            labels = disease_dfs[task]["label"].astype(int).values
            auc    = draw_kde_panel(ax, all_embs[(model_tag, task)], labels,
                                    title=DISEASES[task])
            aucs[(model_tag, task)] = auc
            if col_idx == 0:
                ax.set_ylabel(row_labels[model_tag], fontsize=10,
                              fontweight="bold", color="#333", labelpad=8)

    fig.add_artist(plt.Line2D(
        [0.02, 0.98], [0.505, 0.505],
        transform=fig.transFigure, color="#ccc", linewidth=1,
    ))

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=POS_COLOR, linewidth=2,
               label="Positive (new diagnosis within 1 yr)"),
        Line2D([0], [0], color=NEG_COLOR, linewidth=2, label="Negative"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=10, frameon=True, framealpha=0.95,
               edgecolor="#ddd", bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Cosine Similarity to Positive Centroid  —  Base vs Fine-tuned  (Test Split)\n"
        "Rightward shift of red curve = model assigns higher score to positive patients",
        fontsize=11, y=1.02, color="#222",
    )

    plt.tight_layout(h_pad=2.0, w_pad=0.8)
    plt.savefig(args.output_png, dpi=180, bbox_inches="tight")
    print(f"Saved → {args.output_png}")

    # ── AUC summary ───────────────────────────────────────────────────────────
    print(f"\n{'Disease':<22} {'Base AUC':>10} {'FT AUC':>10} {'Δ':>8}")
    print("-" * 52)
    for task in task_list:
        b = aucs[("base", task)]
        f = aucs[("ft",   task)]
        delta = f - b if not (np.isnan(b) or np.isnan(f)) else float("nan")
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "~")
        print(f"{DISEASES[task]:<22} {b:>10.3f} {f:>10.3f} {delta:>+8.3f} {arrow}")


if __name__ == "__main__":
    main()
