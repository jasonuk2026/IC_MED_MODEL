#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset

from train_embedding_disease_cond_v2 import (
    BERT_DIM,
    TASK_2_IDX,
    EmbeddingStore,
    _collate_event_embs,
    _forward_embeddings,
    build_encoder,
    load_qwen,
)

logger = logging.getLogger("viz_retrieval")


class VizBatchDataset(Dataset):
    def __init__(
        self,
        entries: list[tuple[np.ndarray, int]],
        store: EmbeddingStore,
        batch_size: int,
        pad_to_num_events: int | None = None,
    ):
        self.store = store
        self.pad_to_num_events = pad_to_num_events
        self._batches = [
            entries[i : i + batch_size]
            for i in range(0, len(entries), batch_size)
        ]

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = self._batches[idx]
        eids_list = [e[0] for e in batch]
        task_idxs = [e[1] for e in batch]
        return _collate_event_embs(
            eids_list,
            task_idxs,
            None,
            self.store.embeddings,
            self.pad_to_num_events,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize disease-to-patient retrieval embeddings and score distributions."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_paths", nargs="+", required=True)
    p.add_argument("--bert_embeddings", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder_mode", default="proj")
    p.add_argument("--model_name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--disease_model_name", default="michiyasunaga/BioLinkBERT-base")
    p.add_argument("--shallow_encoder_type", choices=["simple", "mlp", "transformer"], default="transformer")
    p.add_argument("--shallow_num_layers", type=int, default=2)
    p.add_argument("--shallow_num_heads", type=int, default=4)
    p.add_argument("--shallow_intermediate_size", type=int, default=None)
    p.add_argument("--disease_encoder_type", choices=["query_head", "shared_backbone"], default="query_head")
    p.add_argument("--disease_head_layers", type=int, default=0)
    p.add_argument("--disease_head_intermediate_size", type=int, default=None)
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--pad_to_num_events", type=int, default=1000)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--reduction", choices=["auto", "umap", "tsne"], default="auto")
    p.add_argument("--max_points_per_task", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _build_model_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name,
        flash_attn=args.flash_attn,
        qlora=False,
        bf16=args.bf16,
        fp16=args.fp16,
        disease_model_name=args.disease_model_name,
        shallow_encoder_type=args.shallow_encoder_type,
        shallow_num_layers=args.shallow_num_layers,
        shallow_num_heads=args.shallow_num_heads,
        shallow_intermediate_size=args.shallow_intermediate_size,
        disease_encoder_type=args.disease_encoder_type,
        disease_head_layers=args.disease_head_layers,
        disease_head_intermediate_size=args.disease_head_intermediate_size,
    )


def load_encoder_for_viz(args, device: torch.device):
    model_args = _build_model_args(args)
    qwen_model, tokenizer = load_qwen(model_args)
    qwen_model.config.use_cache = False
    encoder = build_encoder(
        qwen_model=qwen_model,
        tokenizer=tokenizer,
        args=model_args,
        device=device,
        rank=0,
        is_ddp=False,
    )
    extra = torch.load(Path(args.checkpoint) / "extra_modules.pt", map_location="cpu")
    saved_disease_encoder_type = extra.get("disease_encoder_type", "query_head")
    if saved_disease_encoder_type != args.disease_encoder_type:
        logger.warning(
            "Checkpoint disease_encoder_type=%s but current args specify %s; visualization may be inconsistent.",
            saved_disease_encoder_type,
            args.disease_encoder_type,
        )
    encoder.bert_proj_1.load_state_dict(extra["bert_proj_1"])
    encoder.bert_proj_2.load_state_dict(extra["bert_proj_2"])
    if "input_norm" in extra:
        encoder.input_norm.load_state_dict(extra["input_norm"])
    if "shallow_layers" in extra:
        encoder.shallow_layers.load_state_dict(extra["shallow_layers"])
    if "task_input_emb" in extra:
        encoder.task_input_emb.load_state_dict(extra["task_input_emb"])
    if "task_input_scale" in extra:
        encoder.task_input_scale.data.copy_(extra["task_input_scale"].to(encoder.task_input_scale.dtype))
    if "task_cond_emb" in extra:
        encoder.task_cond_emb.load_state_dict(extra["task_cond_emb"])
    if "cond_fuse_1" in extra:
        encoder.cond_fuse_1.load_state_dict(extra["cond_fuse_1"])
    if "cond_fuse_2" in extra:
        encoder.cond_fuse_2.load_state_dict(extra["cond_fuse_2"])
    if "task_res_scale" in extra:
        encoder.task_res_scale.data.copy_(extra["task_res_scale"].to(encoder.task_res_scale.dtype))
    if "film_gamma" in extra:
        encoder.film_gamma.load_state_dict(extra["film_gamma"])
    if "film_beta" in extra:
        encoder.film_beta.load_state_dict(extra["film_beta"])
    if "task_query_proj" in extra:
        encoder.task_query_proj.load_state_dict(extra["task_query_proj"])
    if "disease_head_layers" in extra:
        encoder.disease_head_layers.load_state_dict(extra["disease_head_layers"])
    if "task_cross_attn" in extra:
        encoder.task_cross_attn.load_state_dict(extra["task_cross_attn"])
    if "task_xattn_scale" in extra:
        encoder.task_xattn_scale.data.copy_(extra["task_xattn_scale"].to(encoder.task_xattn_scale.dtype))
    if "task_proto_emb" in extra:
        encoder.task_proto_emb.load_state_dict(extra["task_proto_emb"])
    if "query_fuse" in extra:
        encoder.query_fuse.load_state_dict(extra["query_fuse"])
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def load_rows(data_paths: list[str], tasks: set[str] | None):
    idx_to_task = {v: k for k, v in TASK_2_IDX.items()}
    rows = []
    for path in data_paths:
        df = pd.read_parquet(path, columns=["task_idx", "label", "event_ids"])
        for row in df.itertuples(index=False):
            task_name = idx_to_task[int(row.task_idx)]
            if tasks is not None and task_name not in tasks:
                continue
            rows.append(
                {
                    "task_idx": int(row.task_idx),
                    "task": task_name,
                    "label": int(row.label),
                    "event_ids": np.array(row.event_ids, dtype=np.int32),
                }
            )
    return rows


@torch.inference_mode()
def encode_rows(encoder, rows, store, args, device):
    entries = [(r["event_ids"], r["task_idx"]) for r in rows]
    ds = VizBatchDataset(entries, store, args.batch_size, args.pad_to_num_events)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=0)
    all_pre = []
    for batch in dl:
        _, pre_emb = _forward_embeddings(
            encoder,
            batch,
            device,
            args.encoder_mode,
            return_pre_emb=True,
        )
        all_pre.append(pre_emb.cpu())
    pre = torch.cat(all_pre, dim=0)
    all_task_idx = torch.arange(encoder.task_text_embs.size(0), device=device, dtype=torch.long)
    query_bank = encoder.encode_task_query(all_task_idx).float().cpu()
    task_idxs = torch.tensor([r["task_idx"] for r in rows], dtype=torch.long)
    queries = query_bank[task_idxs]
    query_scores = torch.nn.functional.cosine_similarity(queries, pre, dim=1).numpy()
    return pre.numpy(), query_scores


def reduce_to_2d(x: np.ndarray, reduction: str, seed: int) -> tuple[np.ndarray, str]:
    method = reduction
    if reduction in {"auto", "umap"}:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=20,
                min_dist=0.15,
                metric="cosine",
                random_state=seed,
                n_jobs=1,
            )
            return reducer.fit_transform(x), "umap"
        except Exception:
            if reduction == "umap":
                raise
            method = "tsne"

    if method in {"auto", "tsne"}:
        from sklearn.manifold import TSNE

        perplexity = max(5, min(30, x.shape[0] // 10))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        return reducer.fit_transform(x), "tsne"

    raise ValueError(f"Unknown reduction: {reduction}")


def plot_embedding_maps(df: pd.DataFrame, out_dir: Path, seed: int, reduction: str):
    sns.set_theme(style="white", context="talk")
    tasks = sorted(df["task"].unique().tolist())
    ncols = 2
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.6 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    global_vmin = float(df["query_score"].quantile(0.02))
    global_vmax = float(df["query_score"].quantile(0.98))
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9AA0A6", markeredgecolor="none", markersize=6, alpha=0.55, label="Negative"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#111111", markeredgewidth=1.2, markersize=8, label="Positive"),
    ]

    for ax, task in zip(axes, tasks):
        tdf = df[df["task"] == task].copy()
        xy, used_method = reduce_to_2d(tdf[emb_cols].to_numpy(), reduction, seed)
        tdf["x"] = xy[:, 0]
        tdf["y"] = xy[:, 1]

        neg = tdf[tdf["label"] == 0]
        pos = tdf[tdf["label"] == 1]

        ax.scatter(
            neg["x"],
            neg["y"],
            c="#9AA0A6",
            s=16,
            alpha=0.35,
            linewidths=0.0,
            zorder=1,
        )
        sc = ax.scatter(
            pos["x"],
            pos["y"],
            c=pos["query_score"],
            cmap="viridis",
            vmin=global_vmin,
            vmax=global_vmax,
            s=42,
            alpha=0.95,
            linewidths=0.9,
            edgecolors="#111111",
            zorder=3,
        )
        ax.set_title(task.replace("new_", "").replace("_", " "))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{used_method.upper()}-1")
        ax.set_ylabel(f"{used_method.upper()}-2")
        ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=10)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Disease query score")

    for ax in axes[len(tasks):]:
        ax.axis("off")

    fig.suptitle("Disease-conditioned patient embedding maps", fontsize=18)
    fig.savefig(out_dir / "disease_query_embedding_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_relative_embedding_maps(df: pd.DataFrame, out_dir: Path, seed: int, reduction: str):
    sns.set_theme(style="white", context="talk")
    tasks = sorted(df["task"].unique().tolist())
    ncols = 2
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.6 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    global_vmin = float(df["query_score"].quantile(0.02))
    global_vmax = float(df["query_score"].quantile(0.98))
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9AA0A6", markeredgecolor="none", markersize=6, alpha=0.55, label="Negative"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#111111", markeredgewidth=1.2, markersize=8, label="Positive"),
    ]

    rel_cols = [c for c in df.columns if c.startswith("rel_emb_")]
    for ax, task in zip(axes, tasks):
        tdf = df[df["task"] == task].copy()
        xy, used_method = reduce_to_2d(tdf[rel_cols].to_numpy(), reduction, seed)
        tdf["x"] = xy[:, 0]
        tdf["y"] = xy[:, 1]

        neg = tdf[tdf["label"] == 0]
        pos = tdf[tdf["label"] == 1]

        ax.scatter(
            neg["x"],
            neg["y"],
            c="#9AA0A6",
            s=16,
            alpha=0.35,
            linewidths=0.0,
            zorder=1,
        )
        sc = ax.scatter(
            pos["x"],
            pos["y"],
            c=pos["query_score"],
            cmap="viridis",
            vmin=global_vmin,
            vmax=global_vmax,
            s=42,
            alpha=0.95,
            linewidths=0.9,
            edgecolors="#111111",
            zorder=3,
        )
        ax.set_title(task.replace("new_", "").replace("_", " "))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{used_method.upper()}-1")
        ax.set_ylabel(f"{used_method.upper()}-2")
        ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=10)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Disease query score")

    for ax in axes[len(tasks):]:
        ax.axis("off")

    fig.suptitle("Disease-relative patient embedding maps", fontsize=18)
    fig.savefig(out_dir / "disease_relative_embedding_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(df: pd.DataFrame, out_dir: Path):
    sns.set_theme(style="whitegrid", context="talk")
    tasks = sorted(df["task"].unique().tolist())
    ncols = 2
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    palette = {0: "#4C78A8", 1: "#D62728"}
    labels = {0: "Negative", 1: "Positive"}
    for ax, task in zip(axes, tasks):
        tdf = df[df["task"] == task].copy()
        for label in [0, 1]:
            sdf = tdf[tdf["label"] == label]
            if len(sdf) == 0:
                continue
            sns.kdeplot(
                data=sdf,
                x="query_score",
                fill=True,
                common_norm=False,
                alpha=0.35,
                linewidth=1.5,
                color=palette[label],
                label=labels[label],
                ax=ax,
            )
        ax.set_title(task.replace("new_", "").replace("_", " "))
        ax.set_xlabel("Disease query score")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    for ax in axes[len(tasks):]:
        ax.axis("off")

    fig.suptitle("Disease query score distributions by task", fontsize=18)
    fig.savefig(out_dir / "disease_query_score_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distance_distributions(df: pd.DataFrame, out_dir: Path):
    sns.set_theme(style="whitegrid", context="talk")
    tasks = sorted(df["task"].unique().tolist())
    ncols = 2
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    palette = {0: "#4C78A8", 1: "#D62728"}
    labels = {0: "Negative", 1: "Positive"}
    for ax, task in zip(axes, tasks):
        tdf = df[df["task"] == task].copy()
        for label in [0, 1]:
            sdf = tdf[tdf["label"] == label]
            if len(sdf) == 0:
                continue
            sns.kdeplot(
                data=sdf,
                x="query_distance",
                fill=True,
                common_norm=False,
                alpha=0.35,
                linewidth=1.5,
                color=palette[label],
                label=labels[label],
                ax=ax,
            )
        ax.set_title(task.replace("new_", "").replace("_", " "))
        ax.set_xlabel("Distance to disease query (1 - cosine)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    for ax in axes[len(tasks):]:
        ax.axis("off")

    fig.suptitle("Disease query distance distributions by task", fontsize=18)
    fig.savefig(out_dir / "disease_query_distance_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    _setup_logging()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint from {args.checkpoint}")

    tasks = set(args.tasks) if args.tasks else None
    rows = load_rows(args.data_paths, tasks)
    logger.info(f"Loaded {len(rows)} patient rows from {len(args.data_paths)} parquet files")
    if not rows:
        raise ValueError("No rows loaded for visualization.")

    if args.max_points_per_task > 0:
        rng = np.random.default_rng(args.seed)
        keep_rows = []
        by_task: dict[str, list[dict]] = {}
        for r in rows:
            by_task.setdefault(r["task"], []).append(r)
        for task, task_rows in by_task.items():
            if len(task_rows) > args.max_points_per_task:
                idx = rng.choice(len(task_rows), size=args.max_points_per_task, replace=False)
                keep_rows.extend([task_rows[i] for i in sorted(idx.tolist())])
            else:
                keep_rows.extend(task_rows)
        rows = keep_rows
        logger.info(f"Retained {len(rows)} rows after per-task subsampling")

    store = EmbeddingStore(args.bert_embeddings)
    encoder = load_encoder_for_viz(args, device)
    patient_pre, query_scores = encode_rows(encoder, rows, store, args, device)
    all_task_idx = torch.arange(encoder.task_text_embs.size(0), device=device, dtype=torch.long)
    query_bank = encoder.encode_task_query(all_task_idx).float().detach().cpu().numpy()
    query_vecs = query_bank[np.array([r["task_idx"] for r in rows], dtype=np.int64)]
    rel_pre = patient_pre - query_vecs

    emb_cols = {f"emb_{i}": patient_pre[:, i] for i in range(patient_pre.shape[1])}
    rel_emb_cols = {f"rel_emb_{i}": rel_pre[:, i] for i in range(rel_pre.shape[1])}
    df = pd.DataFrame(
        {
            "task": [r["task"] for r in rows],
            "task_idx": [r["task_idx"] for r in rows],
            "label": [r["label"] for r in rows],
            "query_score": query_scores,
            "query_distance": 1.0 - query_scores,
            **emb_cols,
            **rel_emb_cols,
        }
    )

    counts = df.groupby("task")["label"].agg(["count", "sum"])
    for task, row in counts.iterrows():
        pos = int(row["sum"])
        total = int(row["count"])
        logger.info(f"[{task}] n={total} pos={pos} neg={total - pos}")

    plot_embedding_maps(df, out_dir, args.seed, args.reduction)
    plot_relative_embedding_maps(df, out_dir, args.seed, args.reduction)
    plot_score_distributions(df, out_dir)
    plot_distance_distributions(df, out_dir)
    df.to_csv(out_dir / "visualization_scores.csv", index=False)
    logger.info(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
