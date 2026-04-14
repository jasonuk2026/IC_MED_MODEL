#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_soft_token_classifier import DiseaseEventSoftTokenClassifier
from train_embedding_disease_cond_v2 import BERT_DIM, TASK_2_IDX, EmbeddingStore, build_task_text_embs

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect disease token attention over events for the soft-token classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--eval_data_paths", nargs="+", required=True)
    p.add_argument("--bert_embeddings", required=True)
    p.add_argument("--event_index_path", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--disease_model_name", default="michiyasunaga/BioLinkBERT-base")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--pad_to_num_events", type=int, default=1000)
    p.add_argument("--label", type=int, choices=[0, 1], default=1)
    p.add_argument("--sample_rank", type=int, default=0)
    p.add_argument("--top_k", type=int, default=4)
    return p.parse_args()


def _load_event_index(path: Path) -> dict[int, str]:
    df = pd.read_parquet(path, columns=["event_id", "code", "value", "unit"])
    mapping: dict[int, str] = {}
    for row in df.itertuples(index=False):
        code = "" if row.code is None else str(row.code)
        value = "" if row.value is None else str(row.value)
        unit = "" if row.unit is None else str(row.unit)
        parts = [code.strip()]
        if value.strip():
            parts.append(f"value={value.strip()}")
        if unit.strip():
            parts.append(f"unit={unit.strip()}")
        mapping[int(row.event_id)] = " | ".join(parts)
    return mapping


def _load_code_descriptions(data_dir: Path | None) -> dict[str, str]:
    if data_dir is None:
        return {}
    t2c_path = data_dir / "models" / "clmbr" / "token_2_code.json"
    t2d_path = data_dir / "models" / "clmbr" / "token_2_description.json"
    if not t2c_path.exists() or not t2d_path.exists():
        logger.warning("Description files not found under %s; skipping code descriptions", data_dir)
        return {}
    with open(t2c_path) as f:
        token_2_code = json.load(f)
    with open(t2d_path) as f:
        token_2_desc = json.load(f)
    code_to_desc: dict[str, str] = {}
    for token_id, code in token_2_code.items():
        desc = token_2_desc.get(token_id)
        if desc and desc != code:
            code_to_desc[str(code)] = str(desc)
    return code_to_desc


def _format_event(event_text: str, code_to_desc: dict[str, str]) -> str:
    code = event_text.split(" | ", 1)[0].strip()
    desc = code_to_desc.get(code)
    if desc:
        return f"{desc} [{event_text}]"
    return event_text


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    task_idx = TASK_2_IDX[args.task]
    store = EmbeddingStore(args.bert_embeddings)
    task_text_embs = build_task_text_embs(args, device, rank=0, is_ddp=False)
    model = DiseaseEventSoftTokenClassifier.load_checkpoint(
        Path(args.checkpoint),
        task_text_embs=task_text_embs,
        device=device,
        dtype=dtype,
    )
    model.eval()

    event_index_path = Path(args.event_index_path) if args.event_index_path else Path(args.bert_embeddings).with_name("event_index.parquet")
    event_names = _load_event_index(event_index_path)
    code_to_desc = _load_code_descriptions(Path(args.data_dir) if args.data_dir else None)

    candidates: list[tuple[float, np.ndarray]] = []
    for path in args.eval_data_paths:
        df = pd.read_parquet(path, columns=["task_idx", "label", "event_ids"])
        sub = df[(df["task_idx"] == task_idx) & (df["label"] == args.label)]
        for row in sub.itertuples(index=False):
            eids = np.array(row.event_ids, dtype=np.int32)
            eids = eids[: args.pad_to_num_events]
            event_embs = torch.from_numpy(store.embeddings[eids]).unsqueeze(0).to(device)
            event_mask = torch.ones(1, len(eids), dtype=torch.bool, device=device)
            if len(eids) < args.pad_to_num_events:
                pad_len = args.pad_to_num_events - len(eids)
                pad_embs = torch.zeros(1, pad_len, BERT_DIM, dtype=event_embs.dtype, device=device)
                pad_mask = torch.zeros(1, pad_len, dtype=torch.bool, device=device)
                event_embs = torch.cat([event_embs, pad_embs], dim=1)
                event_mask = torch.cat([event_mask, pad_mask], dim=1)
            logits = model(event_embs, event_mask, torch.tensor([task_idx], device=device))
            prob = torch.sigmoid(logits).item()
            score = prob if args.label == 1 else (1.0 - prob)
            candidates.append((score, eids))

    if not candidates:
        raise ValueError(f"No samples found for task={args.task} label={args.label}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    sample_rank = min(args.sample_rank, len(candidates) - 1)
    score, eids = candidates[sample_rank]

    event_embs = torch.from_numpy(store.embeddings[eids]).unsqueeze(0).to(device)
    event_mask = torch.ones(1, len(eids), dtype=torch.bool, device=device)
    if len(eids) < args.pad_to_num_events:
        pad_len = args.pad_to_num_events - len(eids)
        pad_embs = torch.zeros(1, pad_len, BERT_DIM, dtype=event_embs.dtype, device=device)
        pad_mask = torch.zeros(1, pad_len, dtype=torch.bool, device=device)
        event_embs = torch.cat([event_embs, pad_embs], dim=1)
        event_mask = torch.cat([event_mask, pad_mask], dim=1)

    attn_info = model.get_disease_event_attention(
        event_embs,
        event_mask,
        torch.tensor([task_idx], device=device),
    )
    disease_input = attn_info["disease_input"][0].cpu().numpy()
    disease_hidden = attn_info["disease_hidden"][0].cpu().numpy()
    attn = attn_info["attn_mean"][0, -len(eids) :].cpu().numpy()
    top_k = min(args.top_k, len(eids))
    top_idx = np.argsort(-attn)[:top_k]

    print(f"task={args.task} label={args.label} sample_rank={sample_rank} score={score:.6f} num_events={len(eids)}")
    print("disease_input_embedding=")
    print(np.array2string(disease_input, precision=4, suppress_small=False, max_line_width=160))
    print("disease_hidden_embedding=")
    print(np.array2string(disease_hidden, precision=4, suppress_small=False, max_line_width=160))
    print(f"top_{top_k}_events_by_attention:")
    for rank_idx, event_pos in enumerate(top_idx, start=1):
        event_id = int(eids[event_pos])
        event_name = event_names.get(event_id, f"event_id={event_id}")
        event_text = _format_event(event_name, code_to_desc)
        print(f"{rank_idx}. pos={event_pos} event_id={event_id} attn={attn[event_pos]:.6f} event={event_text}")


if __name__ == "__main__":
    main()
