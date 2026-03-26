#!/usr/bin/env python3
"""
test_biolinkbert_encode.py

Loads BioLinkBERT-base and runs sanity checks on:
  1. mean_pool_no_special  — correctness unit test (no model needed)
  2. encode_slice          — real forward pass through BioLinkBERT

Usage:
    conda run -n torch python test_biolinkbert_encode.py
    conda run -n torch python test_biolinkbert_encode.py --model_name /path/to/local/BioLinkBERT-base
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from extract_biolinkbert_embeddings import mean_pool_no_special, encode_slice


def test_mean_pool_no_special():
    print("=" * 60)
    print("Test 1: mean_pool_no_special (no model)")
    print("=" * 60)

    B, L, H = 3, 8, 16
    torch.manual_seed(0)
    hs   = torch.randn(B, L, H)
    # Batch:
    #   row 0: [CLS] t1 t2 t3 [SEP] PAD PAD PAD
    #   row 1: [CLS] t4 t5 [SEP] PAD PAD PAD PAD
    #   row 2: [CLS] t6 [SEP] PAD PAD PAD PAD PAD
    ids  = torch.tensor([
        [101, 10, 11, 12, 102,   0,   0,   0],
        [101, 13, 14, 102,   0,   0,   0,   0],
        [101, 15, 102,   0,   0,   0,   0,   0],
    ])
    mask = (ids != 0).long()
    special_ids = {101, 102}

    out = mean_pool_no_special(hs, ids, mask, special_ids)

    # Verify manually
    expected_0 = hs[0, 1:4].mean(0)          # positions 1,2,3
    expected_1 = hs[1, 1:3].mean(0)          # positions 1,2
    expected_2 = hs[2, 1:2].mean(0)          # position 1 only

    assert out.shape == (B, H), f"Wrong shape: {out.shape}"
    assert torch.allclose(out[0], expected_0, atol=1e-10), "Row 0 mismatch"
    assert torch.allclose(out[1], expected_1, atol=1e-10), "Row 1 mismatch"
    assert torch.allclose(out[2], expected_2, atol=1e-10), "Row 2 mismatch"

    print(f"  Output shape : {out.shape}")
    print(f"  Row 0 matches mean of tokens 1-3 : OK")
    print(f"  Row 1 matches mean of tokens 1-2 : OK")
    print(f"  Row 2 matches mean of token  1   : OK")
    print("  PASSED\n")


def test_encode_slice(model_name: str, device: torch.device):
    print("=" * 60)
    print(f"Test 2: encode_slice with BioLinkBERT")
    print(f"  model : {model_name}")
    print(f"  device: {device}")
    print("=" * 60)

    print("  Loading tokenizer and model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    hidden_size  = model.config.hidden_size
    special_ids  = set(tokenizer.all_special_ids)
    print(f"  Hidden size   : {hidden_size}")
    print(f"  Special tokens: {tokenizer.all_special_tokens}  ids={special_ids}")

    # ── Test texts ────────────────────────────────────────────────────────────
    texts = [
        "Heart rate [LOINC/8867-4] | value=72 | unit=/min",
        "Systolic blood pressure [LOINC/8480-6] | value=120 | unit=mmHg",
        "FEMALE [Gender/F]",
        "Metformin [RxNorm/6809]",
        "Type 2 diabetes mellitus [SNOMED/44054006]",
        # Deliberately long text to verify truncation doesn't crash
        "Some very long description " * 30 + "[LOINC/99999]",
    ]

    print(f"\n  Encoding {len(texts)} texts ...")
    embs = encode_slice(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=4,
        max_length=256,
        special_ids=special_ids,
        rank=0,
        desc="test",
    )

    # ── Shape & dtype ─────────────────────────────────────────────────────────
    assert embs.shape == (len(texts), hidden_size), \
        f"Expected ({len(texts)}, {hidden_size}), got {embs.shape}"
    assert embs.dtype == np.float32, f"Expected float32, got {embs.dtype}"
    print(f"\n  Output shape : {embs.shape}")
    print(f"  Output dtype : {embs.dtype}")

    # ── L2 norms (not normalised; just sanity check they are finite & nonzero) ──
    norms = np.linalg.norm(embs, axis=1)
    print(f"\n  L2 norms: {norms}")
    assert np.all(norms > 0) and np.all(np.isfinite(norms)), f"Bad norms: {norms}"
    print("  L2 norms all finite & nonzero : OK")

    # ── Cosine similarities ────────────────────────────────────────────────────
    # Normalise on-the-fly for cosine sim display only
    embs_normed = embs / norms[:, None]
    cos_sim = embs_normed @ embs_normed.T
    print("\n  Cosine similarity matrix (6×6):")
    labels = ["HeartRate", "SystolicBP", "Female", "Metformin", "Diabetes", "LongText"]
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    print(f"  {header}")
    for i, row_label in enumerate(labels):
        row = "  " + f"{row_label:>12}" + "".join(f"{cos_sim[i, j]:>12.4f}" for j in range(len(labels)))
        print(row)

    # Diagonal should be 1.0
    assert np.allclose(np.diag(cos_sim), 1.0, atol=1e-5), "Diagonal not 1.0"
    print("\n  Diagonal ≈ 1.0 : OK")

    # ── Determinism: same text → same embedding ────────────────────────────────
    embs2 = encode_slice(
        texts=texts[:2], model=model, tokenizer=tokenizer, device=device,
        batch_size=2, max_length=256, special_ids=special_ids, rank=0, desc="rerun",
    )
    assert np.allclose(embs[:2], embs2, atol=1e-5), "Non-deterministic output"
    print("  Determinism check (re-run same texts) : OK")

    print("\n  PASSED\n")
    return embs, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="michiyasunaga/BioLinkBERT-base")
    p.add_argument("--device",     default=None,
                   help="cuda / cpu. Defaults to cuda if available.")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    test_mean_pool_no_special()
    embs, labels = test_encode_slice(args.model_name, device)

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
