"""
Zero-shot pairwise disease prediction via next-token logit comparison.

For each patient in a split, we form (positive, negative) patient pairs and ask
the LLM which patient (A or B) is more likely to develop the disease.  The
per-patient score is the average directed log-odds across all pairs they appear in.

Workflow mirrors predict_logits.py:
  1. Run on val split → finds best F1 threshold on per-patient scores and saves
     it to <output_dir>/<task>/threshold.npy.
  2. Run on test split → loads the val threshold and evaluates on test.

Single-GPU:
    python predict_pairwise.py path/to/task.parquet --split val
    python predict_pairwise.py path/to/task.parquet --split test

Multi-GPU (data-parallel via torchrun):
    torchrun --nproc_per_node=2 predict_pairwise.py path/to/task.parquet --split val
    torchrun --nproc_per_node=2 predict_pairwise.py path/to/task.parquet --split test
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import random

import numpy as np
import pyarrow.parquet as pq
import torch
torch.set_float32_matmul_precision("high")
import torch.distributed as dist

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from dataset import EHRShotDataset


TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[bool, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return False, 0, 0
    dist.init_process_group(backend="nccl", device_id=local_rank)
    torch.cuda.set_device(local_rank)
    return True, dist.get_rank(), local_rank


def is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def get_ab_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Return single-token IDs for ' A' and ' B'."""
    a_ids, b_ids = set(), set()
    for token in (" A",):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            a_ids.add(ids[0])
    for token in (" B",):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            b_ids.add(ids[0])
    return list(a_ids), list(b_ids)


# ---------------------------------------------------------------------------
# Pair Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Dataset of (positive, negative) patient pairs.

    For each positive patient, sample `pairs_per_pos` random negative patients.
    Each positive is equally likely to be placed as Patient A or Patient B.

    Exposes:
        patient_labels (np.ndarray): ground-truth label for each patient in
            the base dataset, indexed 0..len(base_dataset)-1.
    """

    def __init__(self, base_dataset: EHRShotDataset, pairs_per_pos: int, seed: int = 42):
        self._base = base_dataset
        n_patients = len(base_dataset)

        # Read labels efficiently from parquet (avoids n_patients __getitem__ calls)
        table = pq.read_table(base_dataset.parquet_path, columns=["label"])
        all_labels = table["label"].to_pylist()
        self.patient_labels = np.array(
            [int(all_labels[abs_i]) for abs_i in base_dataset._indices], dtype=np.int64
        )

        pos_indices = np.where(self.patient_labels == 1)[0].tolist()
        neg_indices = np.where(self.patient_labels == 0)[0].tolist()

        if not pos_indices:
            raise ValueError("No positive examples in this split.")
        if not neg_indices:
            raise ValueError("No negative examples in this split.")

        rng = random.Random(seed)

        # For each positive, sample pairs_per_pos negatives (with replacement)
        pairs: list[tuple[int, int]] = []  # (pos_dataset_idx, neg_dataset_idx)
        for pos_idx in pos_indices:
            for neg_idx in rng.choices(neg_indices, k=pairs_per_pos):
                pairs.append((pos_idx, neg_idx))

        # Randomly shuffle which slot (A or B) each positive occupies
        self._items: list[tuple[int, int]] = []  # (idx_a, idx_b)
        for pos_idx, neg_idx in pairs:
            if rng.random() < 0.5:
                self._items.append((pos_idx, neg_idx))
            else:
                self._items.append((neg_idx, pos_idx))

        self._n_patients = n_patients
        if is_main(int(os.environ.get("RANK", 0))):
            n_pos, n_neg = len(pos_indices), len(neg_indices)
            print(f"Positives: {n_pos}  Negatives: {n_neg}  Pairs: {len(self._items)}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        idx_a, idx_b = self._items[idx]
        item_a = self._base[idx_a]
        item_b = self._base[idx_b]
        return {
            "idx_a":     idx_a,
            "idx_b":     idx_b,
            "history_a": item_a["event_history"],
            "history_b": item_b["event_history"],
            "label_a":   int(item_a["label"]),
            "label_b":   int(item_b["label"]),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a clinical prediction assistant. You are given the medical event "
    "histories of two patients. Determine which patient is more likely to be "
    "newly diagnosed with {disease} within the next year after the observation "
    "period ends. Answer with a single letter: A or B."
)
USER_PROMPT = (
    "Patient A medical history (chronological, oldest first):\n{history_a}\n\n"
    "Patient B medical history (chronological, oldest first):\n{history_b}\n\n"
    "Which patient is more likely to be newly diagnosed with {disease} within "
    "the next year? Answer:"
)


class PairCollator:
    """Builds and tokenizes two-patient prompts inside DataLoader workers."""

    def __init__(self, tokenizer, disease: str, max_event_tokens: int):
        self.tokenizer = tokenizer
        self.max_event_tokens = max_event_tokens
        self.system_prompt = SYSTEM_PROMPT.format(disease=disease)
        self.user_template = USER_PROMPT.format(
            disease=disease, history_a="{history_a}", history_b="{history_b}"
        )

    def _truncate(self, text: str) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_event_tokens:
            ids = ids[-self.max_event_tokens:]
            return self.tokenizer.decode(ids)
        return text

    def __call__(self, items: list[dict]) -> dict:
        texts, idx_a_list, idx_b_list, label_a_list, label_b_list = [], [], [], [], []

        for item in items:
            ha = self._truncate(item["history_a"])
            hb = self._truncate(item["history_b"])
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self.user_template.format(history_a=ha, history_b=hb)},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            texts.append(text)
            idx_a_list.append(item["idx_a"])
            idx_b_list.append(item["idx_b"])
            label_a_list.append(item["label_a"])
            label_b_list.append(item["label_b"])

        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False,
        )
        encoded["idx_a"]   = np.array(idx_a_list)
        encoded["idx_b"]   = np.array(idx_b_list)
        encoded["label_a"] = np.array(label_a_list)
        encoded["label_b"] = np.array(label_b_list)
        return encoded


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_pair_batch(
    batch: dict,
    model,
    a_ids: torch.Tensor,
    b_ids: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one batch; return (log-odds A-vs-B on GPU, idx_a, idx_b, label_a, label_b)."""
    idx_a   = batch.pop("idx_a")
    idx_b   = batch.pop("idx_b")
    label_a = batch.pop("label_a")
    label_b = batch.pop("label_b")
    inputs  = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    outputs     = model(**inputs)
    last_logits = outputs.logits[:, -1, :]          # (batch, vocab_size)

    a_logit = last_logits.index_select(1, a_ids).logsumexp(dim=-1)
    b_logit = last_logits.index_select(1, b_ids).logsumexp(dim=-1)
    scores  = (a_logit - b_logit).float()  # positive → model prefers A; stays on GPU

    return scores, idx_a, idx_b, label_a, label_b


def aggregate_pair_scores(
    pair_scores: list[float],
    idx_a_arr:   list[int],
    idx_b_arr:   list[int],
    n_patients:  int,
) -> np.ndarray:
    """Convert pair-level directed scores into per-patient scores.

    For each pair with log-odds score s (s > 0 means model prefers A):
      - Patient A receives +s  (being preferred over B is good)
      - Patient B receives -s  (being out-ranked by A is bad)

    Final score for patient i = mean of all contributions; 0.0 if never paired.
    """
    score_sum   = np.zeros(n_patients, dtype=np.float64)
    score_count = np.zeros(n_patients, dtype=np.int64)

    for s, ia, ib in zip(pair_scores, idx_a_arr, idx_b_arr):
        score_sum[ia]   += s
        score_count[ia] += 1
        score_sum[ib]   -= s
        score_count[ib] += 1

    safe_count = np.where(score_count > 0, score_count, 1)
    return np.where(score_count > 0, score_sum / safe_count, 0.0).astype(np.float32)


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    best_f1, best_thresh = -1.0, 0.0
    for thresh in np.unique(scores):
        preds = (scores > thresh).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, thresh
    return float(best_thresh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot pairwise LLM disease prediction.")
    parser.add_argument("parquet_path",
                        help="Path to a task parquet file produced by extract_new_diagnosis.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-GPU batch size (pairs, not patients).")
    parser.add_argument("--pairs_per_pos", type=int, default=5,
                        help="Random negatives paired with each positive (default: 5).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for pair sampling.")
    parser.add_argument("--max_event_tokens", type=int, default=14000,
                        help="Max tokens per patient history within a pair. "
                             "Both A and B are independently truncated to this limit.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results_pairwise")
    return parser.parse_args()


def main():
    args = parse_args()
    distributed, rank, local_rank = init_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if is_main(rank):
        mode = f"distributed ({dist.get_world_size()} GPUs)" if distributed else "single GPU"
        print(f"Mode   : {mode}")
        print(f"Device : {device}")

    stem    = Path(args.parquet_path).stem
    task    = next((k for k in TASK_2_DISEASE_NAME if stem.startswith(k)), None)
    disease = TASK_2_DISEASE_NAME.get(task, task or stem)
    if is_main(rank):
        print(f"Task   : {task or '(unknown)'}  |  Disease: {disease}")

    # Rank 0 downloads model first; others wait at a barrier then load from cache.
    if is_main(rank):
        print(f"\nLoading tokenizer and model '{args.model}'...")
    if distributed and not is_main(rank):
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()
    if distributed and not is_main(rank):
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map={"": device},
    )
    model.eval()

    if distributed and is_main(rank):
        dist.barrier()

    a_ids, b_ids = get_ab_ids(tokenizer)
    if is_main(rank):
        print(f"A token IDs : {a_ids}")
        print(f"B token IDs : {b_ids}")
    if not a_ids or not b_ids:
        raise RuntimeError("Could not find single-token A/B IDs for this tokenizer.")

    a_ids = torch.tensor(a_ids, device=device)
    b_ids = torch.tensor(b_ids, device=device)

    # All ranks build the same PairDataset (deterministic seed) so the
    # DistributedSampler gives each rank a disjoint but consistent slice.
    base_dataset = EHRShotDataset(args.parquet_path, split=args.split)
    pair_dataset = PairDataset(base_dataset, args.pairs_per_pos, seed=args.seed)
    n_patients   = len(base_dataset)

    if is_main(rank):
        print(f"Loaded {n_patients} patients for split='{args.split}'")

    collator = PairCollator(tokenizer, disease, args.max_event_tokens)
    sampler  = DistributedSampler(pair_dataset, shuffle=False) if distributed else None
    loader   = DataLoader(
        pair_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    if distributed:
        dist.barrier()

    # ---- inference --------------------------------------------------------
    local_score_chunks: list[torch.Tensor] = []  # GPU tensors; D2H deferred
    local_idx_a_chunks: list[np.ndarray]   = []
    local_idx_b_chunks: list[np.ndarray]   = []

    for batch in tqdm(loader, desc=f"Rank {rank}", disable=not is_main(rank), dynamic_ncols=True):
        scores, idx_a, idx_b, _, _ = predict_pair_batch(batch, model, a_ids, b_ids, device)
        local_score_chunks.append(scores)
        local_idx_a_chunks.append(idx_a)
        local_idx_b_chunks.append(idx_b)

    # Batch conversions after all inference is done
    local_pair_scores: list[float] = torch.cat(local_score_chunks).cpu().numpy().tolist()
    local_idx_a:       list[int]   = np.concatenate(local_idx_a_chunks).tolist()
    local_idx_b:       list[int]   = np.concatenate(local_idx_b_chunks).tolist()

    # ---- gather across ranks ---------------------------------------------
    if distributed:
        gathered_scores = [None] * dist.get_world_size()
        gathered_idx_a  = [None] * dist.get_world_size()
        gathered_idx_b  = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_scores, local_pair_scores)
        dist.all_gather_object(gathered_idx_a,  local_idx_a)
        dist.all_gather_object(gathered_idx_b,  local_idx_b)

        if is_main(rank):
            all_pair_scores = [s for shard in gathered_scores for s in shard]
            all_idx_a       = [s for shard in gathered_idx_a  for s in shard]
            all_idx_b       = [s for shard in gathered_idx_b  for s in shard]
    else:
        all_pair_scores = local_pair_scores
        all_idx_a       = local_idx_a
        all_idx_b       = local_idx_b

    # ---- aggregate + evaluate (rank 0 only) ------------------------------
    if is_main(rank):
        patient_scores = aggregate_pair_scores(all_pair_scores, all_idx_a, all_idx_b, n_patients)
        patient_labels = pair_dataset.patient_labels  # pre-loaded in PairDataset.__init__

        out_dir = Path(args.output_dir) / (task or stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.split}.npz"
        np.savez(out_path, scores=patient_scores, labels=patient_labels)
        print(f"Saved per-patient scores to {out_path}")

        thresh_path = out_dir / "threshold.npy"
        if args.split == "val":
            threshold = find_best_threshold(patient_scores, patient_labels)
            np.save(thresh_path, threshold)
            print(f"Best threshold (val): {threshold:+.4f}  →  saved to {thresh_path}")
        else:
            if not thresh_path.exists():
                raise FileNotFoundError(
                    f"No threshold found at {thresh_path}. Run with --split val first."
                )
            threshold = float(np.load(thresh_path))
            print(f"Loaded threshold from {thresh_path}: {threshold:+.4f}")

        all_preds = (patient_scores > threshold).astype(int)
        f1 = f1_score(patient_labels, all_preds, zero_division=0)
        print(f"\n{'='*50}")
        print(f"Split      : {args.split}")
        print(f"Task       : {task or stem}")
        print(f"Model      : {args.model}")
        print(f"Pairs/pos  : {args.pairs_per_pos}")
        print(f"Threshold  : {threshold:+.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"\n{classification_report(patient_labels, all_preds, target_names=['Negative', 'Positive'], zero_division=0)}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
