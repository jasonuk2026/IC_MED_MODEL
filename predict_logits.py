"""
Zero-shot disease prediction via next-token logit comparison.

For each patient in an extracted parquet file, builds a prompt containing
the patient's event history and compares the next-token logits for "Yes" vs
"No" to produce a raw log-odds score.  No text is generated.

Workflow (two-pass):
  1. Run on val split → finds the best yes/no log-odds threshold (maximises F1)
     and saves it to <output_dir>/<task>/threshold.npy.
  2. Run on test split → loads the val threshold and evaluates on test.

Single-GPU:
    python predict_logits.py path/to/task.parquet --split val
    python predict_logits.py path/to/task.parquet --split test

Multi-GPU (data-parallel via torchrun):
    torchrun --nproc_per_node=4 predict_logits.py path/to/task.parquet --split val
    torchrun --nproc_per_node=4 predict_logits.py path/to/task.parquet --split test
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import numpy as np
import torch
torch.set_float32_matmul_precision("high")
import torch.distributed as dist

from jinja2 import Template
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from dataset import EHRShotDataset
from task_priors import TASK_PRIORS


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
    """Initialize process group if launched with torchrun.
    Returns (is_distributed, rank, local_rank).
    """
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

def get_yes_no_ids(tokenizer: AutoTokenizer) -> tuple[list[int], list[int]]:
    """Return token IDs (single tokens only) for yes/no variants."""
    yes_ids, no_ids = set(), set()
    for token in (" Yes",):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.add(ids[0])
    for token in (" No",):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.add(ids[0])
    assert len(yes_ids) == len(no_ids), f"Mismatched yes/no token counts: {len(yes_ids)} vs {len(no_ids)}"
    return list(yes_ids), list(no_ids)


# ---------------------------------------------------------------------------
# Collator — runs in DataLoader workers, keeps GPU busy
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = Template("""\
You are a clinical prediction assistant. Based on the patient's medical event history, \
predict whether this patient will be newly diagnosed with {{ disease }} within the next \
year after the observation period ends.
{% if prior_knowledge %}
--- Clinical background on {{ disease }} ---
{{ prior_knowledge }}
--- End of clinical background ---
{% endif %}
Answer with a single word: Yes or No.\
""")

USER_TEMPLATE = Template("""\
Medical history (chronological, oldest first):
{{ event_history }}

Will this patient be newly diagnosed with {{ disease }} within the next year? Answer:\
""")


class PromptCollator:
    """Builds and tokenizes prompts inside DataLoader workers via apply_chat_template.

    Event history is encoded once to check length, truncated to max_event_tokens
    (keeping most recent events), then decoded back to a string for use in the
    chat message dict passed to apply_chat_template.

    If prior_knowledge is provided, it is injected into the system prompt so the
    model can reason about disease-specific clinical signals.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        disease: str,
        max_event_tokens: int,
        prior_knowledge: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_event_tokens = max_event_tokens
        self.disease = disease
        self.system_prompt = SYSTEM_TEMPLATE.render(
            disease=disease,
            prior_knowledge=prior_knowledge,
        )

    def _truncate_history(self, event_history: str) -> str:
        ids = self.tokenizer.encode(event_history, add_special_tokens=False)
        if len(ids) > self.max_event_tokens:
            ids = ids[-self.max_event_tokens:]
            return self.tokenizer.decode(ids)
        return event_history

    def __call__(self, items: list[dict]) -> dict:
        labels = []

        # Build all prompts first (string-only), then batch-tokenize once.
        texts = []
        for item in items:
            history = self._truncate_history(item["event_history"])
            messages = [
                {"role": "system",  "content": self.system_prompt},
                {"role": "user",    "content": USER_TEMPLATE.render(
                    disease=self.disease, event_history=history,
                )},
            ]
            # tokenize=False returns the formatted string, avoiding double tokenization
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            texts.append(text)
            labels.append(int(item["label"]))

        # Single batch tokenization call for all items
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        encoded["labels"] = np.array(labels)
        return encoded


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_batch(
    batch: dict,
    model: AutoModelForCausalLM,
    yes_ids: torch.Tensor,
    no_ids: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, np.ndarray]:
    """Run one batch; return (log-odds scores, labels)."""
    labels = batch.pop("labels")
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

    yes_logit = last_logits.index_select(1, yes_ids).logsumexp(dim=-1)
    no_logit  = last_logits.index_select(1, no_ids).logsumexp(dim=-1)

    scores = yes_logit - no_logit
    return scores, labels


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Sweep unique score values as candidate thresholds; return the one with highest F1."""
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
    parser = argparse.ArgumentParser(description="Zero-shot LLM disease prediction via logits.")
    parser.add_argument("parquet_path",
                        help="Path to a task parquet file produced by extract_new_diagnosis.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test).")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B",
                        help="HuggingFace model name or local path.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-GPU batch size.")
    parser.add_argument("--max_event_tokens", type=int, default=3500,
                        help="Max tokens kept from event history (oldest events dropped). "
                             "Does not include prompt prefix/suffix overhead.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes per rank (default: 4).")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save prediction outputs (default: results/).")
    return parser.parse_args()


def main():
    args = parse_args()
    distributed, rank, local_rank = init_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if is_main(rank):
        mode = f"distributed ({dist.get_world_size()} GPUs)" if distributed else "single GPU"
        print(f"Mode   : {mode}")
        print(f"Device : {device}")

    stem = Path(args.parquet_path).stem
    task = next((k for k in TASK_2_DISEASE_NAME if stem.startswith(k)), None)
    disease = TASK_2_DISEASE_NAME.get(task, task or stem)
    if is_main(rank):
        print(f"Task   : {task or '(unknown)'}  |  Disease: {disease}")

    # Rank 0 downloads the model/tokenizer to the local HF cache first.
    # Other ranks wait at the barrier, then load from cache — no redundant downloads.
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
        dist.barrier()  # release other ranks once rank 0 has finished caching

    yes_ids, no_ids = get_yes_no_ids(tokenizer)
    if is_main(rank):
        print(f"Yes token IDs : {yes_ids}")
        print(f"No  token IDs : {no_ids}")
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not find single-token yes/no IDs for this tokenizer.")
    
    yes_ids = torch.tensor(yes_ids, device=device)
    no_ids  = torch.tensor(no_ids, device=device)

    dataset = EHRShotDataset(args.parquet_path, split=args.split)
    if is_main(rank):
        print(f"\nLoaded {len(dataset)} rows for split='{args.split}'")

    prior_knowledge = TASK_PRIORS.get(task) if task else None
    if is_main(rank):
        if prior_knowledge:
            print(f"Prior knowledge : found for task '{task}' ({len(prior_knowledge)} chars)")
        else:
            print(f"Prior knowledge : none (task '{task or stem}' not in TASK_PRIORS)")

    collator = PromptCollator(tokenizer, disease, args.max_event_tokens, prior_knowledge=prior_knowledge)

    sampler  = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader   = DataLoader(
        dataset,
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
        dist.barrier()  # synchronize before starting inference

    local_scores, local_labels = [], []

    for batch in tqdm(loader, desc=f"Rank {rank}", disable=not is_main(rank), dynamic_ncols=True):
        scores, labels = predict_batch(batch, model, yes_ids, no_ids, device)
        local_scores.append(scores)
        local_labels.append(labels)

    local_scores = torch.cat(local_scores).float().cpu().numpy().tolist()
    local_labels = np.concatenate(local_labels).tolist()

    # Gather results from all ranks onto rank 0
    if distributed:
        all_scores_gathered = [None] * dist.get_world_size()
        all_labels_gathered = [None] * dist.get_world_size()
        dist.all_gather_object(all_scores_gathered, local_scores)
        dist.all_gather_object(all_labels_gathered, local_labels)

        if is_main(rank):
            all_scores = [s for shard in all_scores_gathered for s in shard]
            all_labels = [l for shard in all_labels_gathered for l in shard]
    else:
        all_scores, all_labels = local_scores, local_labels

    if is_main(rank):
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        out_dir = Path(args.output_dir) / (task or stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.split}.npz"
        np.savez(out_path, scores=all_scores, labels=all_labels)
        print(f"Saved scores to {out_path}")

        thresh_path = out_dir / "threshold.npy"
        if args.split == "val":
            threshold = find_best_threshold(all_scores, all_labels)
            np.save(thresh_path, threshold)
            print(f"Best threshold (val): {threshold:+.4f}  →  saved to {thresh_path}")
        else:
            if not thresh_path.exists():
                raise FileNotFoundError(
                    f"No threshold found at {thresh_path}. Run with --split val first."
                )
            threshold = float(np.load(thresh_path))
            print(f"Loaded threshold from {thresh_path}: {threshold:+.4f}")

        all_preds = (all_scores > threshold).astype(int)

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"\n{'='*50}")
        print(f"Split     : {args.split}")
        print(f"Task      : {task or stem}")
        print(f"Model     : {args.model}")
        print(f"Threshold : {threshold:+.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"\n{classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'], zero_division=0)}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
