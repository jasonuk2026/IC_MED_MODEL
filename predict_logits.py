"""
Zero-shot disease prediction via next-token logit comparison.

For each patient in an extracted parquet file, builds a prompt containing
the patient's event history and compares the next-token logits for "Yes" vs
"No" to produce a binary prediction.  No text is generated; the decision is
made entirely from logits.  Computes and prints F1 score.

Single-GPU:
    python predict_logits.py path/to/task.parquet

Multi-GPU (data-parallel via torchrun):
    torchrun --nproc_per_node=4 predict_logits.py path/to/task.parquet
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import numpy as np
import torch
torch.set_float32_matmul_precision("high")
import torch.distributed as dist

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from dataset import EHRShotDataset, ContiguousDistributedSampler


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

SYSTEM_PROMPT = (
    "You are a clinical prediction assistant. Based on the patient's medical "
    "event history, predict whether this patient will be newly diagnosed with "
    "{disease} within the next year after the observation period ends. "
    "Answer with a single word: Yes or No."
)
USER_PROMPT = (
    "Medical history (chronological, oldest first):\n{event_history}\n\n"
    "Will this patient be newly diagnosed with {disease} within the next year? Answer:"
)


class PromptCollator:
    """Builds and tokenizes prompts inside DataLoader workers via apply_chat_template.

    Event history is encoded once to check length, truncated to max_event_tokens
    (keeping most recent events), then decoded back to a string for use in the
    chat message dict passed to apply_chat_template.
    """

    def __init__(self, tokenizer: AutoTokenizer, disease: str, max_event_tokens: int):
        self.tokenizer = tokenizer
        self.max_event_tokens = max_event_tokens
        self.system_prompt = SYSTEM_PROMPT.format(disease=disease)
        self.user_template = USER_PROMPT.format(disease=disease, event_history="{event_history}")

    def _truncate_history(self, event_history: str) -> str:
        ids = self.tokenizer.encode(event_history, add_special_tokens=False)
        if len(ids) > self.max_event_tokens:
            ids = ids[-self.max_event_tokens:]
            return self.tokenizer.decode(ids)
        return event_history

    def __call__(self, items: list[dict]) -> dict:
        all_input_ids = []
        labels = []
        for item in items:
            history = self._truncate_history(item["event_history"])
            messages = [
                {"role": "system",  "content": self.system_prompt},
                {"role": "user",    "content": self.user_template.format(event_history=history)},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                enable_thinking=False, return_dict=False,
            )
            all_input_ids.append(input_ids)
            labels.append(int(item["label"]))

        # Pad to the longest sequence in this batch (respects padding_side="left")
        encoded = self.tokenizer.pad(
            {"input_ids": all_input_ids},
            return_tensors="pt",
            padding=True,
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_prior_bias(
    model: AutoModelForCausalLM,
    collator: PromptCollator,
    yes_ids: list[int],
    no_ids: list[int],
    device: str,
) -> float:
    """Estimate the model's content-free log-odds bias toward Yes vs No.

    Runs a single forward pass with an empty event history and returns
    log P(Yes) - log P(No) on that neutral prompt.  Subtracting this from
    each sample's log-odds calibrates away the model's unconditional preference.
    (Zhao et al., 2021 — "Calibrate Before Use")
    """
    batch = collator([{"event_history": "", "label": 0}])
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    last_logits = model(**inputs).logits[0, -1, :]  # (vocab_size,)
    bias = (last_logits[yes_ids].logsumexp(0) - last_logits[no_ids].logsumexp(0)).item()
    return bias


@torch.inference_mode()
def predict_batch(
    batch: dict,
    model: AutoModelForCausalLM,
    yes_ids: list[int],
    no_ids: list[int],
    device: str,
    prior_bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one batch; return (predictions, labels) as numpy arrays."""
    labels = batch.pop("labels").numpy()
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

    yes_logit = last_logits[:, yes_ids].logsumexp(dim=-1)
    no_logit  = last_logits[:, no_ids].logsumexp(dim=-1)

    # Subtract the content-free bias so the threshold is 0 regardless of the
    # model's unconditional Yes/No preference.
    preds = ((yes_logit - no_logit) > prior_bias).long().cpu().numpy()
    return preds, labels


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

    dataset = EHRShotDataset(args.parquet_path, split=args.split)
    if is_main(rank):
        print(f"\nLoaded {len(dataset)} rows for split='{args.split}'")

    collator = PromptCollator(tokenizer, disease, args.max_event_tokens)

    # Only rank 0 computes the bias; broadcast to all other ranks.
    bias_container = [None]
    if is_main(rank):
        bias_container[0] = compute_prior_bias(model, collator, yes_ids, no_ids, device)
        print(f"Prior bias (Yes - No logit, empty history): {bias_container[0]:+.4f}")
    if distributed:
        dist.broadcast_object_list(bias_container, src=0)
    prior_bias = bias_container[0]

    sampler  = ContiguousDistributedSampler(dataset) if distributed else None
    loader   = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    local_preds, local_labels = [], []

    for batch in tqdm(loader, desc=f"Rank {rank}", disable=not is_main(rank)):
        preds, labels = predict_batch(batch, model, yes_ids, no_ids, device, prior_bias)
        local_preds.extend(preds.tolist())
        local_labels.extend(labels.tolist())

    # Gather results from all ranks onto rank 0
    if distributed:
        all_preds_gathered  = [None] * dist.get_world_size()
        all_labels_gathered = [None] * dist.get_world_size()
        dist.all_gather_object(all_preds_gathered, local_preds)
        dist.all_gather_object(all_labels_gathered, local_labels)

        if is_main(rank):
            all_preds  = [p for shard in all_preds_gathered  for p in shard]
            all_labels = [l for shard in all_labels_gathered for l in shard]
    else:
        all_preds, all_labels = local_preds, local_labels

    if is_main(rank):
        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"\n{'='*50}")
        print(f"Split    : {args.split}")
        print(f"Task     : {task or stem}")
        print(f"Model    : {args.model}")
        print(f"F1 Score : {f1:.4f}")
        print(f"\n{classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'], zero_division=0)}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
