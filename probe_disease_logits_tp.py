"""
Zero-shot disease prediction via next-token logit comparison (tensor-parallel only).

Uses PyTorch DTensor tensor parallelism — all GPUs hold a weight shard of every
layer and process the same batches together.  No data-parallel logic.

Padding direction: default (right-padding).  The last valid token position is
inferred from the attention_mask for each sample, so no positional assumptions
are needed.

Workflow (two-pass):
  1. Run on val split → finds the best yes/no log-odds threshold (maximises F1)
     and saves it to <output_dir>/<task>/threshold.npy.
  2. Run on test split → loads the val threshold and evaluates on test.

Launch:
    torchrun --nproc_per_node=N probe_disease_logits_tp.py path/to/task.parquet --split val
    torchrun --nproc_per_node=N probe_disease_logits_tp.py path/to/task.parquet --split test

Note: the model must be pre-cached locally before launching, because all ranks
call from_pretrained simultaneously and concurrent HF Hub downloads may collide.
Pre-fetch with: huggingface-cli download <model>
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import numpy as np
import torch
import torch.distributed as dist

from jinja2 import Template
from pathlib import Path
from torch.utils.data import DataLoader
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
# Tensor-parallel setup
# ---------------------------------------------------------------------------

def init_tp() -> tuple[int, int, int]:
    """Initialize process group for tensor parallelism.
    Returns (rank, local_rank, world_size).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", device_id=local_rank)
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def get_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
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
    assert yes_ids and no_ids, "Could not find single-token yes/no IDs for this tokenizer."
    return list(yes_ids), list(no_ids)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = Template("""\
You are a clinical prediction assistant. Based on the patient's medical event history, \
predict whether this patient will be newly diagnosed with {{ disease }} within the next \
year after the observation period ends.
Answer with a single word: Yes or No.\
""")

USER_TEMPLATE = Template("""\
Medical history (chronological, oldest first):
{{ event_history }}

Will this patient be newly diagnosed with {{ disease }} within the next year? Answer:\
""")


class PromptCollator:
    """Builds and tokenizes prompts with default (right) padding.

    Event history is truncated to max_event_tokens (keeping most recent events)
    before being embedded in the chat template.
    """

    def __init__(self, tokenizer, disease: str, max_event_tokens: int):
        self.tokenizer = tokenizer
        self.max_event_tokens = max_event_tokens
        self.disease = disease
        self.system_prompt = SYSTEM_TEMPLATE.render(disease=disease)

    def _truncate_history(self, event_history: str) -> str:
        ids = self.tokenizer.encode(event_history, add_special_tokens=False)
        if len(ids) > self.max_event_tokens:
            ids = ids[-self.max_event_tokens:]
            return self.tokenizer.decode(ids)
        return event_history

    def __call__(self, items: list[dict]) -> dict:
        texts, labels = [], []
        for item in items:
            history = self._truncate_history(item["event_history"])
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": USER_TEMPLATE.render(
                    disease=self.disease, event_history=history,
                )},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            texts.append(text)
            labels.append(int(item["label"]))

        # Default padding (right side) — last valid position inferred via attention_mask
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
    model,
    yes_ids: torch.Tensor,
    no_ids: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray]:
    """Run one batch; extract last valid token logits via attention_mask.

    With right-padding, the last non-padding token differs per sample.
    attention_mask.sum(dim=-1) - 1 gives the index of the last valid token.
    """
    labels = batch.pop("labels")
    last_positions = batch["attention_mask"].sum(dim=-1) - 1  # (batch,) on CPU

    inputs = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
    outputs = model(**inputs)

    # outputs.logits is (batch, seq_len, vocab); index both dims to get (batch, vocab)
    B = outputs.logits.shape[0]
    last_logits = outputs.logits[torch.arange(B), last_positions.to(outputs.logits.device)]

    yes_logit = last_logits[:, yes_ids].logsumexp(dim=-1)
    no_logit  = last_logits[:, no_ids].logsumexp(dim=-1)

    return yes_logit - no_logit, labels


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
    parser = argparse.ArgumentParser(description="TP-only zero-shot disease prediction via logits.")
    parser.add_argument("parquet_path",
                        help="Path to a task parquet file produced by extract_new_diagnosis.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test).")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B",
                        help="HuggingFace model name or local path.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (all ranks process the same batches).")
    parser.add_argument("--max_event_tokens", type=int, default=3500,
                        help="Max tokens kept from event history (oldest events dropped).")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4).")
    parser.add_argument("--output_dir", type=str, default="results_tp",
                        help="Directory to save prediction outputs (default: results_tp/).")
    parser.add_argument("--tp_plan", type=str, default="auto",
                        choices=["auto", "moe"],
                        help="TP plan: 'auto' for standard models; 'moe' for Qwen MoE models "
                             "(e.g. Qwen3.5-35B-A3B) which need expert-specific sharding and "
                             "replicated k/v projections to avoid reshape errors when "
                             "num_kv_heads < tp_size.")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = init_tp()
    if rank == 0:
        print(f"Mode   : tensor-parallel ({world_size} GPUs)")

    stem = Path(args.parquet_path).stem
    task = next((k for k in TASK_2_DISEASE_NAME if stem.startswith(k)), None)
    disease = TASK_2_DISEASE_NAME.get(task, task or stem)
    if rank == 0:
        print(f"Task   : {task or '(unknown)'}  |  Disease: {disease}")
        print(f"\nLoading tokenizer and model '{args.model}'...")

    # Suppress redundant progress bars on non-main ranks
    if rank != 0:
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        print(f"Pad token is not set, set to eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    # Default padding_side (right) — last valid token located via attention_mask

    if args.tp_plan == "moe":
        # Slightly hardcoded for Qwen 3.5 Moe model
        tp_plan = {
            'model.layers.*.mlp.experts.gate_up_proj': 'packed_colwise', 
            'model.layers.*.mlp.experts.down_proj': 'rowwise', 
            'model.layers.*.mlp.experts': 'moe_tp_experts', 
            'model.layers.*.mlp.shared_expert.gate_proj': 'colwise', 
            'model.layers.*.mlp.shared_expert.up_proj': 'colwise', 
            'model.layers.*.mlp.shared_expert.down_proj': 'rowwise'
        }
    else:
        tp_plan = "auto"

    # All ranks call from_pretrained simultaneously — TP sharding uses
    # distributed ops during weight initialization.  Model must be pre-cached.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        tp_plan=tp_plan,
    )
    model.eval()

    yes_ids, no_ids = get_yes_no_ids(tokenizer)
    if rank == 0:
        print(f"Yes token IDs : {yes_ids}")
        print(f"No  token IDs : {no_ids}")

    yes_ids = torch.tensor(yes_ids, device=model.device)
    no_ids  = torch.tensor(no_ids, device=model.device)

    dataset = EHRShotDataset(args.parquet_path, split=args.split)
    if rank == 0:
        print(f"\nLoaded {len(dataset)} rows for split='{args.split}'")

    collator = PromptCollator(tokenizer, disease, args.max_event_tokens)
    # No DistributedSampler — all ranks process every batch identically
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    dist.barrier()  # synchronize before inference starts

    all_scores, all_labels = [], []
    for batch in tqdm(loader, desc="Inference", disable=rank != 0, dynamic_ncols=True):
        scores, labels = predict_batch(batch, model, yes_ids, no_ids)
        all_scores.append(scores)   # stay on GPU until loop ends
        all_labels.append(labels)

    # Single D2H transfer after all batches
    all_scores = torch.cat(all_scores).float().cpu().numpy()
    all_labels = np.concatenate(all_labels)

    # All ranks produced identical results — only rank 0 saves
    if rank == 0:

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

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
