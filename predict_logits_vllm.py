"""
Zero-shot disease prediction via next-token logit comparison — vLLM backend.

Replaces the HuggingFace + torchrun pipeline in predict_logits.py with vLLM's
offline LLM.generate(), which provides continuous batching, PagedAttention, and
built-in tensor/expert parallelism without requiring torchrun.

Workflow (two-pass):
  1. Run on val split → finds the best yes/no log-odds threshold (maximises F1)
     and saves it to <output_dir>/<task>/threshold.npy.
  2. Run on test split → loads the val threshold and evaluates on test.

Usage:
    python predict_logits_vllm.py path/to/task.parquet --split val
    python predict_logits_vllm.py path/to/task.parquet --split test

Multi-GPU (tensor-parallel inside vLLM, no torchrun needed):
    python predict_logits_vllm.py path/to/task.parquet --split val  --tensor_parallel_size 4
    python predict_logits_vllm.py path/to/task.parquet --split test --tensor_parallel_size 4
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from jinja2 import Template
from vllm import LLM, SamplingParams
from sklearn.metrics import classification_report, f1_score

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
# Prompt templates (identical to predict_logits.py)
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


# ---------------------------------------------------------------------------
# Tokenizer / token-ID helpers
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
    assert len(yes_ids) == len(no_ids), (
        f"Mismatched yes/no token counts: {len(yes_ids)} vs {len(no_ids)}"
    )
    return list(yes_ids), list(no_ids)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Builds chat-formatted prompt strings from EHRShot dataset items.

    Mirrors PromptCollator from predict_logits.py but returns raw strings
    instead of tokenized tensors — vLLM tokenizes internally.
    """

    def __init__(
        self,
        tokenizer,
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

    def build(self, item: dict) -> str:
        history = self._truncate_history(item["event_history"])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": USER_TEMPLATE.render(
                disease=self.disease, event_history=history,
            )},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_output(output, yes_ids: list[int], no_ids: list[int]) -> float:
    """Extract log-odds score from a single vLLM RequestOutput."""
    lp = output.outputs[0].logprobs[0]  # dict {token_id: Logprob}

    yes_logprobs = [lp[tid].logprob for tid in yes_ids if tid in lp]
    no_logprobs  = [lp[tid].logprob for tid in no_ids  if tid in lp]

    y = torch.tensor(yes_logprobs).logsumexp(0).item() if yes_logprobs else float("-inf")
    n = torch.tensor(no_logprobs).logsumexp(0).item()  if no_logprobs  else float("-inf")
    return y - n


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
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM disease prediction via logits (vLLM backend)."
    )
    parser.add_argument("parquet_path",
                        help="Path to a task parquet file produced by extract_new_diagnosis.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test).")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B",
                        help="HuggingFace model name or local path.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Number of GPUs for tensor parallelism inside vLLM (default: 4).")
    parser.add_argument("--max_num_seqs", type=int, default=16,
                        help="Max sequences vLLM processes concurrently (default: 16).")
    parser.add_argument("--max_event_tokens", type=int, default=3500,
                        help="Max tokens kept from event history (oldest events dropped). "
                             "Does not include prompt prefix/suffix overhead.")
    parser.add_argument("--output_dir", type=str, default="results_priors",
                        help="Directory to save prediction outputs (default: results_priors/).")
    parser.add_argument("--no_expert_parallel", action="store_true",
                        help="Disable expert parallelism (enabled by default for MoE models).")
    return parser.parse_args()


def main():
    args = parse_args()

    stem = Path(args.parquet_path).stem
    task = next((k for k in TASK_2_DISEASE_NAME if stem.startswith(k)), None)
    disease = TASK_2_DISEASE_NAME.get(task, task or stem)

    print(f"Task   : {task or '(unknown)'}  |  Disease: {disease}")
    print(f"Model  : {args.model}")
    print(f"Split  : {args.split}")
    print(f"TP size: {args.tensor_parallel_size}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=not args.no_expert_parallel,
        enable_prefix_caching=True,
        language_model_only=True,
        max_num_seqs=args.max_num_seqs,
        reasoning_parser="qwen3",
    )

    tokenizer = llm.get_tokenizer()

    yes_ids, no_ids = get_yes_no_ids(tokenizer)
    print(f"Yes token IDs : {yes_ids}")
    print(f"No  token IDs : {no_ids}")
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not find single-token yes/no IDs for this tokenizer.")

    all_ids = yes_ids + no_ids

    dataset = EHRShotDataset(args.parquet_path, split=args.split)
    print(f"\nLoaded {len(dataset)} rows for split='{args.split}'")

    prior_knowledge = None  # temporarily disabled
    if prior_knowledge:
        print(f"Prior knowledge : found for task '{task}' ({len(prior_knowledge)} chars)")
    else:
        print(f"Prior knowledge : none (task '{task or stem}' not in TASK_PRIORS)")

    builder = PromptBuilder(tokenizer, disease, args.max_event_tokens, prior_knowledge)

    # Build all prompts and collect labels up front.
    # vLLM performs continuous batching internally — passing all prompts at once
    # allows the scheduler to maximise GPU utilisation.
    print("\nBuilding prompts...")
    prompts, labels = [], []
    for item in tqdm(dataset, desc="Building", dynamic_ncols=True):
        prompts.append(builder.build(item))
        labels.append(int(item["label"]))

    # vLLM caps logprobs at 20 and reports top-k from the pre-masking distribution.
    # allowed_token_ids guarantees the sampled token is yes/no (always included in
    # the logprobs dict per vLLM docs).  logprobs=20 maximises the chance the
    # non-sampled yes/no token also appears in the top-20 pre-masking distribution.
    sp = SamplingParams(
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=all_ids,
    )

    print("\nRunning vLLM inference...")
    outputs = llm.generate(prompts, sp)

    scores_list = [score_output(o, yes_ids, no_ids) for o in outputs]
    n_missing = sum(1 for s in scores_list if not np.isfinite(s))
    if n_missing:
        print(f"WARNING: {n_missing}/{len(scores_list)} samples have non-finite scores "
              f"(yes/no tokens not in top-200 logprobs). Consider increasing logprobs.")
    scores = np.array(scores_list)
    labels = np.array(labels)

    out_dir = Path(args.output_dir) / (task or stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.npz"
    np.savez(out_path, scores=scores, labels=labels)
    print(f"Saved scores to {out_path}")

    thresh_path = out_dir / "threshold.npy"
    if args.split == "val":
        threshold = find_best_threshold(scores, labels)
        np.save(thresh_path, threshold)
        print(f"Best threshold (val): {threshold:+.4f}  →  saved to {thresh_path}")
    else:
        if not thresh_path.exists():
            raise FileNotFoundError(
                f"No threshold found at {thresh_path}. Run with --split val first."
            )
        threshold = float(np.load(thresh_path))
        print(f"Loaded threshold from {thresh_path}: {threshold:+.4f}")

    all_preds = (scores > threshold).astype(int)
    f1 = f1_score(labels, all_preds, zero_division=0)

    print(f"\n{'='*50}")
    print(f"Split     : {args.split}")
    print(f"Task      : {task or stem}")
    print(f"Model     : {args.model}")
    print(f"Threshold : {threshold:+.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"\n{classification_report(labels, all_preds, target_names=['Negative', 'Positive'], zero_division=0)}")


if __name__ == "__main__":
    main()
