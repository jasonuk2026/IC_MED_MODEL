"""
Test yes/no logit scoring with vLLM offline mode using dummy inputs.

Usage:
    python test_yesno_vllm.py --model Qwen/Qwen3.5-122B-A10B
    python test_yesno_vllm.py --model Qwen/Qwen2.5-7B-Instruct  # quick local test
"""

import argparse
import torch
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Dummy yes/no prompts
# ---------------------------------------------------------------------------
DUMMY_PROMPTS = [
    ("Is the sky blue?",                    "Yes"),
    ("Is water dry?",                       "No"),
    ("Is 2 + 2 = 4?",                       "Yes"),
    ("Is the moon made of cheese?",         "No"),
    ("Is Python a programming language?",   "Yes"),
    ("Is the sun a planet?",                "No"),
]


def get_yes_no_ids(tokenizer):
    yes_surfaces = ["Yes", "yes", " Yes", " yes", "YES"]
    no_surfaces  = ["No",  "no",  " No",  " no",  "NO"]

    def encode_single(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    yes_ids, no_ids = [], []

    print("\n=== Token ID mapping ===")
    for s in yes_surfaces:
        tid = encode_single(s)
        if tid is not None:
            yes_ids.append(tid)
            print(f"  Yes  {s!r:8s} -> id {tid}")
        else:
            print(f"  Yes  {s!r:8s} -> MULTI-TOKEN, skipped")
    for s in no_surfaces:
        tid = encode_single(s)
        if tid is not None:
            no_ids.append(tid)
            print(f"  No   {s!r:8s} -> id {tid}")
        else:
            print(f"  No   {s!r:8s} -> MULTI-TOKEN, skipped")

    yes_ids = list(dict.fromkeys(yes_ids))
    no_ids  = list(dict.fromkeys(no_ids))
    print(f"\n  Final yes_ids : {yes_ids}")
    print(f"  Final no_ids  : {no_ids}")
    return yes_ids, no_ids


def score_with_logits_processor(llm, prompts, expected_answers, yes_ids, no_ids):
    print("\n=== allowed_token_ids ===")

    all_ids = yes_ids.tolist() + no_ids.tolist()

    sp = SamplingParams(
        max_tokens=1,
        logprobs=len(all_ids),
        allowed_token_ids=all_ids,  # 只保留 yes/no token 的 logits
    )

    outputs = llm.generate(prompts, sp)

    print()
    for i, (output, expected) in enumerate(zip(outputs, expected_answers)):
        lp = output.outputs[0].logprobs[0]

        yes_scores = [lp[tid].logprob for tid in yes_ids.tolist() if tid in lp]
        no_scores  = [lp[tid].logprob for tid in no_ids.tolist()  if tid in lp]

        y = torch.tensor(yes_scores).logsumexp(0).item() if yes_scores else float("-inf")
        n = torch.tensor(no_scores).logsumexp(0).item()  if no_scores  else float("-inf")

        pred    = "Yes" if y > n else "No"
        correct = "✓" if pred == expected else "✗"
        print(f"  [{correct}] {prompts[i][:55]!r}")
        print(f"       yes={y:.4f}  no={n:.4f}  pred={pred}  expected={expected}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    prompts = []
    expected_answers = []
    for question, answer in DUMMY_PROMPTS:
        prompts.append(
            f"Answer the following yes/no question with only Yes or No.\n"
            f"Question: {question}\nAnswer:"
        )
        expected_answers.append(answer)

    print("=== Loading model ===")
    print(f"  model                 : {args.model}")
    print(f"  data_parallel_size    : 4")
    print(f"  enable_expert_parallel: True")
    print(f"  enable_prefix_caching : True")
    print(f"  language_model_only   : True  (enable_thinking=False)")
    print(f"  reasoning_parser      : qwen3")

    llm = LLM(
        model=args.model,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
        language_model_only=True,
        tensor_parallel_size=4,
        max_num_seqs=4,
        # --reasoning-parser qwen3
        reasoning_parser="qwen3",
    )

    tokenizer = llm.get_tokenizer()
    yes_ids, no_ids = get_yes_no_ids(tokenizer)

    yes_ids_t = torch.tensor(yes_ids)
    no_ids_t  = torch.tensor(no_ids)

    score_with_logits_processor(llm, prompts, expected_answers, yes_ids_t, no_ids_t)


if __name__ == "__main__":
    main()
