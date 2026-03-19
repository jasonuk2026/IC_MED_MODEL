"""
Show the top-N next-token distribution for a given prompt using vLLM.

Usage:
    python vllm_next_token.py --model Qwen/Qwen2.5-7B-Instruct
    python vllm_next_token.py --model Qwen/Qwen3.5-122B-A10B --top_k 30
    python vllm_next_token.py --prompt "The capital of France is"
"""

import argparse
import torch
from vllm import LLM, SamplingParams


def show_top_tokens(llm, prompt: str, top_k: int):
    sp = SamplingParams(
        max_tokens=1,
        logprobs=top_k,
    )

    output = llm.generate([prompt], sp)[0]
    lp = output.outputs[0].logprobs[0]  # {token_id: Logprob}

    # Sort by logprob descending
    ranked = sorted(lp.items(), key=lambda x: x[1].logprob, reverse=True)

    tokenizer = llm.get_tokenizer()

    print(f"\nPrompt: {prompt!r}\n")
    print(f"{'Rank':<6} {'Token':<20} {'LogProb':>10} {'Prob':>10}")
    print("-" * 50)
    for rank, (tid, logprob_obj) in enumerate(ranked[:top_k], 1):
        token_str = repr(tokenizer.decode([tid]))
        lp_val = logprob_obj.logprob
        prob = torch.tensor(lp_val).exp().item()
        print(f"{rank:<6} {token_str:<20} {lp_val:>10.4f} {prob:>10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
        language_model_only=True,
        max_num_seqs=4,
        reasoning_parser="qwen3",
    )

    show_top_tokens(llm, args.prompt, args.top_k)


if __name__ == "__main__":
    main()
