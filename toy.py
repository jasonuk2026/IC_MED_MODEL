"""
Toy script to test tensor-parallel forward pass with dummy inputs.
Tests whether input sequence length affects the TP reshape bug.

Usage:
    # Test with auto tp_plan (likely broken for Qwen3.5 MoE)
    torchrun --nproc_per_node=4 toy.py --tp_plan auto

    # Test with custom tp_plan (k/v proj replicated)
    torchrun --nproc_per_node=4 toy.py --tp_plan custom

    # Sweep multiple sequence lengths
    torchrun --nproc_per_node=4 toy.py --tp_plan auto --seq_lens 64 128 256 512 914
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.distributed.configuration_utils import DistributedConfig

MODEL = "Qwen/Qwen3.5-122B-A10B"

CUSTOM_TP_PLAN = {
    "model.layers.*.self_attn.q_proj": "colwise",
    # k_proj and v_proj omitted → replicated (num_kv_heads=2 < tp_size=4)
    "model.layers.*.self_attn.o_proj": "rowwise",
    "model.layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "model.layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "model.layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "model.layers.*.mlp.experts.down_proj": "rowwise",
    "model.layers.*.mlp.experts": "moe_tp_experts",
    "model.layers.*.mlp.shared_expert.gate_proj": "colwise",
    "model.layers.*.mlp.shared_expert.up_proj": "colwise",
    "model.layers.*.mlp.shared_expert.down_proj": "rowwise",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tp_plan", choices=["auto", "custom"], default="custom")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[64, 256, 512, 914, 1024])
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", device_id=local_rank)
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    if rank == 0:
        print(f"TP world size : {world_size}")
        print(f"tp_plan       : {args.tp_plan}")
        print(f"Model         : {args.model}")

    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    tp_plan = "auto" if args.tp_plan == "auto" else CUSTOM_TP_PLAN

    if rank == 0:
        print("\nLoading model...")
    distributed_config = DistributedConfig(enable_expert_parallel=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        distributed_config=distributed_config,
        # tp_plan=tp_plan,
        # device_mesh=device_mesh,
    )
    model.eval()

    if rank == 0:
        print(f"num_key_value_heads : {model.config.num_key_value_heads}")
        print(f"head_dim            : {model.config.head_dim}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_size = model.config.vocab_size

    for seq_len in args.seq_lens:
        dist.barrier()
        try:
            input_ids = torch.randint(0, vocab_size, (args.batch_size, seq_len), device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            if rank == 0:
                print(f"seq_len={seq_len:6d}  batch={args.batch_size}  →  logits {tuple(outputs.logits.shape)}  OK")
        except Exception as e:
            if rank == 0:
                print(f"seq_len={seq_len:6d}  batch={args.batch_size}  →  FAILED: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
