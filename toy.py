import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

dist.init_process_group(backend="nccl")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-122B-A10B",
    dtype=torch.bfloat16,
    tp_plan="auto"
)
