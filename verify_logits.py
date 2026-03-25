import os
import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer




if __name__ == "__main__":
    model_name = "Qwen/Qwen3.5-35B-A3B"
    text = "Who are the fuck are you"

    if "WORLD_SIZE" in os.environ:
        # Multi GPU TP testing
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", tp_plan="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.inference_mode():
        logits = model(**inputs).logits

    max_logit_per_token, _ = logits[0].max(-1)
    print(f"Max logit per token \n{max_logit_per_token}")

    top_1_token = logits[0].argmax(-1)
    print(f"Top 1 token at each position \n{top_1_token}")

    if "WORLD_SIZE" in os.environ:
        dist.destroy_process_group()