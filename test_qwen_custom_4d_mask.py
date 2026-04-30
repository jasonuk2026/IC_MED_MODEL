#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_warning()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def build_custom_4d_mask(
    input_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    event_ids: torch.Tensor,
    eos_token_id: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    valid = attention_mask_2d.bool()
    pos = torch.arange(seq_len, device=device)
    causal = pos.view(1, 1, seq_len) <= pos.view(1, seq_len, 1)
    same_event = event_ids[:, :, None] == event_ids[:, None, :]
    eos_keys = ((input_ids == eos_token_id) & valid)[:, None, :]
    q_valid = valid[:, :, None]
    k_valid = valid[:, None, :]
    allowed = ((same_event & causal) | (eos_keys & causal)) & q_valid & k_valid

    eye = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
    allowed = allowed | ((~valid)[:, :, None] & eye)

    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return mask


def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal test for Qwen3 with custom 4D attention mask through model(**batch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)

    eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError(f"Tokenizer {args.model_name} has no pad_token_id; expected Qwen-style EOT/pad token.")
    logger.info("Using eos/pad token: %r (id=%d)", tokenizer.pad_token, eos_token_id)

    torch_dtype = torch.float32
    autocast_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
        autocast_dtype = torch.float16

    logger.info("Loading model: %s | attn=%s", args.model_name, args.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    # Synthetic example:
    # t0, t1, e0, t2, t3, e1, t4
    # event ids:
    #  0   0   0   1   1   1   2
    # where e0/e1 are eos tokens.
    base_ids = [100, 101, eos_token_id, 102, 103, eos_token_id, 104]
    event_ids_list = [0, 0, 0, 1, 1, 1, 2]
    attention_mask_list = [1] * len(base_ids)

    input_ids = torch.tensor([base_ids], dtype=torch.long, device=device)
    event_ids = torch.tensor([event_ids_list], dtype=torch.long, device=device)
    attention_mask_2d = torch.tensor([attention_mask_list], dtype=torch.long, device=device)

    attn_mask_4d = build_custom_4d_mask(
        input_ids=input_ids,
        attention_mask_2d=attention_mask_2d,
        event_ids=event_ids,
        eos_token_id=eos_token_id,
        dtype=next(model.parameters()).dtype,
    )

    allowed = (attn_mask_4d[0, 0] == 0).to(torch.int64).cpu().tolist()
    logger.info("Synthetic input_ids=%s", base_ids)
    logger.info("Synthetic event_ids=%s", event_ids_list)
    logger.info("Allowed attention matrix=%s", json.dumps(allowed))

    autocast_enabled = torch.cuda.is_available() and autocast_dtype is not None
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask_4d,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    logger.info("Forward succeeded.")
    logger.info("logits_shape=%s", tuple(logits.shape))
    logger.info("toy_next_token_loss=%.6f", float(loss.item()))


if __name__ == "__main__":
    main()
