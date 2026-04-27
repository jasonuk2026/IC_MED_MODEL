#!/bin/bash
set -euo pipefail

ckpt_paths=(
  # "stage2_qwen3_0.6b_single_local01/checkpoint-125"
  # "stage2_qwen3_0.6b_single_local01/checkpoint-250"
  # "stage2_qwen3_0.6b_single_local01/checkpoint-1000"
  "stage2_qwen3_0.6b_single_local01/checkpoint-125"
)

for path in "${ckpt_paths[@]}"; do
  # CUDA_VISIBLE_DEVICES=0 python 01_gen_meta/extract_event_emb.py \
  #   --encoder qwen3 \
  #   --model_name Qwen/Qwen3-0.6B \
  #   --model_path "output/$path" \
  #   --tokenizer_name Qwen/Qwen3-0.6B \
  #   --output_dir "data/01_outputs/${path}_padt_lasttok" \
  #   --append_token_name pad_token \
  #   --pooling_mode suffix_only &

  CUDA_VISIBLE_DEVICES=1 python 01_gen_meta/extract_event_emb.py \
    --encoder qwen3 \
    --model_name Qwen/Qwen3-0.6B \
    --model_path "output/$path" \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --output_dir "data/01_outputs/${path}_nopadt_mean" &

  wait

  # bash run_one.sh --model_path test --output_name "${path}_padt_lasttok"
  bash run_one.sh --model_path test --output_name "${path}_nopadt_mean"
done
