#!/bin/bash
TIME=$(date +"%Y%m%d%H%M%S")
./run_sm.sh -n 2 -c 16 -m 80G -j train_stage2 \
torchrun --standalone --nproc_per_node=2 train_stage2_with_inline_eval.py \
  --data_path data/verified_cpt_inputs_from_spike/qwen3_0.6b_block2048.parquet \
  --output_dir output/stage2_qwen3_evalwrap_submit_${TIME} \
  --eval_data_dir data/eval_data_latest \
  --eval_every_steps 4000 \
  --early_stop_patience 10 \
  --selection_metric test_macro_auc \
  --wandb_project stage2_qwen3_monitor \
  --wandb_run_name stage2_qwen3_train \
  --model_name Qwen/Qwen3-0.6B \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --encoder qwen3 \
  --epochs 3 \
  --batch_size 4 \
  --lambda_jepa 0.0 \
  --compile \
  --grad_accum 8 \
  --append_token_name pad_token \
  --pooling_mode suffix_only \
  --lr 1e-4 \
  --flash_attn \
  --num_mask_events 4 \
  --eval_wandb_project stage2_qwen3_monitor \
  --eval_wandb_run_name stage2_qwen3_eval


python3 benchmark_foundation_sequence_classifier.py --model_paths output/stage2_qwen3_evalwrap_submit_20260419224900/best --tokenizer_name Qwen/Qwen3-0.6B --encoder qwen3 --eval_data_dir data/eval_data_latest --max_events 100 --batch_size 8