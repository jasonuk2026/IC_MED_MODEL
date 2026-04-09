#!/bin/bash
# Current known-good training command.
# Status: verified working on the current codebase.
# Recommended setting: disease -> patient retrieval with the shallow `proj` encoder.

WANDB_API_KEY=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 \
torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py \
--train_data_dir data/llm_data_ixc_patch \
--eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
--bf16 \
--wandb_project medical_cond_embed \
--wandb_tags shallow_task_cond \
--bert_embeddings data/embeddings.npy \
--batch_size 32 \
--grad_accum 1 \
--eval_batch_size 32 \
--pad_to_num_events 1000 \
--n_eval_triplets_per_task 1024 \
--flash_attn \
--compile \
--num_workers 4 \
--prefetch_factor 8 \
--epochs 1 \
--warmup_ratio 0.1 \
--stage1_steps 1000 \
--lr_proj 2e-4 \
--lr_lora 5e-5 \
--triplet_margin 0.3 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--weight_decay 0.005 \
--grad_clip 1.0 \
--encoder_mode proj \
--train_objective disease_retrieval --shallow_encoder_type transformer --train_data_epochs 3 \
--stage1_steps 1000 \
--legacy_stage1_only \
--shallow_num_layers 4 --shallow_encoder_type simple