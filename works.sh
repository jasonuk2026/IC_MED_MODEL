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
--train_objective disease_retrieval --train_data_epochs 3 \
--stage1_steps 1000 \
--legacy_stage1_only \
--shallow_encoder_type simple \
--disease_encoder_type query_head


WANDB_API_KEY=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 torchrun --nproc_per_node=4 train_disease_jepa.py \
  --train_data_dir data/llm_data_ixc_patch \
  --train_data_epochs 3 \
  --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --bf16 \
  --batch_size 32 \
  --eval_batch_size 16 \
  --pad_to_num_events 1000 \
  --num_workers 4 \
  --prefetch_factor 8 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.005 \
  --grad_clip 1.0 \
  --objective mse_margin \
  --retrieval_supcon_weight 0. \
  --var_reg_weight 0. \
  --neg_margin 1.5 \
  --neg_margin_weight 0 \
  --shallow_encoder_type simple \
  --shallow_num_layers 3 \
  --predictor_layers 1


WANDB_API_KEY=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 torchrun --nproc_per_node=4 train_disease_concat_classifier.py \
  --train_data_dir data/llm_data_ixc_patch \
  --train_data_epochs 3 \
  --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --bf16 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --pad_to_num_events 1000 \
  --num_workers 4 \
  --prefetch_factor 8 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.005 \
  --grad_clip 1.0 \
  --hidden_size 768 \
  --patient_layers 1 \
  --head_layers 1 \
  --wandb_project medical_cond_embed \
  --wandb_tags concat_classifier

python eval_concat_classifier_retrieval.py \
  --checkpoint output/disease-concat-classifier/20260410_095047/epoch_1 \
  --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --bf16 \
  --eval_batch_size 32 \
  --pad_to_num_events 1000

torchrun --nproc_per_node=4 train_disease_cross_attention_classifier.py \
  --train_data_dir data/llm_data_ixc_patch \
  --train_data_epochs 3 \
  --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --bf16 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --pad_to_num_events 1000 \
  --num_workers 4 \
  --prefetch_factor 8 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.005 \
  --grad_clip 1.0 \
  --hidden_size 768 \
  --patient_layers 1 \
  --head_layers 1 \
  --num_heads 4 \
  --wandb_project medical_cond_embed \
  --wandb_tags cross_attn_classifier

torchrun --nproc_per_node=4 train_disease_concat_classifier.py \
  --train_data_dir data/llm_data_ixc_patch \
  --train_data_epochs 3 \
  --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --bf16 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --pad_to_num_events 1000 \
  --num_workers 4 \
  --prefetch_factor 8 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.005 \
  --grad_clip 1.0 \
  --hidden_size 768 \
  --patient_layers 1 \
  --head_layers 1 \
  --wandb_project medical_cond_embed \
  --wandb_tags concat_classifier

torchrun --nproc_per_node=4 train_disease_soft_token_classifier.py \
  --train_data_dir data/llm_data_ixc_patch \
  --train_data_epochs 3 \
  --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --bf16 \
  --batch_size 32 \
  --eval_batch_size 32 \
  --pad_to_num_events 1000 \
  --num_workers 4 \
  --prefetch_factor 8 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.005 \
  --grad_clip 1.0 \
  --hidden_size 768 \
  --num_layers 1 \
  --num_heads 4 \
  --head_layers 1 \
  --pos_weight 1.0 \
  --wandb_project medical_cond_embed \
  --wandb_tags soft_token_classifier

torchrun --nproc_per_node=4 train_disease_soft_token_classifier.py   --train_data_dir EHRSHOT_ASSETS/llm_data_v7   --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet   --bert_embeddings data/embeddings.npy   --bf16   --batch_size 32   --eval_batch_size 32   --pad_to_num_events 1000   --num_workers 4   --prefetch_factor 8   --lr 2e-4   --warmup_ratio 0.1   --weight_decay 0.005   --grad_clip 1.0   --hidden_size 768   --num_layers 2   --num_heads 4   --head_layers 1   --pos_weight 1.0   --wandb_project medical_cond_embed   --wandb_tags soft_token_classifier --position_type learned --attention_type bidirectional --aux_loss_weight 0.2 --align_loss_weight 0.1