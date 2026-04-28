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

torchrun --nproc_per_node=4 train_disease_soft_token_classifier.py   --train_data_dir data/02_collect_train_data_results   --eval_data_paths data/eval_data_latest/*/val.parquet   --bert_embeddings data/01_outputs/01_outputs_biolinkbert_embeddings/embeddings.npy   --bf16   --batch_size 32   --eval_batch_size 32   --pad_to_num_events 1000   --num_workers 4   --prefetch_factor 8   --lr 2e-4   --warmup_ratio 0.1   --weight_decay 0.005   --grad_clip 1.0   --hidden_size 768   --num_layers 2   --num_heads 4   --head_layers 1   --pos_weight 1.0   --wandb_project medical_cond_embed   --wandb_tags soft_token_classifier --position_type learned --attention_type bidirectional --aux_loss_weight 0.2 --align_loss_weight 0.1


torchrun --nproc_per_node=4 train_disease_soft_token_classifier.py   --train_data_dir data/llm_data_ixc_patch   --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet   --bert_embeddings ./embeddings.npy   --bf16   --batch_size 32   --eval_batch_size 32   --pad_to_num_events 1000   --num_workers 4   --prefetch_factor 8   --lr 2e-4   --warmup_ratio 0.1   --weight_decay 0.005   --grad_clip 1.0   --hidden_size 768   --num_layers 2   --num_heads 4   --head_layers 1   --pos_weight 1.0   --wandb_project medical_cond_embed   --wandb_tags soft_token_classifier --position_type learned --attention_type bidirectional --aux_loss_weight 0.2 --align_loss_weight 0.1


conda run -n torch python plot_classifier_score_distributions.py \
  --checkpoint output/disease-soft-token-classifier/20260410_132749/epoch_1 \
  --model_type soft_token \
  --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --bf16 \
  --eval_batch_size 32 \
  --pad_to_num_events 1000 \
  --output_dir figures/classifier_score_distributions

conda run -n torch python inspect_soft_token_attention.py \
  --checkpoint output/disease-soft-token-classifier/20260410_120142/epoch_1 \
  --task new_pancan \
  --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --bf16 \
  --data_dir EHRSHOT_ASSETS \
  --pad_to_num_events 1000 \
  --label 1 \
  --sample_rank 0 \
  --top_k 4

train_disease_soft_token_classifier.py --train_data_dir llm_data_ixc_patch --eval_data_paths llm_eval_data_ixc/new_*/val.parquet --bert_embeddings maybe_best_retrieval/embeddings.npy --bf16 --batch_size 32 --eval_batch_size 32 --pad_to_num_events 1000 --num_workers 4 --prefetch_factor 8 --lr 2e-4 --warmup_ratio 0.1 --weight_decay 0.005 --grad_clip 1.0 --hidden_size 768 --num_layers 2 --num_heads 4 --head_layers 1 --pos_weight 1.0 --wandb_project medical_cond_embed --wandb_tags soft_token_classifier --position_type learned --attention_type bidirectional --aux_loss_weight 0.2 --align_loss_weight 0.1 --train_data_epochs 0 1 2 3
torchrun --standalone --nproc_per_node=2 train_disease_soft_token_classifier.py --train_data_dir data/02_outputs --eval_data_paths data/02_outputs/new_*/val.parquet --bert_embeddings data/01_outputs/01_outputs_biolinkbert_embeddings/embeddings.npy --bf16 --batch_size 32 --eval_batch_size 32 --pad_to_num_events 1000 --num_workers 4 --prefetch_factor 8 --lr 2e-4 --warmup_ratio 0.1 --weight_decay 0.005 --grad_clip 1.0 --hidden_size 768 --num_layers 2 --num_heads 4 --head_layers 1 --pos_weight 1.0 --wandb_project medical_cond_embed --wandb_tags soft_token_classifier --position_type learned --attention_type bidirectional --aux_loss_weight 0.2 --align_loss_weight 0.1 --train_data_epochs 0 --disease_model_name michiyasunaga/BioLinkBERT-base


torchrun --standalone --nproc_per_node=2 train_disease_soft_token_classifier.py --train_data_dir data/02_outputs --eval_data_paths data/02_outputs/new_*/val.parquet --bert_embeddings data/01_outputs/qwen3_0.6b_embs/embeddings.npy --bf16 --batch_size 32 --eval_batch_size 32 --pad_to_num_events 1000 --num_workers 4 --prefetch_factor 8 --lr 2e-4 --warmup_ratio 0.1 --weight_decay 0.005 --grad_clip 1.0 --hidden_size 768 --num_layers 2 --num_heads 4 --head_layers 1 --pos_weight 1.0 --wandb_project medical_cond_embed --wandb_tags soft_token_classifier --position_type learned --attention_type bidirectional --aux_loss_weight 0.2 --align_loss_weight 0.1 --disease_model_name Qwen/Qwen3-0.6B --train_data_epochs 0

python 01_gen_meta/extract_event_emb.py \
  --unique_events_path data/01_outputs/unique_events.parquet \
  --encoder biolinkbert \
  --output_dir data/01_outputs_biolinkbert_embeddings \
  --preview_tokenization_n 5

python 01_gen_meta/extract_event_emb.py --model_name Qwen/Qwen3-0.6B --encoder qwen3 --bf16 --output_dir data/01_outputs/qwen3_0.6b_uncpt_direct_mepa_embs --model_path data/qwen3_0.6b_uncpt_direct_mepa --tokenizer_name Qwen/Qwen3-0.6B

torchrun --standalone --nproc_per_node=2 train_disease_soft_token_classifier.py --train_data_dir data/02_outputs --eval_data_paths data/02_outputs/new_*/val.parquet --bert_embeddings data/01_outputs/qwen3_0.6b_uncpt_direct_mepa_embs/embeddings.npy --bf16 --batch_size 32 --eval_batch_size 32 --pad_to_num_events 1000 --num_workers 4 --prefetch_factor 8 --lr 2e-4 --warmup_ratio 0.1 --weight_decay 0.005 --grad_clip 1.0 --hidden_size 768 --num_layers 2 --num_heads 4 --head_layers 1 --pos_weight 1.0 --wandb_project medical_cond_embed --wandb_tags soft_token_classifier --position_type learned --attention_type bidirectional --aux_loss_weight 0.2 --align_loss_weight 0.1 --disease_model_name Qwen/Qwen3-0.6B --disease_model_path data/qwen3_0.6b_uncpt_direct_mepa --disease_tokenizer_name Qwen/Qwen3-0.6B --train_data_epochs 0

python 01_gen_meta/extract_event_emb.py --model_name Qwen/Qwen3-0.6B --encoder qwen3 --bf16 --output_dir data/01_outputs/qwen3_0.6b_cpt_1000_embs --model_path data/cpt_outputs/Qwen3-0.6B/checkpoint-1000 --tokenizer_name Qwen/Qwen3-0.6B

torchrun --standalone --nproc_per_node=2 train_disease_soft_token_classifier.py --train_data_dir data/02_outputs --eval_data_paths data/02_outputs/new_*/val.parquet --bert_embeddings data/01_outputs/qwen3_0.6b_uncpt_direct_mepa_embs/embeddings.npy --bf16 --batch_size 64 --eval_batch_size 32 --pad_to_num_events 1000 --num_workers 4 --prefetch_factor 8 --lr 2e-4 --warmup_ratio 0.1 --weight_decay 0.005 --grad_clip 1.0 --hidden_size 768 --num_layers 2 --num_heads 4 --head_layers 1 --pos_weight 1.0 --wandb_project medical_cond_embed --wandb_tags soft_token_classifier --position_type learned --attention_type bidirectional --aux_loss_weight 0.2 --align_loss_weight 0.1 --disease_model_name Qwen/Qwen3-0.6B --train_data_epochs 0

python 01_gen_meta/extract_event_emb.py \
  --encoder qwen3 \
  --model_name Qwen/Qwen3-0.6B \
  --model_path data/qwen3_0.6b_uncpt_direct_mepa \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --append_token_text "<EVENT_END>" \
  --output_dir data/01_outputs/qwen_with_suffix

CUDA_VISIBLE_DEVICES=1 python 01_gen_meta/extract_event_emb.py \
  --encoder qwen3 \
  --model_name Qwen/Qwen3-0.6B \
  --model_path output/stage2_qwen3_0.6b_single_local01/checkpoint-250 \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --output_dir data/01_outputs/qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_250_lambdared_0.2_mlpr_1 --append_token_name pad_token --pooling_mode suffix_only

CUDA_VISIBLE_DEVICES=0 python 01_gen_meta/extract_event_emb.py \
  --encoder qwen3 \
  --model_name Qwen/Qwen3-0.6B \
  --model_path output/stage2_qwen3_evalwrap_test04/best \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --append_token_name pad_token \
  --pooling_mode suffix_only \
  --output_dir data/01_outputs/run_temp_test/qwen --bf16 --attn_implementation flash_attention_2 && python3 benchmark_foundation_simple_classifier.py \
  --embedding_dirs run_temp_test/qwen \
  --eval_data_dir data/eval_data_latest \
  --train_split val \
  --test_split test \
  --max_events 1000 \
  --truncate_side first



torchrun --nproc_per_node=2 01_gen_meta/extract_event_emb.py \
  --encoder qwen3 \
  --model_name Qwen/Qwen3-0.6B \
  --model_path data/cpt_outputs/Qwen3-0.6B/final \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --output_dir data/01_outputs/Qwen3-0.6B/final_cpt_nopad_mean_ddp

bash run_one.sh --model_path test --output_name qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_125_lambdared_0.2_mlpr_1 && bash run_one.sh --model_path test --output_name qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_250_lambdared_0.2_mlpr_1

torchrun --standalone --nproc_per_node=2 train_stage2.py \
  --data_path data/verified_cpt_inputs_from_spike/qwen3_0.6b_block2048.parquet \
  --model_name Qwen/Qwen3-0.6B \
  --output_dir output/stage2_qwen3_0.6b_single_local01 \
  --batch_size 2 \
  --grad_accum 8 --num_mask_events 4 --flash_attn --compile --save_at_steps 125 250 1000 \
  --ema_decay 0.996 --lambda_red 0 --lambda_jepa 0

python3 benchmark_foundation_simple_classifier.py \
  --embedding_dirs \
    qwen3_0.6b_embs \
    qwen3_0.6b_uncpt_direct_mepa_embs \
    qwen3_0.6b_uncpt_2nd_mepa_last_tokemb \
    qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_1000_lambdared_0.2_mlpr_1 \
    qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_125_lambdared_0.2_mlpr_1 \
    qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_250_lambdared_0.2_mlpr_1 \
    qwen3_0.6b_uncpt_direct_mepa_embs_with_pad_token_last_tokemb \
    qwen3_0.6b_uncpt_direct_mepa_wo_red_embs_wo_pad_token_avg \
    stage2_qwen3_0.6b_single_local01/checkpoint-125_padt_lasttok_lambda_jepa0_red0 \
  --eval_data_dir data/eval_data_latest \
  --max_events 1000 \
  --truncate_side last

python3 benchmark_foundation_simple_classifier.py \
  --embedding_dirs \
    qwen3_0.6b_embs \
    qwen3_0.6b_uncpt_direct_mepa_embs \
    qwen3_0.6b_uncpt_2nd_mepa_last_tokemb \
    qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_1000_lambdared_0.2_mlpr_1 \
    qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_125_lambdared_0.2_mlpr_1 \
    qwen3_0.6b_uncpt_2nd_mepa_last_tokemb_local_250_lambdared_0.2_mlpr_1 \
    qwen3_0.6b_uncpt_direct_mepa_embs_with_pad_token_last_tokemb \
    qwen3_0.6b_uncpt_direct_mepa_wo_red_embs_wo_pad_token_avg \
    stage2_qwen3_0.6b_single_local01/checkpoint-125_padt_lasttok_lambda_jepa0_red0 \
    Qwen3-0.6B/final_cpt_nopad_mean \
    Qwen3-0.6B/final_cpt_pad_suffix \
  --eval_data_dir data/eval_data_latest \
  --max_events 2000 \
  --truncate_side last


python3 benchmark_foundation_sequence_classifier.py \
  --model_paths output/stage2_qwen3_evalwrap_submit04/best \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --encoder qwen3 \
  --eval_data_dir data/eval_data_latest --max_events 1000 --batch_size 4
  
python3 benchmark_foundation_sequence_classifier.py \
  --model_paths output/stage2_qwen3_evalwrap_local01/best \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --encoder qwen3 \
  --eval_data_dir data/eval_data_latest --max_events 1000 --batch_size 2

python3 benchmark_foundation_sequence_classifier.py \
  --model_paths output/stage2_qwen3_evalwrap_submit04/best \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --encoder qwen3 \
  --append_token_name pad_token \
  --pooling_mode all_suffix_mean \
  --eval_data_dir data/eval_data_latest --max_events 1000 --batch_size 2


python 01_gen_meta/build_next_event_train_parquet.py \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --data_dir EHRSHOT_ASSETS \
  --unique_event_parquet hx1/unique_events.parquet \
  --output_path hx1/qwen3_0.6b_patient_events.parquet \
  --events_per_row 1024 \
  --no_append_eos_per_event

CUDA_VISIBLE_DEVICES=0 python train_next_event_cosine.py \
  --data_path hx1/qwen3_0.6b_patient_events.parquet \
  --model_name Qwen/Qwen3-0.6B \
  --output_dir hx1/next_event_cosine_single_gpu \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --max_events 1024 \
  --max_event_tokens 128 \
  --sequence_truncate_side last \
  --event_truncate_side last \
  --freeze_event_encoder \
  --predictor_hidden_size 128 \
  --predictor_num_heads 4 \
  --predictor_num_layers 1 \
  --predictor_ffn_dim 128 \
  --predictor_dropout 0.0 \
  --event_encoder_batch_size 512 \
  --num_workers 4 \
  --bf16 \
  --flash_attn \
  --compile \
  --log_steps 20 \
  --num_checkpoints 10 \
  --wandb_project mepa_trial

git annex initremote jotta type=rclone rcloneremotename=jotta rcloneprefix=annex  encryption=none chunk=500MiB  --whatelse

git annex initremote jotta \
  type=rclone \
  rcloneremotename=jotta \
  rcloneprefix=git-annex encryption=none               

git annex add EHRSHOT_ASSETS/splits/person_id_map.csv EHRSHOT_ASSETS/benchmark/*/labeled_patients.csv EHRSHOT_ASSETS/models/clmbr/token_2_code.json EHRSHOT_ASSETS/models/clmbr/token_2_description.json

git add EHRSHOT_ASSETS/splits/person_id_map.csv EHRSHOT_ASSETS/benchmark/*/labeled_patients.csv EHRSHOT_ASSETS/models/clmbr/token_2_code.json EHRSHOT_ASSETS/models/clmbr/token_2_description.json -f

python benchmark_next_event_sequence_classifier.py \
--checkpoint_paths hx1/next_event_cosine_single_gpu/step_001370 \
--unique_events_path hx1/unique_events.parquet \
--eval_data_dir data/eval_data_latest \
--train_split val \
--test_split test \
--max_events 1000 \
--truncate_side last \
--sequence_pooling mean \
--encode_batch_size 8 \
--classifier_epochs 20 \
--device auto

torchrun --n_proc_per_node=4 benchmark_next_event_sequence_classifier.py \
--checkpoint_paths hx1/next_event_cosine_single_gpu/step_001370 \
--unique_events_path hx1/unique_events.parquet \
--eval_data_dir data/eval_data_latest \
--train_split val \
--test_split test \
--max_events 1000 \
--truncate_side last \
--sequence_pooling mean \
--encode_batch_size 8 \
--classifier_epochs 20 \
--device auto