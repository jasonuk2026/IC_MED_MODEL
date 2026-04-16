python train_embedding_custom.py \
--data_paths data/new_hypertension_all_p0.5_m500_n1000.parquet \
--val_data_paths data/new_hypertension_all_p0.5_m500_n1000.parquet \
--val_split train \
--flash_attn --bf16 --batch_size 4 --grad_accum 1 --eval_batch_size 8 --epochs 1

  CUDA_VISIBLE_DEVICES=1 python train_embedding_custom.py \
      --data_paths data/embedding_inputs/new_diagnosis/new_hypertension_all_p0.5_m500_n1000.parquet \
      --val_data_paths data/embedding_inputs/new_diagnosis/new_hypertension_all_p0.5_m500_n1000.parquet \
      --val_split train \
      --bf16 --flash_attn --batch_size 4 --grad_accum 8 --epochs 1

salloc --nodes=1 --ntasks=1 --cpus-per-gpu=72 --mem-per-gpu=100G --gres=gpu:1 --time=08:00:00
salloc --nodes=1 --ntasks=1 --cpus-per-task=72 --mem-per-cpu=2G --time=04:00:00
salloc --nodes=1 --ntasks=1 --exclusive --gres=gpu:4 --time=02:00:00
·

module load brics/tmux/3.4


# 1000 samples take 0.5 hour, then 100,000 would be 50 hours. We have 6 tasks. Amount to 300 hours. If we use 16 GPUs, that would be 20 hours.

torchrun --nproc_per_node=4 train_embedding_custom.py \                                                                                                                                               
--data_paths data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000/shard_*.parquet \                                                                                                                             
--val_data_paths data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000/val.parquet \                                                                                                                             
--wandb_project ehrshot_embed \                                                                                                                                                                   
--wandb_run_name qwen3-0.6b-lora-r4 \                                                                                                                                                             
--bf16 --flash_attn --batch_size 8 --grad_accum 4 --eval_batch_size 8 --epochs 5 --eval_only --gradient_checkpointing

python extract_embedding_data.py --num_workers 16 --event_sample_prob 0.5 --event_sample_target 500 --train_target 50000 --val_target 2000 --test_target 5000

python shard_embedding_data.py \
--input_files data/embedding_inputs/new_diagnosis/*m500_ntrain50000_nval2000_ntest5000.parquet \
--output_dir data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000 \
--n_shards 16

torchrun --nproc_per_node=2 train_embedding_custom.py \
--data_paths data/new_hypertension_all_p0.5_m500_n1000.parquet \
--val_data_paths data/new_hypertension_all_p0.5_m500_n1000.parquet \
--wandb_project ehrshot_embed \
--val_split train \
--bf16 --flash_attn --batch_size 8 --grad_accum 4 --eval_batch_size 8 --epochs 1 --gradient_checkpointing

torchrun --nproc_per_node=2 train_embedding_custom.py \
--data_paths data/new_hypertension_all_p0.5_m500_n1000.parquet data/new_hypertension_all_p0.5_m500_n1000.parquet \
--val_data_paths data/new_hypertension_all_p0.5_m500_n1000.parquet \
--val_split train \
--bf16 --flash_attn --batch_size 32 --grad_accum 1 --eval_batch_size 32 --epochs 5 --gradient_checkpointing

./run_sm.sh -n 4 torchrun --nproc_per_node=4 train_embedding_custom.py --val_data_paths data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000/val.parquet --val_split val --bf16 --flash_attn --checkpoint output/ehrshot_embed/20260326_143823/final --n_eval_triplets_per_task 640 --eval_batch_size 16 --eval_only


CUDA_VISIBLE_DEVICES=0 
python train_embedding_disease_cond.py \
--data_paths data/embedding_inputs/new_diagnosis/*_all_p0.5_m500_ntrain50000_nval2000_ntest5000.parquet \
--val_data_paths data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000/val.parquet \
--bf16 --wandb_project medical_cond_embed --wandb_tags trial \
--bert_index data/biolinkbert_embeddings/event_index.parquet \
--bert_embeddings data/biolinkbert_embeddings/embeddings.npy \
--batch_size 64 --grad_accum 1 --eval_batch_size 64 --flash_attn \
--pad_to_num_events 500 --n_eval_triplets_per_task 256

python train_embedding_disease_cond.py \
--data_paths data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000/shard_*.parquet \
--val_data_paths data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000/val.parquet \
--bf16 --wandb_project medical_cond_embed --wandb_tags trial_for_abnormal_gnorm \
--bert_index data/biolinkbert_embeddings/event_index.parquet \
--bert_embeddings data/biolinkbert_embeddings/embeddings.npy \
--batch_size 64 --grad_accum 1 --eval_batch_size 64 \
--pad_to_num_events 500 --n_eval_triplets_per_task 256 \
--compile --flash_attn \
--num_workers 8 --prefetch_factor 8 \
--epochs 5 \
--lora_r 8 --lora_alpha 16 \
--grad_clip 10.0

python train_embedding_disease_cond.py \
--data_paths data/embedding_inputs/new_diagnosis/new_*_all_p0.5_m500_ntrain50000_nval2000_ntest5000.parquet \
--val_data_paths data/embedding_inputs/new_diagnosis/new_lupus_all_p0.5_m500_ntrain50000_nval2000_ntest5000.parquet \
--bf16 --wandb_project medical_cond_embed --wandb_tags trial_for_abnormal_gnorm \
--bert_index data/biolinkbert_embeddings/event_index.parquet \
--bert_embeddings data/biolinkbert_embeddings/embeddings.npy \
--batch_size 64 --grad_accum 1 --eval_batch_size 64 \
--pad_to_num_events 500 --n_eval_triplets_per_task 256 \
--compile --flash_attn \
--num_workers 8 --prefetch_factor 8 \
--epochs 5 \
--lora_r 8 --lora_alpha 16 \
--grad_clip 1.0


srun --jobid=3575090 \
--pty bash


# Evaluation

TERM=dumb python train_embedding_disease_cond.py \
--val_data_paths data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000/shard_000.parquet \
--bf16 --wandb_project medical_cond_embed --wandb_tags eval_on_train \
--bert_index data/biolinkbert_embeddings/event_index.parquet \
--bert_embeddings data/biolinkbert_embeddings/embeddings.npy \
--batch_size 64 --grad_accum 1 --eval_batch_size 64 \
--pad_to_num_events 500 --n_eval_triplets_per_task 256 \
--compile --flash_attn \
--num_workers 8 --prefetch_factor 8 \
--eval_only --checkpoint output/medical-embedding-disease-cond/20260401_181151/final \
--val_split train


python train_embedding_disease_cond_v2.py \
--train_data_paths data/compiled_data/ehrshot_train/ep0/new_acutemi_new_celiac_new_hyperlipidemia_new_hypertension_new_lupus_new_pancan/train/shard_*.parquet \
--eval_data_paths data/compiled_data/ehrshot_eval/new_*/val.parquet \
--bf16 --wandb_project medical_cond_embed --wandb_tags trial_for_abnormal_gnorm \
--bert_embeddings data/biolinkbert_embeddings/embeddings.npy \
--batch_size 32 --grad_accum 1 --eval_batch_size 32 \
--pad_to_num_events 1000 --n_eval_triplets_per_task 256 \
--compile --flash_attn \
--num_workers 8 --prefetch_factor 8 \
--epochs 4 \
--lora_r 8 --lora_alpha 16 \
--grad_clip 1.0


WANDB_API_KEY=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 torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py \
--train_data_dir data/llm_data_ixc_patch \
--eval_data_paths data/llm_eval_data_ixc/new_*/val.parquet \
--bf16 --wandb_project medical_cond_embed --wandb_tags shallow_task_cond \
--bert_embeddings data/embeddings.npy \
--batch_size 32 --grad_accum 1 --eval_batch_size 32 \
--pad_to_num_events 1000 --n_eval_triplets_per_task 1024 \
--flash_attn --compile \
--num_workers 4 --prefetch_factor 8 \
--epochs 5 --warmup_ratio 0.1 \
--stage1_steps 1000 --lr_proj 2e-4 --lr_lora 5e-5 \
--triplet_margin 0.3 \
--lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
--weight_decay 0.005 \
--grad_clip 1.0 \
--encoder_mode proj_cond_query_proto



output/medical-embedding-disease-cond-v2/20260408_153231/best

WANDB_API_KEY=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 CUDA_VISIBLE_DEVICES=0 python train_embedding_disease_cond_v2.py \
--train_data_dir data/llm_data_ixc_patch \
--eval_data_paths data/llm_eval_data_ixc/new_*/val.parquet \
--checkpoint output/medical-embedding-disease-cond-v2/20260408_153231/best \
--eval_only \
--bf16 --wandb_project medical_cond_embed --wandb_tags stage1_meanpool_triplet \
--bert_embeddings data/embeddings.npy \
--batch_size 32 --grad_accum 1 --eval_batch_size 32 \
--pad_to_num_events 1000 --n_eval_triplets_per_task 256 \
--flash_attn \
--num_workers 4 --prefetch_factor 8 \
--epochs 5 --warmup_ratio 0.1 \
--stage1_steps 1000 --lr_proj 2e-4 --lr_lora 5e-5 \
--triplet_margin 0.3 \
--lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
--weight_decay 0.005 \
--grad_clip 1.0

WANDB_API_KEY=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 torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py --train_data_dir data/llm_data_ixc_patch --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet --bf16 --wandb_project medical_cond_embed --wandb_tags shallow_task_cond --bert_embeddings data/embeddings.npy --batch_size 32 --grad_accum 1 --eval_batch_size 32 --pad_to_num_events 1000 --n_eval_triplets_per_task 1024 --flash_attn --compile --num_workers 4 --prefetch_factor 8 --epochs 5 --warmup_ratio 0.1 --stage1_steps 1000 --lr_proj 2e-4 --lr_lora 5e-5 --triplet_margin 0.3 --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --weight_decay 0.005 --grad_clip 1.0 --encoder_mode proj --train_objective disease_retrieval

python visualize_disease_retrieval.py \
  --checkpoint output/medical-embedding-disease-cond-v2/20260409_114239/stage1 \
  --data_paths data/llm_eval_data_ixc/new_*/val.parquet \
  --bert_embeddings data/embeddings.npy \
  --output_dir figures/retrieval_viz \
  --encoder_mode proj \
  --bf16 \
  --flash_attn \
  --reduction auto \
  --max_points_per_task 0


WANDB_API_KEY=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 
torchrun --nproc_per_node=2 train_embedding_disease_cond_v2.py --train_data_dir EHRSHOT_ASSETS/llm_data_v7 --eval_data_paths data/llm_eval_data_ixc/new_*/val.parquet --bf16 --wandb_project medical_cond_embed --wandb_tags shallow_task_cond --bert_embeddings data/embeddings.npy --batch_size 32 --grad_accum 1 --eval_batch_size 32 --pad_to_num_events 1000 --n_eval_triplets_per_task 1024 --flash_attn --compile --num_workers 4 --prefetch_factor 8 --epochs 1 --warmup_ratio 0.1 --stage1_steps 1000 --lr_proj 2e-4 --lr_lora 5e-5 --triplet_margin 0.3 --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --weight_decay 0.005 --grad_clip 1.0 --encoder_mode proj --train_objective disease_retrieval

WANDB_API_KEY=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 python train_embedding_disease_cond_v2.py --train_data_dir data/llm_data_ixc_patch --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet --bf16 --wandb_project medical_cond_embed --wandb_tags shallow_task_cond --bert_embeddings data/embeddings.npy --batch_size 32 --grad_accum 1 --eval_batch_size 4 --n_eval_triplets_per_task 1024 --flash_attn --compile --num_workers 4 --prefetch_factor 8 --epochs 1 --warmup_ratio 0.1 --stage1_steps 1000 --lr_proj 2e-4 --lr_lora 5e-5 --triplet_margin 0.3 --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --weight_decay 0.005 --grad_clip 1.0 --encoder_mode proj --train_objective disease_retrieval --eval_only --checkpoint output/medical-embedding-disease-cond-v2/20260409_114239/stage1 --pad_to_num_events 1000

torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py \
--train_data_dir data/llm_data_ixc_patch \
--eval_data_paths data/llm_eval_data_ixc/new_*/val.parquet \
--bert_embeddings data/embeddings.npy \
--encoder_mode proj \
--train_objective disease_retrieval \
--bf16 --compile \
--batch_size 32 --grad_accum 1 --eval_batch_size 32 \
--pad_to_num_events 1000 \
--num_workers 4 --prefetch_factor 8 \
--warmup_ratio 0.1 \
--lr_proj 2e-4 \
--weight_decay 0.005 \
--grad_clip 1.0 \
--n_eval_triplets_per_task 1024 \
--wandb_project medical_cond_embed \
--wandb_tags shallow_task_cond

python train_embedding_disease_cond_v2.py \
--train_data_dir EHRSHOT_ASSETS/llm_data_v7 \
--eval_data_paths data/llm_eval_data_ixc/new_*/val.parquet \
--bert_embeddings data/embeddings.npy \
--encoder_mode proj \
--train_objective disease_retrieval \
--bf16 --compile \
--batch_size 32 --grad_accum 1 --eval_batch_size 32 \
--pad_to_num_events 1000 \
--num_workers 4 --prefetch_factor 8 \
--warmup_ratio 0.1 \
--lr_proj 2e-4 \
--weight_decay 0.005 \
--grad_clip 1.0 \
--n_eval_triplets_per_task 1024 \
--wandb_project medical_cond_embed \
--wandb_tags shallow_task_cond


WANDB_API_KEY=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 torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py --train_data_dir data/llm_data_ixc_patch --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet --bf16 --wandb_project medical_cond_embed --wandb_tags shallow_task_cond --bert_embeddings data/embeddings.npy --batch_size 32 --grad_accum 1 --eval_batch_size 32 --pad_to_num_events 1000 --n_eval_triplets_per_task 1024 --flash_attn --compile --num_workers 4 --prefetch_factor 8 --epochs 1 --warmup_ratio 0.1 --stage1_steps 1000 --lr_proj 2e-4 --lr_lora 5e-5 --triplet_margin 0.3 --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --weight_decay 0.005 --grad_clip 1.0 --encoder_mode proj --train_objective disease_retrieval

WANDB_API_KEY=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 torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py --train_data_dir data/llm_data_ixc_patch --eval_data_paths data/llm_eval_data_ixc/new_acutemi/val.parquet data/llm_eval_data_ixc/new_celiac/val.parquet data/llm_eval_data_ixc/new_hyperlipidemia/val.parquet data/llm_eval_data_ixc/new_hypertension/val.parquet data/llm_eval_data_ixc/new_lupus/val.parquet data/llm_eval_data_ixc/new_pancan/val.parquet --bf16 --wandb_project medical_cond_embed --wandb_tags shallow_task_cond --bert_embeddings data/embeddings.npy --batch_size 32 --grad_accum 1 --eval_batch_size 32 --pad_to_num_events 1000 --n_eval_triplets_per_task 1024 --flash_attn --compile --num_workers 4 --prefetch_factor 8 --epochs 1 --warmup_ratio 0.1 --stage1_steps 1000 --lr_proj 2e-4 --lr_lora 5e-5 --triplet_margin 0.3 --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --weight_decay 0.005 --grad_clip 1.0 --encoder_mode proj --train_objective disease_retrieval