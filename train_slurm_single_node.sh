#!/bin/bash
#SBATCH --job-name=ehrshot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=72
#SBATCH --mem-per-gpu=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

torchrun --nproc_per_node=2 train_embedding_custom.py \
--data_paths data/new_hypertension_all_p0.5_m500_n1000.parquet data/new_hypertension_all_p0.5_m500_n1000.parquet \
--val_data_paths data/new_hypertension_all_p0.5_m500_n1000.parquet \
--val_split val \
--bf16 --flash_attn --batch_size 8 --grad_accum 4 --eval_batch_size 8 --epochs 1 --gradient_checkpointing --wandb_project ehrshot_embed --wandb_run_name "qwen3-0.6b-lora-r4-test"
