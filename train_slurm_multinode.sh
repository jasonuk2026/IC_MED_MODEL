#!/bin/bash
#SBATCH --job-name=ehrshot_embed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun process per node
#SBATCH --gres=gpu:4                 # GPUs per node
#SBATCH --exclusive                                       
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# ── Environment ───────────────────────────────────────────────────────────────
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate torch

export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Avoid NCCL timeout on slow nodes; increase if jobs are large
export NCCL_TIMEOUT=1800
# Uncomment for debugging NCCL issues:
# export NCCL_DEBUG=INFO

cd "$SLURM_SUBMIT_DIR"

# ── Paths — edit these ────────────────────────────────────────────────────────
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="data/embedding_inputs/sharded_m500_ntrain50000_nval2000_ntest50000"
OUTPUT_DIR="output/ehrshot_embed"

# ── Derived DDP settings ──────────────────────────────────────────────────────
NNODES="${SLURM_NNODES}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

# Master node: first hostname in the allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29500

echo "===== Job ${SLURM_JOB_ID} ====="
echo "  Nodes        : ${NNODES}"
echo "  GPUs/node    : ${GPUS_PER_NODE}"
echo "  World size   : ${WORLD_SIZE}"
echo "  Master       : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Node list    : ${SLURM_JOB_NODELIST}"
echo "==============================="

mkdir -p logs

# ── Training args — edit as needed ────────────────────────────────────────────
TRAIN_ARGS=(
    # Data: pass all shards; each rank picks its slice via [rank::world_size]
    --data_paths     "${DATA_DIR}"/shard_*.parquet
    --val_data_paths "${DATA_DIR}/val.parquet"
    --val_split      val

    # Model
    --model_name  Qwen/Qwen3-Embedding-0.6B
    --bf16
    --flash_attn

    # LoRA
    --lora_r              8
    --lora_alpha          16
    --lora_dropout        0.05
    --lora_target_modules q_proj,k_proj,v_proj,o_proj

    # Training
    --output_dir   "${OUTPUT_DIR}"
    --epochs       1
    --batch_size   16
    --grad_accum   4
    --lr           2e-4
    --warmup_ratio 0.02
    --weight_decay 0.01
    --grad_clip    1.0
    --log_steps    10
    --gradient_checkpointing

    # Loss / eval
    --triplet_margin           0.5
    --n_eval_triplets_per_task 640
    --eval_batch_size          16

    # wandb (remove or set project name)
    --wandb_project ehr-embedding
    --wandb_run_name "qwen3-0.6b-lora-r8-${SLURM_JOB_ID}"
)

# ── Launch ────────────────────────────────────────────────────────────────────
srun --mem=0 torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    "train_embedding_custom.py" \
    "${TRAIN_ARGS[@]}"
