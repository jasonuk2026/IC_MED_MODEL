#!/bin/bash
#SBATCH --job-name=ehr-embed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun process per node
#SBATCH --gres=gpu:4                 # GPUs per node
#SBATCH --cpus-per-task=32           # CPU cores per node (for data loading)
#SBATCH --mem=256G                   # RAM per node
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Uncomment and fill in your partition / account as needed:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch

export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Avoid NCCL timeout on slow nodes; increase if jobs are large
export NCCL_TIMEOUT=1800
# Uncomment for debugging NCCL issues:
# export NCCL_DEBUG=INFO

# ── Paths — edit these ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/embedding_inputs/sharded_m500_n16"
OUTPUT_DIR="${SCRIPT_DIR}/output/medical-embedding-custom"

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
    --lora_r              16
    --lora_alpha          32
    --lora_dropout        0.05
    --lora_target_modules q_proj,k_proj,v_proj,o_proj

    # Training
    --output_dir   "${OUTPUT_DIR}"
    --epochs       10
    --batch_size   8
    --grad_accum   4
    --lr           2e-4
    --warmup_ratio 0.1
    --weight_decay 0.01
    --grad_clip    1.0
    --log_steps    10
    --gradient_checkpointing

    # Loss / eval
    --triplet_margin           0.5
    --n_eval_triplets_per_task 200
    --eval_batch_size          8

    # wandb (remove or set project name)
    # --wandb_project ehr-embedding
    # --wandb_run_name "qwen3-0.6b-lora-r16-${SLURM_JOB_ID}"
)

# ── Launch ────────────────────────────────────────────────────────────────────
srun torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    "${SCRIPT_DIR}/train_embedding_custom.py" \
    "${TRAIN_ARGS[@]}"
