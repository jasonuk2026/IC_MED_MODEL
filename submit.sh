#!/bin/bash
#SBATCH --job-name=ehrshot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --exclusive                 # grab all CPUs and memory on the node
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Required by torchrun on SLURM (each job gets a unique port to avoid clashes)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# ---------------------------------------------------------------------------
# Tasks to run  (uncomment as needed)
# ---------------------------------------------------------------------------
TASKS=(
    # new_hypertension
    # new_hyperlipidemia
    # new_pancan
    new_celiac
    # new_lupus
    # new_acutemi
)

MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B}"    # override with: sbatch --export=MODEL=... submit.sh

# ---------------------------------------------------------------------------
# Pre-fetch model weights to HF cache on this node before TP loading
# (all ranks call from_pretrained simultaneously, so cache must exist first)
# ---------------------------------------------------------------------------
echo "Pre-fetching model: $MODEL"
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$MODEL")
EOF

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
for TASK in "${TASKS[@]}"; do
    PARQUET="data/llm_inputs/new_diagnosis/${TASK}_all.parquet"

    echo "========== $TASK  [val — threshold search] =========="
    torchrun --nproc_per_node=4 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        predict_logits.py "$PARQUET" \
        --model "$MODEL" \
        --split val \
        --batch_size 4 \
        --num_workers 4 \
        --max_event_tokens 30000 \
        --output_dir results_tp \
        --tensor_parallel

    echo "========== $TASK  [test — evaluation] =========="
    torchrun --nproc_per_node=4 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        predict_logits.py "$PARQUET" \
        --model "$MODEL" \
        --split test \
        --batch_size 4 \
        --num_workers 4 \
        --max_event_tokens 30000 \
        --output_dir results_tp \
        --tensor_parallel
done

echo "All tasks complete."
