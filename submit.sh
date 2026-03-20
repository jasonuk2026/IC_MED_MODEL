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
eval "$(~/miniforge3/bin/conda shell.bash hook)"
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
    # new_pancan
    # new_hypertension
    new_hyperlipidemia
    # new_celiac
    # new_lupus
    # new_acutemi
)

MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B}"    # override with: sbatch --export=MODEL=... submit.sh


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
for TASK in "${TASKS[@]}"; do
    PARQUET="$PROJECTDIR/zduan/data/ehrshot_extracted/new_diagnosis/${TASK}_all.parquet"

    hf download Qwen/Qwen3.5-122B-A10B

    echo "========== $TASK  [val — threshold search] =========="
    # torchrun --nproc_per_node=4 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    #     probe_disease_logits_tp.py "$PARQUET" \
    #     --model "$MODEL" \
    #     --split val \
    #     --batch_size 1 \
    #     --num_workers 4 \
    #     --max_event_tokens 30000 \
    #     --output_dir results_tp_beta \
    #     --tp_plan moe

    hf download Qwen/Qwen3.5-122B-A10B
    
    echo "========== $TASK  [test — evaluation] =========="

    torchrun --nproc_per_node=4 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        probe_disease_logits_tp.py "$PARQUET" \
        --model "$MODEL" \
        --split test \
        --batch_size 1 \
        --num_workers 4 \
        --max_event_tokens 30000 \
        --output_dir results_tp_beta \
        --tp_plan moe
done

echo "All tasks complete."
