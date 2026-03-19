#!/bin/bash
#SBATCH --job-name=ehrshot_vllm
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

# ---------------------------------------------------------------------------
# Tasks to run  (uncomment as needed)
# ---------------------------------------------------------------------------
TASKS=(
    # new_hypertension
    # new_hyperlipidemia
    new_pancan
    # new_celiac
    # new_lupus
    new_acutemi
)

MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B}"    # override with: sbatch --export=MODEL=... submit_vllm.sh

# ---------------------------------------------------------------------------
# Inference loop  (no torchrun — vLLM handles multi-GPU internally)
# ---------------------------------------------------------------------------
for TASK in "${TASKS[@]}"; do
    PARQUET="$PROJECTDIR/zduan/data/ehrshot_extracted/new_diagnosis/${TASK}_all.parquet"

    echo "========== $TASK  [val — threshold search] =========="
    python predict_logits_vllm.py "$PARQUET" \
        --model "$MODEL" \
        --split val \
        --tensor_parallel_size 4 \
        --max_num_seqs 16 \
        --max_event_tokens 30000 \
        --output_dir results_vllm

    echo "========== $TASK  [test — evaluation] =========="
    python predict_logits_vllm.py "$PARQUET" \
        --model "$MODEL" \
        --split test \
        --tensor_parallel_size 4 \
        --max_num_seqs 16 \
        --max_event_tokens 30000 \
        --output_dir results_vllm
done

echo "All tasks complete."
