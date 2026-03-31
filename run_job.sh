#!/bin/bash
# Usage: ./run_job.sh <command> [args...]
# Example: ./run_job.sh python visualize_embeddings.py --checkpoint output/best

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# Extract job name from the first .py file found in the arguments
JOB_NAME="ehr"
for arg in "$@"; do
    if [[ "$arg" == *.py ]]; then
        JOB_NAME="$(basename "$arg" .py)"
        break
    fi
done

sbatch \
    --job-name="$JOB_NAME" \
    --partition=workq \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-gpu=72 \
    --mem-per-gpu=100G \
    --time=24:00:00 \
    --output=logs/%x_%j.out \
    --error=logs/%x_%j.err \
    --wrap="
        set -e
        mkdir -p logs
        eval \"\$(~/miniforge3/bin/conda shell.bash hook)\"
        conda activate torch
        cd \"$SLURM_SUBMIT_DIR\"
        $*
    "
