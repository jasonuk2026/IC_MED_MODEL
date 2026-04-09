#!/bin/bash
# Usage: ./run_sm.sh [-n <num_gpus>] <command> [args...]
# Example: ./run_sm.sh -n 4 python train.py --lr 1e-4

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800

N_GPUS=1
while getopts ":n:" opt; do
    case $opt in
        n) N_GPUS="$OPTARG" ;;
        *) break ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
    echo "Usage: $0 [-n <num_gpus>] <command> [args...]"
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
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:"$N_GPUS" \
    --cpus-per-gpu=8 \
    --mem-per-gpu=128G \
    --time=24:00:00 \
    --output=logs/%x_%j.out \
    --error=logs/%x_%j.err \
    --wrap="
        set -e
        mkdir -p logs
        echo \"CMD: $*\"
        echo \"DATE: \$(date)\"
        eval \"\$(~/miniforge3/bin/conda shell.bash hook)\"
        conda activate torch
        cd \"$SLURM_SUBMIT_DIR\"
        $*
    "
