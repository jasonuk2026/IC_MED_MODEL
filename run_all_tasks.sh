#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

TASKS=(
    new_hypertension
    new_hyperlipidemia
    new_pancan
    new_celiac
    new_lupus
    new_acutemi
)

for TASK in "${TASKS[@]}"; do
    echo "========== $TASK =========="
    torchrun --nproc_per_node=4 predict_logits.py \
        data/llm_inputs/new_diagnosis/${TASK}_all.parquet \
        --split test \
        --batch_size 2 \
        --num_workers 4 \
        --max_event_tokens 30000
done
