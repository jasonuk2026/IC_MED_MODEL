#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

TASKS=(
    # new_hypertension
    # new_hyperlipidemia
    # new_pancan
    new_celiac
    # new_lupus
    # new_acutemi
)

for TASK in "${TASKS[@]}"; do
    echo "========== $TASK  [val — threshold search] =========="
    TORCH_DISTRIBUTED_DEBUG=DETAIL conda run -n torch torchrun --nproc_per_node=4 predict_logits.py \
        $PROJECTDIR/zduan/data/ehrshot_extracted/new_diagnosis/${TASK}_all.parquet \
        --split val \
	--model Qwen/Qwen3.5-122B-A10B \
        --batch_size 1 \
        --num_workers 4 \
        --max_event_tokens 30000 \
        --output_dir results_tp \
        --tensor_parallel

    echo "========== $TASK  [test — evaluation] =========="
    conda run -n torch torchrun --nproc_per_node=4 predict_logits.py \
        $PROJECTDIR/zduan/data/ehrshot_extracted/new_diagnosis/${TASK}_all.parquet \
        --split test \
	--model Qwen/Qwen3.5-122B-A10B \
        --batch_size 4 \
        --num_workers 4 \
        --max_event_tokens 30000 \
        --output_dir results_tp \
        --tensor_parallel
done
