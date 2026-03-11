#!/usr/bin/env bash
# run.sh — command reference for the EHRSHOT project
# Usage: bash run.sh <command> [options]
#   e.g. bash run.sh extract_all
#        bash run.sh extract_task new_hypertension last

set -euo pipefail

TASKS="new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi"

# ------------------------------------------------------------------ #
# Extract LLM inputs for ALL tasks (default: all prediction rows)
# ------------------------------------------------------------------ #
extract_all() {
    python extract_new_diagnosis.py \
        --prediction_strategy all \
        --path_to_output_dir ./data/llm_inputs/new_diagnosis \
        --num_workers 8
}

# ------------------------------------------------------------------ #
# Extract LLM inputs for a SINGLE task
# Usage: bash run.sh extract_task <task_name> [first|last|all]
#   e.g. bash run.sh extract_task new_hypertension last
# ------------------------------------------------------------------ #
extract_task() {
    local task=${1:?"Usage: extract_task <task_name> [first|last|all]"}
    local strategy=${2:-all}
    python extract_new_diagnosis.py \
        --tasks "$task" \
        --prediction_strategy "$strategy" \
        --path_to_output_dir ./data/llm_inputs/new_diagnosis \
        --num_workers 8
}

# ------------------------------------------------------------------ #
# (Placeholder) Run LLM inference on extracted prompts
# ------------------------------------------------------------------ #
# run_inference() {
#     python run_inference.py --input_dir ./llm_inputs/first_diagnosis
# }

# ------------------------------------------------------------------ #
# Dispatch
# ------------------------------------------------------------------ #
cmd=${1:?"Available commands: extract_all, extract_task"}
shift
"$cmd" "$@"
