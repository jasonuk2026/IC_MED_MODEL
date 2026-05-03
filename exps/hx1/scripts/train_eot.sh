#!/bin/bash
#PBS -N ehr-event-eot-cpt
#PBS -l select=1:ncpus=16:ngpus=2:mem=200gb:gpu_type=A100
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

# Usage:
#   qsub exps/hx1/scripts/train_eot.sh
#   qsub -N my-run-name -v CONDA_ENV=torch,NGPUS=2 exps/hx1/scripts/train_eot.sh
#
# Optional env vars:
#   CONDA_ENV       conda environment name (default: torch)
#   NGPUS           number of GPUs to use (default: 2)
#   OUTPUT_BASE_DIR checkpoint root before job-name/job-id subdirs
#   EXTRA_ARGS      extra train_ehr_event_eot_cpt.py flags appended verbatim

CONDA_ENV=${CONDA_ENV:-torch}
NGPUS=${NGPUS:-2}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
TRAIN_PARQUET=${TRAIN_PARQUET:-exps/hx1/ckpts/ehr_event_eot_cpt/qwen3_0.6b_seq2048.parquet}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-exps/hx1/ckpts/ehr_event_eot_cpt}

cd "$PBS_O_WORKDIR"

JOB_NAME=${PBS_JOBNAME:-ehr-event-eot-cpt}
JOB_NUM=${PBS_JOBID%%.*}
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${JOB_NAME}/${JOB_NUM}"

LOG_DIR="exps/hx1/logs/${JOB_NAME}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

LIVE_LOG="${LOG_DIR}/${PBS_JOBID}.log"
exec > >(stdbuf -oL -eL tee -a "$LIVE_LOG") 2>&1

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

echo "job id        : ${PBS_JOBID}"
echo "job name      : ${JOB_NAME}"
echo "host          : $(hostname)"
echo "workdir       : ${PBS_O_WORKDIR}"
echo "live log      : ${LIVE_LOG}"
echo "conda env     : ${CONDA_ENV}"
echo "gpus          : ${NGPUS}"
echo "model         : ${MODEL_NAME}"
echo "train parquet : ${TRAIN_PARQUET}"
echo "output dir    : ${OUTPUT_DIR}"
echo "started       : $(date)"

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate "$CONDA_ENV"

stdbuf -oL -eL torchrun \
  --standalone \
  --nproc_per_node="${NGPUS}" \
  train_ehr_event_eot_cpt.py \
  --model_name "${MODEL_NAME}" \
  --train_parquet "${TRAIN_PARQUET}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 2 \
  --num_workers 4 \
  --epochs 1 \
  --lr 2e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.05 \
  --grad_accum 8 \
  --bf16 \
  --attn_implementation eager \
  --compile \
  --compile_mode default \
  ${EXTRA_ARGS:-}

echo "finished      : $(date)"
