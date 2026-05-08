#!/bin/bash
# Usage: ./run_sm.sh [-n <num_gpus>] [-j <job_name>] [-c <cpus_per_gpu>] [-m <mem_per_gpu>] <command> [args...]
# Example: 
#   ./run_sm.sh -n 4 python train.py --lr 1e-4
#   ./run_sm.sh -n 2 -j my_finetune -c 12 -m 80G python train.py --model qwen

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800

# 默认值
N_GPUS=1
JOB_NAME="ehr"
CPUS_PER_GPU=16
MEM_PER_GPU="100G"

# 参数解析
while getopts ":n:j:c:m:" opt; do
    case $opt in
        n) N_GPUS="$OPTARG" ;;
        j) JOB_NAME="$OPTARG" ;;
        c) CPUS_PER_GPU="$OPTARG" ;;
        m) MEM_PER_GPU="$OPTARG" ;;
        \?) echo "无效选项: -$OPTARG" >&2; exit 1 ;;
        :) echo "选项 -$OPTARG 需要一个参数" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
    echo "Usage: $0 [-n <num_gpus>] [-j <job_name>] [-c <cpus_per_gpu>] [-m <mem_per_gpu>] <command> [args...]"
    exit 1
fi

# 如果没有手动指定 -j，则尝试从命令行参数中提取第一个 .py 文件名作为 JOB_NAME
if [ "$JOB_NAME" = "ehr" ]; then
    for arg in "$@"; do
        if [[ "$arg" == *.py ]]; then
            JOB_NAME="$(basename "$arg" .py)"
            break
        fi
    done
fi

echo "Submitting job: $JOB_NAME | GPUs: $N_GPUS | CPUs/GPU: $CPUS_PER_GPU | Mem/GPU: $MEM_PER_GPU"

sbatch \
    --job-name="$JOB_NAME" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:"$N_GPUS" \
    --cpus-per-gpu="$CPUS_PER_GPU" \
    --mem-per-gpu="$MEM_PER_GPU" \
    --time=24:00:00 \
    --output=tracked_logs/%x_%j.out \
    --error=tracked_logs/%x_%j.err \
    --wrap="
        set -e
        mkdir -p tracked_logs
        echo \"CMD: $*\"
        echo \"DATE: \$(date)\"
        echo \"JOB_NAME: $JOB_NAME | GPUs: $N_GPUS\"
        module load cuda/12.6
        eval \"\$(~/miniforge3/bin/conda shell.bash hook)\"
        conda activate torch
        cd \"$SLURM_SUBMIT_DIR\"
        $*
    "