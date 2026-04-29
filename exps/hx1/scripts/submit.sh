#!/bin/bash
set -euo pipefail

# Usage:
#   bash exps/hx1/scripts/submit.sh [options] <python|script> <training_args...>
#
# Examples:
#   bash exps/hx1/scripts/submit.sh python train_next_event_concat_mean.py --epochs 1
#   bash exps/hx1/scripts/submit.sh -N 3 -g 4 -j hurry-test -e nanotron pretrain.py --tokens 50B

# Defaults matching the HX1 PBS A100 setup.
JOB_NAME="ehr"
NUM_NODES=1
GPUS_PER_NODE=1
CPUS_PER_GPU=8
MEM_PER_GPU="64gb"
TOTAL_CPUS=""
TOTAL_MEM=""
WALLTIME="72:00:00"
CONDA_ENV="${CONDA_ENV:-torch}"
LOG_DIR="exps/hx1/logs"
CKPT_DIR="exps/hx1/ckpts"

usage() {
    echo "Usage: $0 [-j job_name] [-N nodes] [-g gpus] [-c total_cpus] [-m total_mem] [-cpg cpus_per_gpu] [-mpg mem_per_gpu] [-t walltime] [-e conda_env] <command> [args...]"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -j) JOB_NAME="$2"; shift 2 ;;
        -N) NUM_NODES="$2"; shift 2 ;;
        -g) GPUS_PER_NODE="$2"; shift 2 ;;
        -c) TOTAL_CPUS="$2"; shift 2 ;;
        -m) TOTAL_MEM="$2"; shift 2 ;;
        -cpg) CPUS_PER_GPU="$2"; shift 2 ;;
        -mpg) MEM_PER_GPU="$2"; shift 2 ;;
        -t) WALLTIME="$2"; shift 2 ;;
        -e) CONDA_ENV="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "Invalid submit option: $1" >&2; usage; exit 1 ;;
        *) break ;;
    esac
done

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# If -j is not set, infer the job name from the first Python file in the command.
if [ "$JOB_NAME" = "ehr" ]; then
    for arg in "$@"; do
        if [[ "$arg" == *.py ]]; then
            JOB_NAME="$(basename "$arg" .py)"
            break
        fi
    done
fi

if [ -n "$TOTAL_CPUS" ]; then
    NCPUS_PER_NODE="$TOTAL_CPUS"
elif [ "$GPUS_PER_NODE" -gt 0 ]; then
    NCPUS_PER_NODE=$((GPUS_PER_NODE * CPUS_PER_GPU))
else
    NCPUS_PER_NODE="$CPUS_PER_GPU"
fi

if [ -n "$TOTAL_MEM" ]; then
    MEM_PER_NODE="$TOTAL_MEM"
elif [ "$GPUS_PER_NODE" -gt 0 ]; then
    if [[ "$MEM_PER_GPU" =~ ^([0-9]+)([A-Za-z]+)$ ]]; then
        MEM_PER_NODE="$((${BASH_REMATCH[1]} * GPUS_PER_NODE))${BASH_REMATCH[2]}"
    else
        echo "Memory per GPU must look like 64gb, 64G, or 64000mb: $MEM_PER_GPU" >&2
        exit 1
    fi
else
    MEM_PER_NODE="$MEM_PER_GPU"
fi

JOB_LOG_DIR="$LOG_DIR/$JOB_NAME"
mkdir -p "$JOB_LOG_DIR" "$CKPT_DIR"

RAW_CMD_ARGS=("$@")
TORCHRUN_ARGS=("$@")
if [[ "${TORCHRUN_ARGS[0]}" == "python" || "${TORCHRUN_ARGS[0]}" == "python3" ]]; then
    TORCHRUN_ARGS=("${TORCHRUN_ARGS[@]:1}")
fi

printf -v RAW_CMD "%q " "${RAW_CMD_ARGS[@]}"
RAW_CMD="${RAW_CMD% }"
RAW_CMD="${RAW_CMD//\\$/\$}"

printf -v TRAIN_CMD "%q " "${TORCHRUN_ARGS[@]}"
TRAIN_CMD="${TRAIN_CMD% }"
# Allow runtime PBS environment variables such as $EHR_RUN_NAME in training args.
TRAIN_CMD="${TRAIN_CMD//\\$/\$}"

if [ "$GPUS_PER_NODE" -gt 0 ]; then
    RUN_MODE="torchrun"
else
    RUN_MODE="direct"
fi

if [ "$GPUS_PER_NODE" -gt 0 ]; then
    RESOURCE_SPEC="select=${NUM_NODES}:ncpus=${NCPUS_PER_NODE}:ngpus=${GPUS_PER_NODE}:mem=${MEM_PER_NODE}"
else
    RESOURCE_SPEC="select=${NUM_NODES}:ncpus=${NCPUS_PER_NODE}:mem=${MEM_PER_NODE}"
fi

JOB_SCRIPT="$JOB_LOG_DIR/${JOB_NAME}.pbs.sh"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail

CONDA_ENV="\${CONDA_ENV:-$CONDA_ENV}"
LOG_DIR="$LOG_DIR"
JOB_LOG_DIR="$JOB_LOG_DIR"
CKPT_DIR="$CKPT_DIR"
RAW_CMD='$RAW_CMD'
TRAIN_CMD='$TRAIN_CMD'
GPUS_PER_NODE="$GPUS_PER_NODE"
RUN_MODE="$RUN_MODE"
SUBMIT_JOB_NAME="$JOB_NAME"

cd "\$PBS_O_WORKDIR"
mkdir -p "\$JOB_LOG_DIR" "\$CKPT_DIR"
LIVE_LOG="\$JOB_LOG_DIR/\${PBS_JOBID}.live.log"
exec > >(stdbuf -oL -eL tee -a "\$LIVE_LOG") 2>&1

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=\${NCCL_IB_HCA:-mlx5_0}
export EHR_LOG_DIR="\$LOG_DIR"
export EHR_JOB_LOG_DIR="\$JOB_LOG_DIR"
export EHR_CKPT_DIR="\$CKPT_DIR"
export EHR_JOB_NAME="\$SUBMIT_JOB_NAME"
export EHR_JOB_ID="\${PBS_JOBID%%.*}"
export EHR_RUN_NAME="\${EHR_JOB_NAME}/\${EHR_JOB_ID}"

echo "live log: \$LIVE_LOG"
echo "job id: \$PBS_JOBID"
echo "workdir: \$PBS_O_WORKDIR"
echo "conda env: \$CONDA_ENV"
echo "checkpoint dir: \$CKPT_DIR"
echo "job log dir: \$JOB_LOG_DIR"
echo "run name: \$EHR_RUN_NAME"
echo "run mode: \$RUN_MODE"
echo "raw command: \$RAW_CMD"
echo "train command: \$TRAIN_CMD"

mapfile -t ALL_NODES < <(sort -u "\$PBS_NODEFILE")
CURRENT_SHORT=\$(hostname -s)
CURRENT_FQDN=\$(hostname -f)
CURRENT_NODE="\$CURRENT_SHORT"
for node in "\${ALL_NODES[@]}"; do
    if [[ "\$node" == "\$CURRENT_SHORT" || "\$node" == "\$CURRENT_FQDN" || "\${node%%.*}" == "\$CURRENT_SHORT" ]]; then
        CURRENT_NODE="\$node"
        break
    fi
done

NODES=("\$CURRENT_NODE")
for node in "\${ALL_NODES[@]}"; do
    if [[ "\$node" != "\$CURRENT_NODE" && "\${node%%.*}" != "\$CURRENT_SHORT" ]]; then
        NODES+=("\$node")
    fi
done

NUM_NODES=\${#NODES[@]}
MASTER_ADDR=\${NODES[0]}
JOB_NUM=\${PBS_JOBID%%.*}
MASTER_PORT=\${MASTER_PORT:-\$((20000 + JOB_NUM % 40000))}

echo "nodes: \${NODES[*]}"
echo "master: \${MASTER_ADDR}:\${MASTER_PORT}"
echo "gpus per node: \$GPUS_PER_NODE"

if [ "\$RUN_MODE" = "direct" ]; then
  eval "\$(~/miniforge3/bin/conda shell.bash hook)"
  conda activate "\$CONDA_ENV"
  eval "\$RAW_CMD"
  exit 0
fi

TORCHRUN_CMD="
  eval \\"\\\$(~/miniforge3/bin/conda shell.bash hook)\\"
  conda activate \$CONDA_ENV
  cd \$PBS_O_WORKDIR
  export PYTHONUNBUFFERED=1
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export NCCL_TIMEOUT=1800
  export NCCL_IB_DISABLE=0
  export NCCL_IB_HCA=\${NCCL_IB_HCA:-mlx5_0}
  export EHR_LOG_DIR=\$LOG_DIR
  export EHR_JOB_LOG_DIR=\$JOB_LOG_DIR
  export EHR_CKPT_DIR=\$CKPT_DIR
  export EHR_JOB_NAME=\$EHR_JOB_NAME
  export EHR_JOB_ID=\$EHR_JOB_ID
  export EHR_RUN_NAME=\$EHR_RUN_NAME
  stdbuf -oL -eL torchrun \\
    --nproc_per_node=\$GPUS_PER_NODE \\
    --nnodes=\$NUM_NODES \\
    --node_rank=\\\$NODE_RANK \\
    --master_addr=\$MASTER_ADDR \\
    --master_port=\$MASTER_PORT \\
    \$TRAIN_CMD
"

for node_rank in \$(seq 1 \$((NUM_NODES - 1))); do
    node=\${NODES[\$node_rank]}
    echo "launching node_rank=\$node_rank on \$node"
    ssh "\$node" "export NODE_RANK=\$node_rank; \$TORCHRUN_CMD" &
done

export NODE_RANK=0
echo "launching node_rank=0 on \$(hostname)"
eval "\$TORCHRUN_CMD"

wait
EOF

echo "Submitting PBS job: $JOB_NAME"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Per GPU resources: cpus=${CPUS_PER_GPU}, mem=${MEM_PER_GPU}"
echo "Total resources per node: cpus=${NCPUS_PER_NODE}, mem=${MEM_PER_NODE}"
echo "Resources: $RESOURCE_SPEC"
echo "Walltime: $WALLTIME"
echo "Log directory: $JOB_LOG_DIR"
echo "Checkpoint root: $CKPT_DIR"
echo "Run mode: $RUN_MODE"
echo "Raw command: $RAW_CMD"
echo "Training command: $TRAIN_CMD"

qsub \
    -N "$JOB_NAME" \
    -l "$RESOURCE_SPEC" \
    -l place=scatter \
    -l "walltime=${WALLTIME}" \
    -j oe \
    -o "$JOB_LOG_DIR/" \
    -v "CONDA_ENV=${CONDA_ENV}" \
    "$JOB_SCRIPT"
