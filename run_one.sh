#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_one.sh --model_path /path/to/checkpoint --output_name my_qwen_run [options]

Options:
  --model_path PATH              Local checkpoint/model path used by extract_event_emb.py and disease encoder.
  --output_name NAME             Subdirectory name under data/01_outputs/.
  --tokenizer_name NAME          Tokenizer source. Default: Qwen/Qwen3-0.6B
  --model_name NAME              Base model name / encoder identity. Default: Qwen/Qwen3-0.6B
  --nproc_per_node N             torchrun worker count for training. Default: 2
  --batch_size N                 Training batch size. Default: 64
  --eval_batch_size N            Eval batch size. Default: 32
  --train_data_epochs E [E ...]  Prepared data epochs. Default: 0
  --help                         Show this message.
EOF
}

model_path=""
output_name=""
tokenizer_name="Qwen/Qwen3-0.6B"
model_name="Qwen/Qwen3-0.6B"
nproc_per_node=2
batch_size=64
eval_batch_size=32
train_data_epochs=(0)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      model_path="$2"
      shift 2
      ;;
    --output_name)
      output_name="$2"
      shift 2
      ;;
    --tokenizer_name)
      tokenizer_name="$2"
      shift 2
      ;;
    --model_name)
      model_name="$2"
      shift 2
      ;;
    --nproc_per_node)
      nproc_per_node="$2"
      shift 2
      ;;
    --batch_size)
      batch_size="$2"
      shift 2
      ;;
    --eval_batch_size)
      eval_batch_size="$2"
      shift 2
      ;;
    --train_data_epochs)
      shift
      train_data_epochs=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        train_data_epochs+=("$1")
        shift
      done
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$model_path" || -z "$output_name" ]]; then
  echo "--model_path and --output_name are required." >&2
  usage >&2
  exit 1
fi

output_dir="data/01_outputs/${output_name}"

# python 01_gen_meta/extract_event_emb.py \
#   --model_name "$model_name" \
#   --encoder qwen3 \
#   --bf16 \
#   --output_dir "$output_dir" \
#   --model_path "$model_path" \
#   --tokenizer_name "$tokenizer_name"

torchrun --standalone --nproc_per_node="$nproc_per_node" train_disease_soft_token_classifier.py \
  --train_data_dir data/02_outputs \
  --eval_data_paths data/02_outputs/new_*/val.parquet \
  --bert_embeddings "$output_dir/embeddings.npy" \
  --bf16 \
  --batch_size "$batch_size" \
  --eval_batch_size "$eval_batch_size" \
  --pad_to_num_events 1000 \
  --num_workers 4 \
  --prefetch_factor 8 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.005 \
  --grad_clip 1.0 \
  --hidden_size 768 \
  --num_layers 2 \
  --num_heads 4 \
  --head_layers 1 \
  --pos_weight 1.0 \
  --wandb_project medical_cond_embed \
  --wandb_tags soft_token_classifier \
  --position_type learned \
  --attention_type bidirectional \
  --aux_loss_weight 0.2 \
  --align_loss_weight 0.1 \
  --disease_model_name "$model_name" \
  --train_data_epochs "${train_data_epochs[@]}"
