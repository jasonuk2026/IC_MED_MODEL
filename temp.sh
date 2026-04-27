CUDA_VISIBLE_DEVICES=0 python 01_gen_meta/extract_event_emb.py \
  --encoder qwen3 \
  --model_name Qwen/Qwen3-0.6B \
  --model_path data/cpt_outputs/Qwen3-0.6B/final \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --output_dir data/01_outputs/Qwen3-0.6B/final_cpt_pad_suffix --append_token_name pad_token --pooling_mode suffix_only &

CUDA_VISIBLE_DEVICES=1 python 01_gen_meta/extract_event_emb.py \
  --encoder qwen3 \
  --model_name Qwen/Qwen3-0.6B \
  --model_path data/cpt_outputs/Qwen3-0.6B/final \
  --tokenizer_name Qwen/Qwen3-0.6B \
  --output_dir data/01_outputs/Qwen3-0.6B/final_cpt_nopad_mean &

wait