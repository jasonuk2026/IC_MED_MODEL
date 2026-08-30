# IC_MED_MODEL

Medical embedding experiments built around EHRShot electronic health-record event sequences. The project covers:

1. Building labeled disease-prediction examples from raw EHRShot events;
2. Extracting reusable BioLinkBERT embeddings for unique medical events;
3. Training patient-level embeddings with Qwen3-Embedding using supervised contrastive or triplet objectives;
4. Evaluating representations with triplet accuracy, similarity distributions, or independent Qwen causal-LM Yes/No logits.

This README was prepared by statically inspecting the current `main`, `MEPA`, and `collapse_codex` branches. It does not replace cluster-specific configuration. Update all paths, model names, and Slurm resources for your environment before submitting jobs.

## 1. Main pipeline

```text
EHRSHOT_ASSETS/
├── data/ehrshot.csv
├── femr/logs/omop_dir/concept.csv
├── benchmark/<task>/labeled_patients.csv
└── splits/person_id_map.csv
        │
        ├── extract_biolinkbert_embeddings.py
        │       └── data/biolinkbert_embeddings/{event_index.parquet,embeddings.npy}
        │
        ├── dataset/build_task_data.py
        │       └── EHRSHOT_ASSETS/llm_data_v6/<task>/{train,val,test}.parquet
        │
        ├── prepare_task_data.py or prepare_task_data_v2.py
        │       └── training-ready parquet files containing event_ids
        │
        └── train_embedding_*.py
                └── output/<experiment>/<timestamp>/{best,epoch_*,final}
```

The recommended order is: build the BioLinkBERT event store, build task data, validate the generated data, and only then submit GPU training jobs. Evaluation and diagnostic scripts read existing data/models and should not be run with large models on a login node.

## 2. Environment and resources

The repository expects the Conda environment `torch` (see `CLAUDE.md`):

```bash
conda activate torch
```

Based on the imports, the selected scripts require some combination of PyTorch, PyArrow, pandas, NumPy, Transformers, PEFT, Sentence Transformers, scikit-learn, tqdm, jinja2, matplotlib, scipy, `wandb`, `unsloth`, and `flash-attn`. There is currently no `requirements.txt` or `environment.yml`; use the existing cluster environment or install versions compatible with the cluster CUDA/PyTorch stack.

Prepare the following:

- The EHRShot asset directory. Set `EHRSHOT_ASSETS`, or pass explicit paths such as `--data_dir` and `--ehrshot_csv`;
- Hugging Face model caches. Common models are `michiyasunaga/BioLinkBERT-base`, `Qwen/Qwen3-Embedding-0.6B`, and Qwen causal LMs for logit probing;
- GPU compute nodes. BioLinkBERT extraction and Qwen training/inference may require substantial memory. Use `torchrun` for multi-GPU jobs and Slurm for cluster submission.

Do not run full data scans, model forward passes, training, or inference on an HPC login node. Use login nodes for code inspection and command preparation; run the actual workloads with `sbatch` or `srun`.

## 3. Supported tasks

`dataset/build_task_data.py` supports:

- `guo_los`, `guo_readmission`, and `guo_icu`;
- `new_hypertension`, `new_hyperlipidemia`, `new_pancan`, `new_celiac`, `new_lupus`, and `new_acutemi`;
- `lab_thrombocytopenia`, `lab_hyperkalemia`, `lab_hypoglycemia`, `lab_hyponatremia`, `lab_anemia`, and `chexpert`.

The embedding-training scripts mainly support six `new_*` disease tasks: hypertension, hyperlipidemia, pancreatic cancer, celiac disease, systemic lupus erythematosus, and acute myocardial infarction. Do not pass unsupported tasks to disease-conditioned training scripts without first updating their `TASK_2_DISEASE_NAME` mappings.

For most `new_*` tasks, the label indicates whether a first diagnosis occurs within one year after the prediction time. The exact semantics of `guo_*` and `lab_*` tasks are defined in `TASK_DESCRIPTIONS` in `dataset/build_task_data.py`.

## 4. Data formats and conventions

### 4.1 Raw EHRShot data

The main inputs are:

- `EHRSHOT_ASSETS/data/ehrshot.csv`: patient events and timestamps;
- `EHRSHOT_ASSETS/femr/logs/omop_dir/concept.csv`: OMOP code-to-description mapping;
- `EHRSHOT_ASSETS/benchmark/<task>/labeled_patients.csv`: task labels and prediction times;
- `EHRSHOT_ASSETS/splits/person_id_map.csv`: split information.

Only events with `start < prediction_time` are included in a patient history. `build_task_data.py` excludes `condition_occurrence` events and looks up event embeddings using `(code, value, unit)`. Events without a matching embedding cannot be used directly by the event-embedding models.

### 4.2 BioLinkBERT event store

`extract_biolinkbert_embeddings.py` scans the raw events, deduplicates them, and writes:

```text
data/biolinkbert_embeddings/
├── event_index.parquet   # event_id, event_text, code, value, unit
├── embeddings.npy        # shape=(number_of_events, hidden_size), float32
└── shards/               # temporary distributed-extraction shards
```

`event_id` is the row number in `embeddings.npy`. Every downstream `event_ids` value depends on this exact index, so rebuilding the event store requires rebuilding downstream task data as well.

### 4.3 Two task-parquet formats

The v6 files produced by `dataset/build_task_data.py` retain relatively complete information, typically including:

```text
patient_id, task, split, prediction_time, label,
n_events_in_history, event_history/events, n_tokens
```

The timeline/event column contains JSONL or event dictionaries and is useful for debugging and legacy training scripts.

`prepare_task_data.py`, `prepare_task_data_v2.py`, and `dataset/build_eval_task_data.py` use a compact representation:

```text
patient_id?   int64
task_idx      int16
label         int8
event_ids     list<int32>
source_row?   int32
```

`source_row` traces a prepared sample back to its source task parquet. Evaluation data normally does not contain this column. `task_idx` is generated by sorting the disease task names; all scripts using it must share the same task mapping.

## 5. Building training and evaluation data

### 5.1 Extract BioLinkBERT event embeddings

```bash
python extract_biolinkbert_embeddings.py \
  --ehrshot_csv EHRSHOT_ASSETS/data/ehrshot.csv \
  --concept_csv EHRSHOT_ASSETS/femr/logs/omop_dir/concept.csv \
  --model_name michiyasunaga/BioLinkBERT-base \
  --output_dir data/biolinkbert_embeddings \
  --batch_size 256 --bf16
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 extract_biolinkbert_embeddings.py \
  --output_dir data/biolinkbert_embeddings --bf16
```

The script has three phases: rank 0 builds the unique-event index; all ranks encode their index slices and write files under `shards/`; rank 0 merges them into `embeddings.npy`. It supports `--fp16`, `--bf16`, and `--local_files_only`.

### 5.2 Build task timeline parquet files

```bash
python dataset/build_task_data.py \
  --task new_hypertension \
  --data_dir EHRSHOT_ASSETS \
  --embed_dir data/biolinkbert_embeddings \
  --output_dir EHRSHOT_ASSETS/llm_data_v6 \
  --max_events 1000 --num_workers 32 \
  --splits train val test
```

`--epochs N` creates multiple randomly sampled training passes, such as `train_000.parquet` and `train_001.parquet`; val/test are written once. The repository's `build_task_data.sh` processes five `new_*` tasks by default and currently enables only the training split. Check its task list and paths before use.

### 5.3 Build evaluation data without oversampling

```bash
python dataset/build_eval_task_data.py \
  --task new_hypertension \
  --data_dir EHRSHOT_ASSETS \
  --embed_dir data/biolinkbert_embeddings \
  --output_dir EHRSHOT_ASSETS/llm_eval_data \
  --splits val test --num_events 1000 --num_workers 32
```

Unlike the training-data builder, this script writes exactly one row per labeled sample and uses `eval_sample_strategy.py` to select events.

### 5.4 Prepare embedding-training inputs

```bash
python prepare_task_data.py \
  --tasks new_hypertension new_hyperlipidemia new_pancan \
  --input_dir EHRSHOT_ASSETS/llm_data_v6 \
  --output_dir data/prepared \
  --num_shards 8 --pos_per_group 1 --neg_per_group 1 \
  --splits train val test
```

The standard preparer merges tasks, controls the positive/negative group ratio, and creates shards. `prepare_task_data_v2.py` targets patient deduplication and epoch-organized training data for `train_embedding_disease_cond_v2.py`; it expects a prepared-v2 directory through `--train_data_dir`. Do not interchange the v2 format with the text `events` parquet format.

`shard_embedding_data.py` is a separate sharder for older large parquet files that still contain `task`, `split`, `label`, and `events`. It distributes each `(task, split, label)` group round-robin across shards and writes a separate `val.parquet`. It is not the same stage as `prepare_task_data.py`'s `event_ids` format.

## 6. Data validation

Run these checks before submitting training jobs:

```bash
python verify_embedding_data.py --parquet path/to/task.parquet --task new_hypertension
python validate_train_data.py
python validate_eval_data.py
python spot_check_train_data.py
python spot_check_eval_data.py
python check_event_lookup.py \
  --data_paths data/embedding_inputs/sharded_m500/train_shard_*.parquet \
  --bert_index data/biolinkbert_embeddings/event_index.parquet
python compare_embeddings.py --dir_a path/to/store_a --dir_b path/to/store_b
```

These scripts check labels, time boundaries, schemas, duplicate patients, event coverage, and vector-store consistency. The `validate_*` and `spot_check_*` scripts contain hard-coded default directories and task sets; update their constants if your paths differ. `analyze_token_lengths.py` estimates prompt lengths before training.

## 7. Training routes

### 7.1 `train_embedding_custom.py`: text-prompt route

This script reads parquet files containing `task`, `events`, and `label`, formats each sample as a disease-prediction prompt, and trains Qwen3-Embedding with LoRA. The default objective is `BatchAllTripletLoss`; validation uses triplet accuracy.

```bash
torchrun --nproc_per_node=2 train_embedding_custom.py \
  --data_paths data/embedding_inputs/new_diagnosis/*.parquet \
  --val_data_paths data/embedding_inputs/new_diagnosis/*.parquet \
  --val_split val --model_name Qwen/Qwen3-Embedding-0.6B \
  --output_dir output/medical-embedding-custom \
  --bf16 --flash_attn --gradient_checkpointing \
  --batch_size 8 --grad_accum 4 --epochs 5 \
  --wandb_project ehr-embedding
```

`--qlora` enables 4-bit NF4 QLoRA and is documented as single-GPU only.

### 7.2 `train_embedding_disease_cond.py`: BioLinkBERT lookup plus disease conditioning

This route reads `event_ids`, retrieves BioLinkBERT vectors from `embeddings.npy`, and projects them into Qwen space through `DiseaseAwareEHREncoder`. The disease name is represented by a fixed prefix embedding, and the final EOS hidden state is used as the patient embedding. The default objective is cosine-distance soft-margin batch-hard triplet loss.

```bash
torchrun --nproc_per_node=4 train_embedding_disease_cond.py \
  --data_paths data/embedding_inputs/sharded_m500/train_shard_*.parquet \
  --val_data_paths data/embedding_inputs/sharded_m500/val.parquet \
  --val_split val \
  --bert_index data/biolinkbert_embeddings/event_index.parquet \
  --bert_embeddings data/biolinkbert_embeddings/embeddings.npy \
  --model_name Qwen/Qwen3-Embedding-0.6B \
  --output_dir output/medical-embedding-disease-cond \
  --bf16 --flash_attn --gradient_checkpointing
```

`--compile` must be combined with `--pad_to_num_events` because static sequence shapes are required.

### 7.3 `train_embedding_disease_cond_v2.py`: patient deduplication plus InfoNCE/Triplet

This newer route uses `prepare_task_data_v2.py` output through `--train_data_dir` and evaluation files through `--eval_data_paths`. The default loss is supervised-contrastive InfoNCE; `--loss triplet` selects a triplet objective.

```bash
torchrun --nproc_per_node=4 train_embedding_disease_cond_v2.py \
  --train_data_dir data/prepared_v2 \
  --tasks new_hypertension new_hyperlipidemia new_pancan \
  --eval_data_paths EHRSHOT_ASSETS/llm_eval_data/new_hypertension/val.parquet \
  --bert_embeddings data/biolinkbert_embeddings/embeddings.npy \
  --model_name Qwen/Qwen3-Embedding-0.6B \
  --output_dir output/medical-embedding-disease-cond-v2 \
  --loss infonce --temperature 0.07 --bf16 --gradient_checkpointing
```

The `task_idx`, disease ordering, and prepared data must use the same task set.

### 7.4 Other training scripts

- `train_embedding.py`: earlier general Qwen3-Embedding plus BatchHardTripletLoss implementation;
- `train_embedding_bert_lookup.py`: earlier BioLinkBERT lookup plus Qwen projection implementation;
- `train_embedding_unsloth.py`: Unsloth/Sentence Transformers implementation using text `events` parquet;
- `train_embedding_imdb.py`: IMDb experiment, not part of the EHRShot route;
- `com.py`: another implementation closely related to `train_embedding_custom.py`.

Keep one implementation's input format and checkpoint layout fixed; do not load checkpoints across implementations without checking compatibility.

## 8. Evaluation and visualization

### 8.1 Embedding triplet evaluation

```bash
python evaluate_embedding.py \
  --base_model Qwen/Qwen3-Embedding-0.6B \
  --checkpoint output/medical-embedding-custom/<timestamp>/best \
  --data_paths data/embedding_inputs/new_diagnosis/*.parquet \
  --split val --bf16
```

`evaluate_embedding.py` reports per-task/overall triplet accuracy and similarity statistics. Omitting `--checkpoint` evaluates the base model. `visualize_embeddings.py` creates disease-wise cosine-similarity KDE plots and AUC values:

```bash
python visualize_embeddings.py \
  --checkpoint output/ehrshot_embed/<timestamp>/best \
  --data_dir data/embedding_inputs/new_diagnosis \
  --output_png figures/embedding_kde.png
```

`compare_embeddings.py` compares event-vector stores; `inspect_peft_checkpoint.py` lists LoRA parameter names and shapes without loading full weights.

### 8.2 Causal-LM zero-shot Yes/No logits

`probe_disease_logits.py` and `probe_disease_logits_tp.py` compare the next-token Yes/No logits of a Qwen causal LM. Both use val threshold search followed by test evaluation.

```bash
torchrun --nproc_per_node=4 probe_disease_logits.py \
  path/to/task.parquet --split val --model Qwen/Qwen3.5-4B --output_dir results_priors
torchrun --nproc_per_node=4 probe_disease_logits.py \
  path/to/task.parquet --split test --model Qwen/Qwen3.5-4B --output_dir results_priors

torchrun --nproc_per_node=4 probe_disease_logits_tp.py \
  path/to/task.parquet --split val --model Qwen/Qwen3.5-4B --tp_plan auto --output_dir results_tp
torchrun --nproc_per_node=4 probe_disease_logits_tp.py \
  path/to/task.parquet --split test --model Qwen/Qwen3.5-4B --tp_plan auto --output_dir results_tp
```

For tensor parallelism, cache the model before launching `torchrun`; concurrent rank downloads may collide. Results are generally `<output_dir>/<task>/{val,test}.npz` and `threshold.npy`, with `scores` and `labels` in the NPZ files. `reprint_val_test_results.py` recomputes F1, AUROC, AUPRC, and the classification report.

## 9. Slurm submission

- `submit_emb.sh`: single-GPU embedding-training example;
- `train_slurm_single_node.sh`: two-GPU single-node custom training example;
- `train_slurm_multinode.sh`: two-node, four-GPU-per-node DDP template;
- `submit.sh`: four-GPU zero-shot logit-inference template;
- `run_sm.sh`: wraps arbitrary commands in a single-node Slurm job.

Check partition, GPU count, memory, time limit, model cache, paths, `MASTER_ADDR`/`MASTER_PORT`, log directory, per-GPU batch size, and DataLoader workers before submission. The current `submit.sh` uses `$PROJECTDIR`; replace it if the cluster does not define that variable.

## 10. Checkpoint layouts

Standard custom training usually saves a Transformers/PEFT directory:

```text
<run>/best/ or <run>/epoch_1/ or <run>/final/
├── adapter_config.json / adapter_model.safetensors
├── config.json and other model configuration
└── tokenizer files
```

Disease-conditioned training uses:

```text
<run>/best/
├── lora/                 # Qwen PEFT adapter
└── extra_modules.pt      # input_norm, bert_proj_1, bert_proj_2
<run>/tokenizer/
```

The base model, event dimension, task mapping, and tokenizer must match training.

## 11. Common pitfalls

- Event indices are not general token IDs: `event_id` can only index the matching `embeddings.npy`.
- Event-index keys use normalized `(code, value, unit)` values; formatting differences can cause misses.
- The temporal boundary is strict: `start < prediction_time`.
- v6 `events/event_history`, prepared `event_ids`, and evaluation `event_ids` are different formats.
- `--compile` requires fixed event lengths via `--pad_to_num_events` in the disease-conditioned route.
- `--qlora` is documented as single-GPU only.
- Batch size is per rank; effective batch size also depends on GPU count and `--grad_accum`.
- Some scripts preload large stores/dataframes; tune shards, workers, and memory.
- Historical docstrings may mention old filenames such as `build_llm_dataset_v6.py` and `predict_logits.py`; use the actual current filenames and argparse definitions.

## 12. Main-branch file index

| Path | Function |
|---|---|
| `dataset/build_task_data.py` | Build v6 task parquet with timelines |
| `dataset/build_eval_task_data.py` | Build no-oversampling val/test parquet |
| `extract_biolinkbert_embeddings.py` | Build the unique-event index and BioLinkBERT vectors |
| `extract_embedding_data.py` | Older probability-based event extraction |
| `prepare_task_data.py` | Merge tasks and create prepared shards |
| `prepare_task_data_v2.py` | Patient-deduplicated epoch preparation |
| `shard_embedding_data.py` | Shard older large parquet files |
| `dataset.py` | Lazy Parquet dataset and contiguous distributed sampler |
| `model.py` | `DiseaseAwareEHREncoder` |
| `train_embedding_custom.py` | Main text-prompt plus LoRA route |
| `train_embedding_disease_cond.py` | Event lookup plus disease-conditioned training |
| `train_embedding_disease_cond_v2.py` | Patient-deduplicated InfoNCE/Triplet training |
| `evaluate_embedding.py` | Embedding triplet-accuracy evaluation |
| `probe_disease_logits*.py` | Causal-LM Yes/No logit inference |
| `visualize_embeddings.py` | Cosine-similarity KDE/AUC plots |
| `validate_*.py`, `spot_check_*.py` | Data consistency checks |
| `trial_scripts/` | Performance and functionality experiments |
| `utils/` | CUDA prefetch and asynchronous DataLoader helpers |

## 13. Other branches: `MEPA` and `collapse_codex`

The following comparison is based on static differences from `main`. Both branches retain historical files, experimental outputs, and data/checkpoint placeholders, so they represent broader research directions rather than small isolated patches.

### 13.1 `MEPA`

`MEPA` extends the project from patient embedding and zero-shot probing to event-level language modeling, disease classification, and disease-to-patient retrieval.

#### A. Continued pre-training (CPT)

`cpt/build_cpt_block.py` converts EHR events into fixed-length blocks with `patient_id`, `block_idx`, `num_tokens`, and `input_ids`. Events are serialized through a Jinja template and ordered by patient/time; `condition_occurrence` is excluded by default. `cpt/train_cpt.py` continues pre-training a Qwen causal LM with next-token prediction and supports DDP, Flash Attention, gradient checkpointing, checkpoint resume, and step-based checkpoint saving.

Unlike `main`, which mainly trains Qwen3-Embedding LoRA/projection layers for patient vectors, MEPA directly trains a generative causal-LM backbone.

#### B. Separate event extraction and encoding layers

MEPA adds `encode_events/`: `extract_event_parquet.py` extracts deduplicated events; `event_to_text.j2` defines the shared rendering template; `encode_event_bert.py` and `encode_event_qwen.py` encode events with BioLinkBERT and Qwen; `count_unique_events.py` reports statistics; and `prepare_embedded_task_data.py` preloads, truncates, left-pads, and writes fixed-shape embeddings.

Compared with `main`'s direct CSV-based extractor, MEPA separates unique-event parquet generation, template rendering, encoding, and embedding-store creation, and preserves the exact template used.

#### C. Disease-conditioned classifiers

MEPA adds concat, cross-attention, soft-token, Transformer, and lightweight mean-pooling classifiers through `model_*_classifier.py` and `train_disease_*_classifier*.py`. `eval_concat_classifier_retrieval.py`, `plot_classifier_score_distributions.py`, `visualize_disease_retrieval.py`, and `inspect_soft_token_attention.py` provide retrieval evaluation and diagnostics.

These scripts generally read `event_ids` and `embeddings.npy`, optimize BCEWithLogitsLoss, and report AUROC, AUPRC, F1, precision, and recall. Their metrics are not directly interchangeable with `main`'s triplet-accuracy/cosine-retrieval metrics.

#### D. JEPA-style disease-to-patient retrieval

`model_jepa.py` and `train_disease_jepa.py` introduce a shared event backbone, predictor, and EMA teacher. Patient events pass through the online encoder, disease text passes through the teacher, and the objective aligns patient predictions with disease targets. Supported objectives include `retrieval_jepa`, `mse_margin`, `signed_mse`, and `negative_only_mse`, with optional supervised-contrastive and variance regularization.

This differs from `main`'s `train_embedding_disease_cond_v2.py`: the main script uses one Qwen disease-conditioned encoder with InfoNCE/Triplet, whereas MEPA uses online/EMA shallow event backbones and a predictor.

#### E. Stage 2 multi-objective event pre-training

`train_stage2.py` adds three losses on top of a CPT model:

1. CPT: causal next-token loss on the full sequence;
2. JEPA: mask complete events and predict EMA-teacher representations;
3. RED: regress masked-event predictions to the corresponding teacher EOS representation.

Weights are controlled by `--lambda_cpt`, `--lambda_jepa`, and `--lambda_red`. `train_stage2_with_inline_eval.py` evaluates inside training; periodic-evaluation variants extract event vectors, run downstream benchmarks, and support early stopping.

### 13.2 `collapse_codex`

`collapse_codex` contains and further expands the MEPA direction. It reorganizes the code into pipeline stages and adds MIMIC experiments, next-event pre-training, benchmark scripts, external-LLM prompts, and generated artifacts.

#### A. Stage-oriented layout

```text
01_gen_meta/           # unique events, templates, event embeddings, CPT/next-event metadata
02_collect_train_data/ # EHRShot task data, oversampling, retrieval preparation
exps/                  # experiment checkpoints and results
ordered_data/          # ordered MIMIC/EHR data
```

`01_gen_meta` generates metadata and event-level inputs; `02_collect_train_data` builds task samples and retrieval data. This is more structured than `main`'s flat layout.

#### B. Generic event-encoder abstraction

`01_gen_meta/encoders/base.py`, `encoders/biolinkbert.py`, `encoders/qwen3.py`, and `extract_event_emb.py` provide a shared BioLinkBERT/Qwen encoder interface with Jinja templates, mean/suffix pooling, special-token previews, distributed shard merging, and template preservation.

`01_gen_meta/build_cpt_block.py` creates full CPT blocks, while `build_next_event_train_parquet.py` creates data for predicting subsequent events. This generic encoder layer is absent from `main`.

#### C. Event-EOT CPT and next-event prediction

`build_ehr_event_eot_cpt_parquet.py` creates fixed-length sequences with event-level EOT markers and records `attention_mask`, `event_ids`, and event labels. `train_ehr_event_eot_cpt.py` supports event-EOT attention masks, causal masks, and last-token/mean-EOT pooling.

`train_next_event_cosine.py` and `train_next_event_concat_mean.py` learn sequence representations for next-event prediction, with `benchmark_next_event_*` downstream evaluations. This separate next-event pre-training route does not exist in `main`.

#### D. Foundation-model benchmarks and downstream classification

`benchmark_foundation_simple_classifier.py` and `benchmark_foundation_sequence_classifier.py` compare foundation models on full event sequences. `benchmark_next_event_*` evaluates next-event models with mean, suffix-only, or all-suffix-mean pooling and configurable truncation.

`finetune_mimic_classifier.py` fine-tunes a CPT backbone plus linear head on MIMIC, supporting frozen/unfrozen backbones, event-EOT/causal masks, and early stopping. `build_mimic_cpt_parquet.py` and `build_mimic_eval_parquet.py` create MIMIC data. `generate_llm_ehr_prompts.py` and `query_openai_compatible_prompts.py` support external OpenAI-compatible LLM comparisons. None of these MIMIC/external-API capabilities exists in `main`.

#### E. Stage 2 automation

`collapse_codex` retains `train_stage2.py` and adds inline, periodic, and synchronized periodic evaluation variants. They can extract event vectors, run foundation benchmarks, write summaries, and early-stop according to a selected metric. `main` provides independent training/evaluation scripts but no equivalent orchestration.

#### F. Data and path differences

`collapse_codex` commonly uses `data/EHRSHOT_ASSETS`, `data/01_outputs`, `data/01_results`, `data/02_outputs`, `hx1/`, and `ordered_data/`. `main` commonly uses repository-level `EHRSHOT_ASSETS`, `data/biolinkbert_embeddings`, and `output/`. Identical filenames do not imply identical defaults or parquet schemas.

The branch commits many checkpoints, results, logs, and data placeholders. When migrating code, copy only required source/configuration files and re-check data versions, model versions, and metadata.

### 13.3 Summary comparison

| Capability | `main` | `MEPA` | `collapse_codex` |
|---|---|---|---|
| EHRShot task data | Yes, flat scripts | Retained and expanded | Reorganized into `01/02` stages |
| BioLinkBERT event store | Yes | Adds templated encoding | Generic BERT/Qwen encoders |
| Qwen embedding/LoRA | Core route | Retained plus classifiers | Historical routes retained; emphasis shifts to CPT/foundation models |
| Triplet/InfoNCE retrieval | Core route | Adds JEPA retrieval | Adds next-event retrieval/benchmarks |
| Causal-LM CPT | No | Yes | Yes, including event-EOT variants |
| Stage 2 CPT+JEPA+RED | No | Yes | Yes, with automated evaluation |
| Disease classifiers | Mainly logits probing | Concat/cross-attention/soft-token/JEPA | Retained and benchmarked further |
| MIMIC data/classification | No | Little or none | Yes |
| External LLM prompt API | No | Little or none | Yes |
| Directory organization | Flat top-level | Adds modules | Stage-oriented pipeline |

### 13.4 Which branch to use

- Use `main` for the EHRShot Qwen embedding, BioLinkBERT lookup, and Yes/No-logit workflows described above.
- Refer to `MEPA` for event-level CPT, event-EOT representations, JEPA/RED, and disease classifiers.
- Refer to `collapse_codex` for the stage-oriented pipeline, next-event pre-training, MIMIC transfer, automated benchmarking, and external-LLM prompting.

Do not mix a training script into another branch without also checking its data builder, template, event-index format, checkpoint loader, and evaluator.

> Security warning: `collapse_codex/query_openai_compatible_prompts.py` contains an API-key-like credential in an argparse default. Revoke/rotate it before using the branch and switch to an environment variable such as `OPENAI_API_KEY`. Never place a real key in commands, README files, logs, or commits.
