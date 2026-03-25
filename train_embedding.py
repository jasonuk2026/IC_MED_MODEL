"""
Train a medical embedding model using BatchHardTripletLoss (Method 1).

Base model: Qwen/Qwen3-Embedding-4B
  - 4 special tokens are added: <|disease_start|>, <|disease_end|>,
    <|event_start|>, <|event_end|>
  - Each token is initialised from the mean embedding of a descriptive phrase
    (following toy_resize_embd.py).
  - Pad token is set to <|event_end|>.

Text format per sample:
  <|disease_start|>{disease name}<|disease_end|><|event_start|>{event lines}<|event_end|>

  where {event lines} is one event per line:
    {description} [{code}] | value={val} | unit={unit}

Input: parquet files produced by extract_embedding_data.py
  - One row per sample (patient history at a prediction time)
  - 'events'  : list[dict] — keys: event_pos, start, code, description, value, unit
  - 'label'   : bool  (True = patient develops the disease, False = does not)
  - 'split'   : str   ('train' / 'val' / 'test')
  - 'task'    : str   (e.g. 'new_hypertension')

Multi-GPU:
    torchrun --nproc_per_node=NUM_GPUS train_embedding.py [args]
  or
    accelerate launch --num_processes=NUM_GPUS train_embedding.py [args]

Single-GPU / CPU:
    python train_embedding.py [args]

Example:
    python train_embedding.py \\
        --data_paths data/embedding_inputs/new_diagnosis/new_hypertension_all_p0.5_m1000_ntrain1000_nval200_ntest200.parquet \\
        --output_dir output/medical-embedding \\
        --batch_size 4 --epochs 5 --bf16
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import logging

import random
from datetime import datetime

import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Task → human-readable disease name  (mirrors extract_embedding_data.py)
# --------------------------------------------------------------------------- #

TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}

# Special tokens (must match toy_resize_embd.py)
SPECIAL_TOKENS = ["<|disease_start|>", "<|disease_end|>", "<|event_start|>", "<|event_end|>"]


# --------------------------------------------------------------------------- #
# Model setup: add special tokens + initialise their embeddings
# --------------------------------------------------------------------------- #

def init_token_from_phrase(auto_model, tokenizer, new_token: str, phrase: str):
    """Initialise a new token's embedding as the mean of an existing phrase.

    Mirrors the function in toy_resize_embd.py.
    auto_model should be the raw HuggingFace model (e.g. Qwen3Model), which
    exposes .embed_tokens directly.
    """
    ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    mean_embedding = auto_model.embed_tokens.weight.data[ids].mean(dim=0)
    new_id = tokenizer.convert_tokens_to_ids(new_token)
    auto_model.embed_tokens.weight.data[new_id] = mean_embedding


def setup_qwen_special_tokens(model: SentenceTransformer):
    """Add special tokens to the tokenizer, resize the embedding table, and
    initialise each new token from a descriptive phrase.

    Must be called right after SentenceTransformer is loaded and before
    any training or inference.
    """
    tokenizer  = model.tokenizer
    auto_model = model[0].auto_model   # Qwen3Model (base, no LM head)

    # Add the four special tokens
    tokenizer.add_special_tokens({"extra_special_tokens": SPECIAL_TOKENS})
    auto_model.resize_token_embeddings(len(tokenizer))

    # Initialise embeddings from descriptive phrases (same as toy_resize_embd.py)
    init_token_from_phrase(auto_model, tokenizer, "<|disease_start|>", "disease name")
    init_token_from_phrase(auto_model, tokenizer, "<|disease_end|>",   "disease name end")
    init_token_from_phrase(auto_model, tokenizer, "<|event_start|>",   "medical event start")
    init_token_from_phrase(auto_model, tokenizer, "<|event_end|>",     "<|endoftext|>")

    # Use <|event_end|> as the pad token (same as toy_resize_embd.py)
    pad_id = tokenizer.convert_tokens_to_ids("<|event_end|>")
    tokenizer.pad_token_id = pad_id
    tokenizer.pad_token    = "<|event_end|>"

    logger.info(f"Added {len(SPECIAL_TOKENS)} special tokens. "
                f"Vocab size: {len(tokenizer)}. Pad token: '<|event_end|>'.")


# --------------------------------------------------------------------------- #
# LoRA setup
# --------------------------------------------------------------------------- #

def setup_lora(model: SentenceTransformer, r: int, lora_alpha: int, lora_dropout: float,
               target_modules: list[str], qlora: bool = False, train_norm: bool = False,
               train_new_tokens: bool = False):
    """Attach a LoRA (or QLoRA) adapter to the model.

    When qlora=True, calls prepare_model_for_kbit_training first to fix dtype
    mismatches between the 4-bit quantized layers and the fp32 norm layers.

    When train_norm=True, additionally unfreezes all RMSNorm weight vectors
    (input_layernorm, post_attention_layernorm, q_norm, k_norm, and final norm).
    These are tiny (~65K params for 0.6B, ~196K for 4B) but help the model
    adapt activation scales to new input distributions (following LongLoRA).
    """
    if qlora:
        prepare_model_for_kbit_training(model[0].auto_model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    # Create LoRA weights directly in bf16 so the entire forward pass stays in bf16
    # and flash attention does not need to cast fp32→bf16.
    model.add_adapter(peft_config)
    # Cast norm layers (kept fp32 by prepare_model_for_kbit_training) to bf16
    for name, param in model[0].auto_model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    # Optionally unfreeze all RMSNorm gamma (weight) parameters.
    # Qwen3 has: layers.*.input_layernorm, layers.*.post_attention_layernorm,
    #            layers.*.self_attn.q_norm, layers.*.self_attn.k_norm, and model.norm
    if train_norm:
        norm_params = 0
        for name, param in model[0].auto_model.named_parameters():
            if "norm.weight" in name:
                param.requires_grad = True
                norm_params += param.numel()
        logger.info(f"  Norm weights unfrozen: {norm_params:,} additional trainable params")

    # Optionally train only the 4 new special token embeddings via a gradient hook.
    # The full embedding weight is unfrozen (required for gradients), but the hook
    # zeroes all rows except the 4 special token ids — so only those rows are updated.
    # Memory cost: negligible (4 × hidden_size extra optimizer state).
    if train_new_tokens:
        embed = model[0].auto_model.embed_tokens
        embed.weight.requires_grad = True
        special_ids = [model.tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS]
        def _mask_grad(grad, ids=special_ids):
            mask = torch.zeros_like(grad)
            mask[ids] = 1.0
            return grad * mask
        embed.weight.register_hook(_mask_grad)
        logger.info(f"  Embedding: only {len(special_ids)} new special token rows trainable "
                    f"(ids={special_ids}), rest zeroed via grad hook")

    # Log trainable vs total parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied — trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    logger.info(f"  r={r}, lora_alpha={lora_alpha}, target_modules={target_modules}, "
                f"train_norm={train_norm}, train_new_tokens={train_new_tokens}")


# --------------------------------------------------------------------------- #
# Text formatting
# --------------------------------------------------------------------------- #

def format_events(events: list) -> str:
    """Convert a list of event dicts into a plain-text event history.
    One event per line: '{description} [{code}] | value=... | unit=...'
    """
    lines = []
    for event in events:
        desc  = event.get("description") or ""
        code  = event.get("code") or ""
        value = event.get("value") or ""
        unit  = event.get("unit") or ""

        if desc and code:
            line = f"{desc} [{code}]"
        elif code:
            line = f"[{code}]"
        elif desc:
            line = desc
        else:
            continue

        if value:
            line += f" | value={value}"
        if unit:
            line += f" | unit={unit}"
        lines.append(line)

    return "\n".join(lines)


def build_prompt(disease_name: str, events: list) -> str:
    """Wrap the event history with special tokens.

    Format (mirrors toy_resize_embd.py test_prompt):
      <|disease_start|>{disease_name}<|disease_end|><|event_start|>{events}<|event_end|>
    """
    event_text = format_events(events)
    return (
        f"<|disease_start|>{disease_name}<|disease_end|>"
        f"<|event_start|>{event_text}<|event_end|>"
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_split(paths: list[str], split: str) -> pd.DataFrame:
    """Load and concatenate parquet files, keeping only the requested split."""
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        frames.append(df[df["split"] == split])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_label_map(tasks: list[str]) -> dict[tuple[str, bool], int]:
    """Assign a unique integer class to every (task, polarity) pair.

    This is the key to correct multi-disease training: two positive samples
    from *different* diseases get different class IDs, so BatchHardTripletLoss
    pushes them apart rather than together.

    Example for two tasks:
      (new_hypertension, False) → 0
      (new_hypertension, True)  → 1
      (new_pancan, False)       → 2
      (new_pancan, True)        → 3
    """
    label_map = {}
    for idx, task in enumerate(sorted(set(tasks))):
        label_map[(task, False)] = idx * 2
        label_map[(task, True)]  = idx * 2 + 1
    return label_map


def build_train_dataset(df: pd.DataFrame) -> Dataset:
    """Build a HuggingFace Dataset with 'sentence' (str) and 'label' (int) columns.

    Label encoding: each (task, polarity) pair gets a unique integer so that
    BatchHardTripletLoss only groups samples of the *same disease and same
    polarity* — positive samples from different diseases are pushed apart.

    The SentenceTransformerDataCollator tokenises 'sentence' and passes
    'label' as the label tensor to the loss.
    """
    label_map = build_label_map(df["task"].tolist())
    logger.info(f"  Label map: { {f'{t}+{int(p)}': v for (t,p),v in sorted(label_map.items(), key=lambda x: x[1])} }")

    sentences, labels = [], []
    skipped = 0
    for task, events, label in zip(df["task"], df["events"], df["label"]):
        disease_name = TASK_2_DISEASE_NAME.get(task, task)
        text = build_prompt(disease_name, events)
        if not format_events(events).strip():
            skipped += 1
            continue
        sentences.append(text)
        labels.append(label_map[(task, bool(label))])

    if skipped:
        logger.warning(f"Skipped {skipped} samples with empty event lists.")

    from collections import Counter
    counts = Counter(labels)
    logger.info(f"  {len(labels)} samples across {len(counts)} classes: {dict(sorted(counts.items()))}")
    return Dataset.from_dict({"sentence": sentences, "label": labels})


def build_triplet_eval_dataset(df: pd.DataFrame, n_triplets_per_task: int = 100, seed: int = 42) -> Dataset:
    """Build (anchor, positive, negative) triplets for TripletEvaluator.

    Triplets are constructed **per task** to ensure correct semantics:
      anchor   = positive sample for disease X
      positive = another positive sample for disease X  (same disease, same polarity)
      negative = any sample NOT in class (X, positive)  (different disease OR negative)

    This directly measures the thing we care about: does a positive record for
    disease X rank closer to other positive records of the same disease than to
    any other record?
    """
    rng = random.Random(seed)

    # Group texts by (task, polarity)
    from collections import defaultdict
    groups: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for task, events, label in zip(df["task"], df["events"], df["label"]):
        if not format_events(events).strip():
            continue
        disease_name = TASK_2_DISEASE_NAME.get(task, task)
        groups[(task, bool(label))].append(build_prompt(disease_name, events))

    anchors, positives, negatives = [], [], []

    for task in sorted({t for t, _ in groups}):
        pos_texts = groups[(task, True)]
        if len(pos_texts) < 2:
            logger.warning(f"Task '{task}': only {len(pos_texts)} positive val samples — skipping triplets.")
            continue

        # All texts that are NOT (task, True) are valid negatives
        neg_texts = [t for (tsk, pol), texts in groups.items()
                     if not (tsk == task and pol)
                     for t in texts]
        if not neg_texts:
            logger.warning(f"Task '{task}': no negative val samples — skipping triplets.")
            continue

        n = min(n_triplets_per_task, len(pos_texts))
        pos_idx_pool = list(range(len(pos_texts)))
        rng.shuffle(pos_idx_pool)

        for i in range(n):
            anchor_idx = pos_idx_pool[i % len(pos_texts)]
            other      = [j for j in pos_idx_pool if j != anchor_idx]
            pos_idx    = rng.choice(other)
            neg_idx    = rng.randint(0, len(neg_texts) - 1)
            anchors.append(pos_texts[anchor_idx])
            positives.append(pos_texts[pos_idx])
            negatives.append(neg_texts[neg_idx])

    if not anchors:
        raise ValueError("No evaluation triplets could be built. Check val split contents.")

    logger.info(f"  Built {len(anchors)} evaluation triplets across {len({t for t,_ in groups})} tasks.")
    return Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-Embedding with BatchHardTripletLoss on EHR data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_paths", nargs="+", required=True,
        help="One or more parquet files from extract_embedding_data.py.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_name", default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model name or local checkpoint path.",
    )
    parser.add_argument(
        "--flash_attn", action="store_true",
        help="Enable flash_attention_2 (requires flash-attn package, Ampere+ GPU).",
    )


    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", default="output/medical-embedding",
    )

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Per-device batch size. Must be even and >= 4. "
             "Qwen3-4B is large; 4–8 is typical for a single 80 GB GPU.",
    )
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="LoRA adapters can use a higher lr than full fine-tuning.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio in [0, 1] (passed as warmup_steps float to Transformers v5).")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum",   type=int,   default=1,
                        help="Gradient accumulation steps. "
                             "Effective batch = batch_size × grad_accum × n_gpus.")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true",
                        help="Recommended for Qwen3 on Ampere+ GPUs.")
    parser.add_argument("--qlora", action="store_true",
                        help="Load model in 4-bit NF4 quantization (QLoRA). "
                             "Reduces model memory from ~8 GB to ~2 GB. "
                             "Implies --bf16 for compute dtype.")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    parser.add_argument("--lora_r",       type=int,   default=4,
                        help="LoRA rank. Higher = more capacity but more memory/params.")
    parser.add_argument("--lora_alpha",   type=int,   default=8,
                        help="LoRA alpha (scaling = lora_alpha / lora_r). Usually 2×r.")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list of linear module names to apply LoRA to. "
             "Add gate_proj,up_proj,down_proj to also adapt the MLP layers.",
    )
    parser.add_argument(
        "--train_norm", action="store_true",
        help="Also unfreeze all RMSNorm gamma (weight) parameters in addition to LoRA. "
             "Follows LongLoRA: norm weights help adapt activation scales to new distributions. "
             "Adds ~65K params for 0.6B / ~196K for 4B — negligible cost.",
    )
    parser.add_argument(
        "--train_new_tokens", action="store_true",
        help="Train only the 4 new special token embeddings via a gradient hook. "
             "Adds only 4 × hidden_size params (~4K for 0.6B / ~10K for 4B).",
    )

    # ── Loss variant ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--loss", default="hard",
        choices=["hard", "hard_soft_margin", "semi_hard", "all"],
    )
    parser.add_argument("--triplet_margin", type=float, default=5.0)

    # ── Evaluation ───────────────────────────────────────────────────────────
    parser.add_argument("--n_eval_triplets_per_task", type=int, default=100,
                        help="Number of eval triplets to build per disease task.")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_name}")

    model_kwargs = {}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        args.bf16 = True  # compute dtype for the trainer must match
    elif args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = SentenceTransformer(
        args.model_name,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
        # local_files_only=True
    )

    # Suppress the "use_cache is incompatible with gradient checkpointing" warning
    model[0].auto_model.config.use_cache = False

    # ── Add special tokens (must happen before LoRA and data formatting) ────────
    logger.info("Setting up special tokens...")
    setup_qwen_special_tokens(model)

    # ── Attach LoRA adapter (must happen after special tokens are added) ──────
    logger.info("Attaching LoRA adapter...")
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    setup_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha,
               lora_dropout=args.lora_dropout, target_modules=target_modules,
               qlora=args.qlora, train_norm=args.train_norm,
               train_new_tokens=args.train_new_tokens)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading data from: {args.data_paths}")
    train_df = load_split(args.data_paths, "train")
    val_df   = load_split(args.data_paths, "val")

    logger.info(f"Train split: {len(train_df)} rows")
    logger.info(f"Val split:   {len(val_df)} rows")

    if train_df.empty:
        raise ValueError("Train split is empty. Check --data_paths.")
    if val_df.empty:
        raise ValueError("Val split is empty. Check --data_paths.")

    logger.info("Building train dataset...")
    train_dataset = build_train_dataset(train_df)

    logger.info("Building evaluation triplets...")
    val_triplets = build_triplet_eval_dataset(
        val_df, n_triplets_per_task=args.n_eval_triplets_per_task, seed=args.seed
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_map = {
        "hard":             losses.BatchHardTripletLoss,
        "hard_soft_margin": losses.BatchHardSoftMarginTripletLoss,
        "semi_hard":        losses.BatchSemiHardTripletLoss,
        "all":              losses.BatchAllTripletLoss,
    }
    LossCls = loss_map[args.loss]
    no_margin = {losses.BatchHardSoftMarginTripletLoss, losses.BatchAllTripletLoss}
    loss_fn = LossCls(model=model) if LossCls in no_margin \
              else LossCls(model=model, margin=args.triplet_margin)
    logger.info(f"Loss: {loss_fn.__class__.__name__}")

    # ── Evaluator ─────────────────────────────────────────────────────────────
    evaluator = TripletEvaluator(
        anchors=val_triplets["anchor"],
        positives=val_triplets["positive"],
        negatives=val_triplets["negative"],
        name="medical_val",
        batch_size=2,
        show_progress_bar=True,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = SentenceTransformerTrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        learning_rate=args.lr,
        warmup_steps=args.warmup_ratio,  # float in [0,1] is treated as a ratio in Transformers v5
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_medical_val_cosine_accuracy",
        greater_is_better=True,
        logging_steps=10,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss_fn,
        evaluator=evaluator,
    )

    # logger.info("Baseline evaluation (before training):")                                                                                                                                                                      
    # trainer.evaluate()

    logger.info("Starting training...")
    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────────
    final_dir = os.path.join(run_dir, "final")
    model.save_pretrained(final_dir)
    logger.info(f"Final model saved to: {final_dir}")

    logger.info("Final evaluation on val set:")
    trainer.evaluate()


if __name__ == "__main__":
    main()
