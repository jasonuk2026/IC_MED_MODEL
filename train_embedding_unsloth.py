"""
Train a medical embedding model using BatchHardTripletLoss (Unsloth version).

Base model: Qwen/Qwen3-Embedding-4B (or any model supported by FastSentenceTransformer)
  - 4 special tokens are added: <|disease_start|>, <|disease_end|>,
    <|event_start|>, <|event_end|>
  - Each token is initialised from the mean embedding of a descriptive phrase
    (following toy_resize_embd.py).
  - Pad token is set to <|event_end|>.

NOTE: Unsloth does NOT support multi-GPU / DDP. Single-GPU only.
  Unsloth handles quantisation, flash attention, and optimised gradient
  checkpointing internally — no --qlora / --flash_attn flags needed.

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

Example:
    python train_embedding_unsloth.py \\
        --data_paths data/embedding_inputs/new_diagnosis/new_hypertension_all_p0.5_m1000_ntrain1000_nval200_ntest200.parquet \\
        --output_dir output/medical-embedding-unsloth \\
        --batch_size 4 --epochs 5
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime

import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastSentenceTransformer, is_bf16_supported
from sentence_transformers import SentenceTransformerTrainer, losses
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
    """Initialise a new token's embedding as the mean of an existing phrase."""
    ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    mean_embedding = auto_model.embed_tokens.weight.data[ids].mean(dim=0)
    new_id = tokenizer.convert_tokens_to_ids(new_token)
    auto_model.embed_tokens.weight.data[new_id] = mean_embedding


def setup_qwen_special_tokens(model: FastSentenceTransformer):
    """Add special tokens, resize embedding table, initialise new embeddings.

    Must be called BEFORE get_peft_model so the resized embedding table is
    present when Unsloth patches the model.
    """
    tokenizer  = model.tokenizer
    auto_model = model[0].auto_model

    tokenizer.add_special_tokens({"extra_special_tokens": SPECIAL_TOKENS})
    auto_model.resize_token_embeddings(len(tokenizer))

    init_token_from_phrase(auto_model, tokenizer, "<|disease_start|>", "disease name")
    init_token_from_phrase(auto_model, tokenizer, "<|disease_end|>",   "disease name end")
    init_token_from_phrase(auto_model, tokenizer, "<|event_start|>",   "medical event start")
    init_token_from_phrase(auto_model, tokenizer, "<|event_end|>",     "<|endoftext|>")

    pad_id = tokenizer.convert_tokens_to_ids("<|event_end|>")
    tokenizer.pad_token_id = pad_id
    tokenizer.pad_token    = "<|event_end|>"

    logger.info(f"Added {len(SPECIAL_TOKENS)} special tokens. "
                f"Vocab size: {len(tokenizer)}. Pad token: '<|event_end|>'.")


# --------------------------------------------------------------------------- #
# LoRA setup via Unsloth
# --------------------------------------------------------------------------- #

def setup_lora(model: FastSentenceTransformer, r: int, lora_alpha: int,
               lora_dropout: float, target_modules: list[str], seed: int = 42):
    """Attach a LoRA adapter using Unsloth's get_peft_model.

    Unsloth applies its own QLoRA quantisation, flash attention, and
    gradient checkpointing internally — no extra flags needed.
    """
    FastSentenceTransformer.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        # "unsloth" uses 30% less VRAM and is optimised for long-context inputs
        use_gradient_checkpointing=False,
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
        task_type="FEATURE_EXTRACTION",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied — trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    logger.info(f"  r={r}, lora_alpha={lora_alpha}, target_modules={target_modules}")


# --------------------------------------------------------------------------- #
# Text formatting
# --------------------------------------------------------------------------- #

def format_events(events: list) -> str:
    """Convert a list of event dicts into a plain-text event history."""
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
    event_text = format_events(events)
    return (
        f"<|disease_start|>{disease_name}<|disease_end|>"
        f"<|event_start|>{event_text}<|event_end|>"
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_split(paths: list[str], split: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        frames.append(df[df["split"] == split])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_label_map(tasks: list[str]) -> dict[tuple[str, bool], int]:
    """Assign a unique integer class to every (task, polarity) pair.

    Two positive samples from different diseases get different class IDs, so
    BatchHardTripletLoss pushes them apart rather than together.
    """
    label_map = {}
    for idx, task in enumerate(sorted(set(tasks))):
        label_map[(task, False)] = idx * 2
        label_map[(task, True)]  = idx * 2 + 1
    return label_map


def build_train_dataset(df: pd.DataFrame) -> Dataset:
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

    counts = Counter(labels)
    logger.info(f"  {len(labels)} samples across {len(counts)} classes: {dict(sorted(counts.items()))}")
    return Dataset.from_dict({"sentence": sentences, "label": labels})


def build_triplet_eval_dataset(df: pd.DataFrame, n_triplets_per_task: int = 100, seed: int = 42) -> Dataset:
    """Build (anchor, positive, negative) triplets for TripletEvaluator."""
    rng = random.Random(seed)

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
        description="Train Qwen3-Embedding with BatchHardTripletLoss on EHR data (Unsloth).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_paths", nargs="+", required=True,
        help="One or more parquet files from extract_embedding_data.py.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_name", default="unsloth/Qwen3-Embedding-0.6B",
        help="Unsloth model name or local checkpoint path.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=20000,
        help="Maximum sequence length. Unsloth will truncate to this.",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", default="output/medical-embedding-unsloth")

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum",   type=int,   default=1)
    parser.add_argument("--seed",         type=int,   default=42)

    # ── LoRA ─────────────────────────────────────────────────────────────────
    parser.add_argument("--lora_r",       type=int,   default=2)
    parser.add_argument("--lora_alpha",   type=int,   default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Unsloth recommends 0 for optimised kernels.")
    parser.add_argument(
        "--lora_target_modules", type=str,
        default="q_proj,k_proj,v_proj,o_proj",
    )

    # ── Loss variant ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--loss", default="hard",
        choices=["hard", "hard_soft_margin", "semi_hard", "all"],
    )
    parser.add_argument("--triplet_margin", type=float, default=5.0)

    # ── Evaluation ───────────────────────────────────────────────────────────
    parser.add_argument("--n_eval_triplets_per_task", type=int, default=100)

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_name}")
    model = FastSentenceTransformer.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        full_finetuning=False,
    )

    # ── Add special tokens (must happen before get_peft_model) ───────────────
    logger.info("Setting up special tokens...")
    setup_qwen_special_tokens(model)

    # ── Attach LoRA adapter via Unsloth ──────────────────────────────────────
    logger.info("Attaching LoRA adapter (Unsloth)...")
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    setup_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha,
               lora_dropout=args.lora_dropout, target_modules=target_modules,
               seed=args.seed)

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
        warmup_steps=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_medical_val_cosine_accuracy",
        greater_is_better=True,
        logging_steps=10,
        dataloader_drop_last=True,
        # gradient_checkpointing is handled internally by Unsloth's get_peft_model
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss_fn,
        evaluator=evaluator,
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_dir = os.path.join(run_dir, "final")
    model.save_pretrained(final_dir)
    logger.info(f"Final model saved to: {final_dir}")

    logger.info("Final evaluation on val set:")
    trainer.evaluate()


if __name__ == "__main__":
    main()
