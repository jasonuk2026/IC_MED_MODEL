#!/usr/bin/env python3

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import benchmark_foundation_simple_classifier as bench
import train_stage2 as stage2


GEN_META_DIR = Path(__file__).resolve().parent / "01_gen_meta"
if str(GEN_META_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_META_DIR))

from encoders import get_encoder


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage2 training with fully inline in-memory periodic evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_path", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lambda_cpt", type=float, default=1.0)
    p.add_argument("--lambda_jepa", type=float, default=1.0)
    p.add_argument("--lambda_red", type=float, default=1.0)
    p.add_argument("--future_len", type=int, default=256)
    p.add_argument("--num_mask_events", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.996)
    p.add_argument("--pred_mlp_ratio", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)

    p.add_argument("--eval_every_steps", type=int, required=True,
                   help="Evaluate every N actual micro-batch steps.")
    p.add_argument("--eval_at_steps", type=int, nargs="+", default=None,
                   help="Additionally evaluate at these exact micro-batch steps.")
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--selection_metric", default="test_macro_auc")

    p.add_argument("--unique_events_path", default="data/01_outputs/unique_events.parquet")
    p.add_argument("--encoder", default="qwen3")
    p.add_argument("--tokenizer_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--template_path", default=None)
    p.add_argument("--append_token_name", default=None)
    p.add_argument("--append_token_text", default=None)
    p.add_argument("--pool_max_tokens", type=int, default=None)
    p.add_argument("--pooling_mode", default="mean", choices=["mean", "suffix_only"])
    p.add_argument("--extract_batch_size", type=int, default=256)
    p.add_argument("--extract_max_length", type=int, default=256)

    p.add_argument("--eval_data_dir", default="data/llm_eval_data_ixc")
    p.add_argument("--benchmark_tasks", nargs="+", default=bench.TASKS)
    p.add_argument("--benchmark_train_split", default="val", choices=["val", "test"])
    p.add_argument("--benchmark_test_split", default="test", choices=["val", "test"])
    p.add_argument("--benchmark_max_events", type=int, default=1000)
    p.add_argument("--benchmark_truncate_side", default="first", choices=["first", "last"])
    p.add_argument("--benchmark_batch_size", type=int, default=512)
    p.add_argument("--benchmark_eval_batch_size", type=int, default=2048)
    p.add_argument("--benchmark_epochs", type=int, default=20)
    p.add_argument("--benchmark_lr", type=float, default=1e-3)
    p.add_argument("--benchmark_weight_decay", type=float, default=1e-4)
    p.add_argument("--benchmark_dropout", type=float, default=0.0)
    p.add_argument("--benchmark_hidden_dim", type=int, default=0)
    p.add_argument("--benchmark_classifier_patience", type=int, default=5)
    p.add_argument("--benchmark_num_workers", type=int, default=0)
    p.add_argument("--benchmark_seed", type=int, default=42)
    p.add_argument("--benchmark_device", default="auto")
    p.add_argument("--eval_wandb_project", default=None)
    p.add_argument("--eval_wandb_run_name", default=None)
    p.add_argument("--eval_wandb_tags", nargs="+", default=None)
    p.add_argument("--state_path", default=None)
    return p.parse_args()


def maybe_init_wandb(project, run_name, tags, config, out_dir):
    if project is None:
        return None
    import wandb
    return wandb.init(
        project=project,
        name=run_name,
        tags=tags,
        config=config,
        dir=str(out_dir),
        reinit="create_new",
        settings=wandb.Settings(console="wrap"),
    )


def write_state(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(payload, f, indent=2)


def get_raw_online(online, is_ddp):
    m = online.module if is_ddp else online
    return getattr(m, "_orig_mod", m)


def build_event_text_cache(args):
    encoder = get_encoder(
        args.encoder,
        model_name=args.model_name,
        template_path=args.template_path,
        append_token_text=args.append_token_text,
        append_token_name=args.append_token_name,
        pool_max_tokens=args.pool_max_tokens,
        pooling_mode=args.pooling_mode,
    )
    # Resolve append token before building event texts so the suffix token is
    # actually included in each text (required by suffix_only pooling).
    if encoder.append_token_name and not encoder.append_token_text:
        tmp_tok = stage2.AutoTokenizer.from_pretrained(
            args.tokenizer_name or args.model_name,
            local_files_only=args.local_files_only,
        )
        encoder.resolve_append_token(tmp_tok)
        logger.info(
            "build_event_text_cache: resolved append_token_name=%s → %r",
            encoder.append_token_name,
            encoder.append_token_text,
        )
    event_df = pd.read_parquet(args.unique_events_path)
    required_cols = {"event_id", "omop_table", "event_type", "code", "description", "value", "unit"}
    missing = sorted(required_cols - set(event_df.columns))
    if missing:
        raise ValueError("Unique events parquet missing columns: %s" % ", ".join(missing))

    event_ids = []
    texts = []
    for row in event_df.itertuples(index=False):
        text = encoder.format_event_text(
            code=str(row.code or "").strip(),
            description=str(row.description or "").strip(),
            value=str(row.value or "").strip(),
            unit=str(row.unit or "").strip(),
            omop_table=str(row.omop_table or "").strip(),
            event_type=str(row.event_type or "").strip(),
        )
        if text is None:
            continue
        event_ids.append(int(row.event_id))
        texts.append(text)
    logger.info("Prepared %d encodable unique event texts in memory.", len(texts))
    return encoder, np.asarray(event_ids, dtype=np.int64), texts


def load_eval_rows_cache(args):
    cache = {}
    for task in args.benchmark_tasks:
        cache[(task, args.benchmark_train_split)] = bench.read_eval_rows(args.eval_data_dir, task, args.benchmark_train_split)
        cache[(task, args.benchmark_test_split)] = bench.read_eval_rows(args.eval_data_dir, task, args.benchmark_test_split)
    return cache


@torch.inference_mode()
def encode_event_embeddings_in_memory(raw_online, tokenizer, encoder, event_ids, texts, args, device):
    raw_online.eval()
    encoder.resolve_append_token(tokenizer)
    if encoder.append_token_text:
        ok = encoder._set_append_token_id_from_existing_vocab(tokenizer)
        if not ok:
            raise ValueError("Inline eval requires append token to already exist in tokenizer vocab.")

    backbone = getattr(raw_online, getattr(raw_online, "base_model_prefix", "model"), None)
    if backbone is None:
        backbone = raw_online

    emb_dim = int(raw_online.config.hidden_size)
    max_event_id = int(event_ids.max()) if len(event_ids) else -1
    embeddings = np.zeros((max_event_id + 1, emb_dim), dtype=np.float32)
    total = len(texts)

    for start in range(0, len(texts), args.extract_batch_size):
        batch = texts[start:start + args.extract_batch_size]
        batch_event_ids = event_ids[start:start + args.extract_batch_size]
        if start == 0 or ((start // args.extract_batch_size) % 20 == 0):
            logger.info("Encoding events %d/%d", min(start + len(batch), total), total)
        enc = tokenizer(
            batch,
            padding=True,
            truncation=False,
            add_special_tokens=encoder.ADD_SPECIAL_TOKENS,
            return_tensors="pt",
        ).to(device)
        if enc.input_ids[0].size(0) >= args.extract_max_length:
            raise AssertionError("Sequence exceeded extract_max_length=%d" % args.extract_max_length)

        outputs = backbone(**enc, return_dict=True)
        batch_embs = encoder.get_embeddings(outputs, enc, tokenizer)
        batch_embs = encoder.postprocess_embeddings(batch_embs).cpu().numpy()
        embeddings[batch_event_ids] = batch_embs

    return embeddings


def make_benchmark_args(args):
    class Obj(object):
        pass
    out = Obj()
    out.hidden_dim = args.benchmark_hidden_dim
    out.dropout = args.benchmark_dropout
    out.lr = args.benchmark_lr
    out.weight_decay = args.benchmark_weight_decay
    out.batch_size = args.benchmark_batch_size
    out.eval_batch_size = args.benchmark_eval_batch_size
    out.num_workers = args.benchmark_num_workers
    out.early_stop_patience = args.benchmark_classifier_patience
    out.epochs = args.benchmark_epochs
    return out


def run_inline_benchmark(embeddings, eval_rows_cache, args):
    bench.set_seed(args.benchmark_seed)
    device = bench.resolve_device(args.benchmark_device)
    bench_args = make_benchmark_args(args)
    task_results = {}
    total_tasks = len(args.benchmark_tasks)

    for task_idx, task in enumerate(args.benchmark_tasks, start=1):
        logger.info("Running benchmark task %s (%d/%d)", task, task_idx, total_tasks)
        train_rows = eval_rows_cache[(task, args.benchmark_train_split)]
        test_rows = eval_rows_cache[(task, args.benchmark_test_split)]
        train_x, train_y = bench.mean_pool_rows(train_rows, embeddings, args.benchmark_max_events, args.benchmark_truncate_side)
        test_x, test_y = bench.mean_pool_rows(test_rows, embeddings, args.benchmark_max_events, args.benchmark_truncate_side)
        train_x, transformed, _, _ = bench.standardize(train_x, [test_x])
        test_x = transformed[0]
        logger.info("Inline benchmark task=%s train=%s test=%s", task, tuple(train_x.shape), tuple(test_x.shape))
        task_result = bench.train_single_task(
            train_x=train_x,
            train_y=train_y,
            val_x=train_x,
            val_y=train_y,
            test_x=test_x,
            test_y=test_y,
            args=bench_args,
            device=device,
        )
        task_results[task] = task_result
        logger.info("Task %d/%d done", task_idx, total_tasks)

    aggregate = {
        "train": bench.aggregate_task_metrics(task_results, "train"),
        "test": bench.aggregate_task_metrics(task_results, "test"),
    }
    return {"aggregate": aggregate, "tasks": task_results}


def resolve_selection_metric(eval_result, metric_name):
    if "_" not in metric_name:
        raise ValueError("selection_metric should look like '<split>_<metric>', got %r" % metric_name)
    split_name, metric_key = metric_name.split("_", 1)
    if split_name not in eval_result["aggregate"]:
        raise KeyError("Split %r not found in aggregate metrics" % split_name)
    if metric_key not in eval_result["aggregate"][split_name]:
        raise KeyError("Metric %r not found under split %r" % (metric_key, split_name))
    return float(eval_result["aggregate"][split_name][metric_key])


def log_eval_to_wandb(wandb_run, metric_name, step, eval_result, best_record):
    if wandb_run is None:
        return
    payload = {"eval/current_metric": resolve_selection_metric(eval_result, metric_name)}
    for split_name, metrics in eval_result["aggregate"].items():
        for key, value in metrics.items():
            payload["eval/%s_%s" % (split_name, key)] = float(value)
    if best_record is not None:
        payload["eval/is_best"] = 1.0 if step == best_record["step"] else 0.0
    wandb_run.log(payload, step=step)
    if best_record is not None:
        wandb_run.summary["best_checkpoint_label"] = best_record["label"]
        wandb_run.summary["best_metric_name"] = best_record["metric_name"]
        wandb_run.summary["best_metric_value"] = best_record["metric_value"]
        wandb_run.summary["best_checkpoint_step"] = best_record["step"]


def log_eval_result(eval_result, selection_metric, best_record, step_label):
    agg_train = eval_result["aggregate"]["train"]
    agg_test = eval_result["aggregate"]["test"]
    logger.info(
        "[Eval %s] train macro: auc=%.4f auprc=%.4f bal_acc=%.4f",
        step_label,
        agg_train["macro_auc"],
        agg_train["macro_auprc"],
        agg_train["macro_balanced_accuracy"],
    )
    logger.info(
        "[Eval %s] test macro: auc=%.4f auprc=%.4f bal_acc=%.4f f1=%.4f",
        step_label,
        agg_test["macro_auc"],
        agg_test["macro_auprc"],
        agg_test["macro_balanced_accuracy"],
        agg_test["macro_f1"],
    )
    for task in sorted(eval_result["tasks"]):
        stats = eval_result["tasks"][task]["test"]
        logger.info(
            "    %s: auc=%.4f auprc=%.4f bal_acc=%.4f f1=%.4f",
            task,
            stats["auc"],
            stats["auprc"],
            stats["balanced_accuracy"],
            stats["f1"],
        )
    if best_record is not None:
        logger.info(
            "[Eval %s] current best: %s=%.6f at step=%s",
            step_label,
            best_record["metric_name"],
            best_record["metric_value"],
            best_record["step"],
        )


def maybe_save_best(output_dir, raw_online, predictor, mask_embedding, optimizer, scheduler, tokenizer, global_step, global_micro_step, args):
    best_dir = output_dir / "best"
    if best_dir.exists():
        import shutil
        shutil.rmtree(str(best_dir))
    best_dir.mkdir(parents=True, exist_ok=True)
    raw_online.save_pretrained(best_dir)
    torch.save((predictor.module if isinstance(predictor, DDP) else predictor).state_dict(), best_dir / "predictor.pt")
    torch.save(mask_embedding.state_dict(), best_dir / "mask_embedding.pt")
    torch.save(
        {
            "step": global_step,
            "micro_step": global_micro_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        best_dir / "trainer_state.pt",
    )
    tokenizer.save_pretrained(best_dir)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path) if args.state_path else (output_dir / "periodic_eval_state.json")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1
    device = torch.device("cuda:%d" % local_rank if torch.cuda.is_available() else "cpu")
    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    if is_ddp:
        dist.init_process_group(backend="nccl", device_id=device)
    if rank != 0:
        logging.disable(logging.CRITICAL)

    train_wandb = None
    eval_wandb = None
    if rank == 0:
        train_wandb = maybe_init_wandb(
            args.wandb_project,
            args.wandb_run_name or ("stage2-train-%s" % output_dir.name),
            args.wandb_tags,
            vars(args),
            output_dir,
        )
        eval_wandb = maybe_init_wandb(
            args.eval_wandb_project,
            args.eval_wandb_run_name or ("stage2-eval-%s" % output_dir.name),
            args.eval_wandb_tags,
            vars(args),
            output_dir,
        )
        encoder, event_ids, event_texts = build_event_text_cache(args)
        eval_rows_cache = load_eval_rows_cache(args)
    else:
        encoder = None
        event_ids = None
        event_texts = None
        eval_rows_cache = None

    tokenizer = stage2.AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    eos_token_id = tokenizer.pad_token_id
    dataset = stage2.CPTDataset(args.data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=stage2.partial(stage2.collate_fn, eos_token_id=eos_token_id, num_mask_events=args.num_mask_events),
    )

    dtype = torch.bfloat16
    online = stage2.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        local_files_only=args.local_files_only,
    ).to(device)
    if args.gradient_checkpointing:
        online.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.compile:
        online = torch.compile(online)
    if is_ddp:
        online = DDP(online, device_ids=[local_rank], output_device=local_rank)

    teacher = stage2.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        local_files_only=args.local_files_only,
    ).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    hidden_size = online.config.hidden_size if not is_ddp else online.module.config.hidden_size
    predictor = stage2.Predictor(hidden_size, args.pred_mlp_ratio).to(device).to(dtype)
    mask_embedding = stage2.LearnableMaskEmbedding(hidden_size, dtype).to(device)
    if is_ddp:
        predictor = DDP(predictor, device_ids=[local_rank], output_device=local_rank)

    all_params = list(online.parameters()) + list(predictor.parameters()) + list(mask_embedding.parameters())
    decay_p = [p for p in all_params if p.requires_grad and p.dim() >= 2]
    nodecay_p = [p for p in all_params if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": args.weight_decay},
         {"params": nodecay_p, "weight_decay": 0.0}],
        lr=args.lr,
    )
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = stage2.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    global_micro_step = 0
    if args.resume_from:
        ckpt = Path(args.resume_from)
        raw = get_raw_online(online, is_ddp)
        raw.from_pretrained(ckpt)
        global_step, global_micro_step = stage2.load_checkpoint(ckpt, predictor, mask_embedding, optimizer, scheduler)

    preview_printed = False
    best_metric = None
    best_record = None
    bad_periods = 0
    next_eval_target = ((global_micro_step // args.eval_every_steps) + 1) * args.eval_every_steps
    eval_history = []
    TIMING_WARMUP = 10
    total_train_time = 0.0
    tokens_per_step = None
    early_stopped = False
    pending_eval_at_steps = sorted(set(args.eval_at_steps or []))

    try:
        online.train()
        predictor.train()
        for epoch in range(args.epochs):
            if is_ddp:
                sampler.set_epoch(epoch)
            run_loss = run_cpt = run_jepa = run_red = 0.0
            micro_steps = 0
            optimizer.zero_grad()
            t0 = None
            pbar = stage2.tqdm(loader, desc="Epoch %d/%d" % (epoch + 1, args.epochs), disable=(rank != 0), dynamic_ncols=True)

            for batch in pbar:
                if rank == 0 and not preview_printed:
                    stage2.log_objective_preview(tokenizer, batch, eos_token_id=eos_token_id, future_len=args.future_len)
                    preview_printed = True

                full_ids = batch["full_input_ids"].to(device)
                mask_pos = batch["masked_positions"].to(device)
                masked_ev_bounds = batch["masked_event_boundaries"]
                B, T = full_ids.shape
                if tokens_per_step is None:
                    tokens_per_step = args.batch_size * T * world_size * args.grad_accum
                if micro_steps % args.grad_accum == 0:
                    synchronize()
                    t0 = time.time()

                full_out = online(input_ids=full_ids, labels=full_ids)
                loss_cpt = full_out.loss
                loss_jepa = torch.zeros((), device=device, dtype=torch.float32)
                loss_red = torch.zeros((), device=device, dtype=torch.float32)

                if args.lambda_jepa > 0 or args.lambda_red > 0:
                    embed_tokens = get_raw_online(online, is_ddp).get_input_embeddings()
                    masked_embeds = mask_embedding.apply(full_ids, embed_tokens, mask_pos)
                    masked_out = online(inputs_embeds=masked_embeds, output_hidden_states=True)
                    online_h = masked_out.hidden_states[-1]
                    pred_h = predictor(online_h)
                    with torch.no_grad():
                        teacher_h = teacher(input_ids=full_ids, output_hidden_states=True).hidden_states[-1]
                    if args.lambda_jepa > 0:
                        pred_masked = pred_h[mask_pos]
                        target_masked = teacher_h[mask_pos].detach()
                        if pred_masked.shape[0] > 0:
                            loss_jepa = stage2.F.mse_loss(pred_masked, target_masked)
                    if args.lambda_red > 0:
                        red_terms = []
                        for b in range(B):
                            for ev_start, eos_pos in masked_ev_bounds[b]:
                                pred_event = pred_h[b, ev_start:eos_pos]
                                if pred_event.shape[0] == 0:
                                    continue
                                red_terms.append(stage2.F.mse_loss(pred_event.mean(dim=0), teacher_h[b, eos_pos].detach()))
                        if red_terms:
                            loss_red = torch.stack(red_terms).mean()

                loss = (args.lambda_cpt * loss_cpt + args.lambda_jepa * loss_jepa + args.lambda_red * loss_red) / args.grad_accum
                loss.backward()

                run_loss += loss.item() * args.grad_accum
                run_cpt += loss_cpt.item()
                run_jepa += loss_jepa.item()
                run_red += loss_red.item() if isinstance(loss_red, torch.Tensor) else float(loss_red)
                micro_steps += 1
                global_micro_step += 1

                if micro_steps % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(online.parameters()) + list(predictor.parameters()) + list(mask_embedding.parameters()),
                        args.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    stage2.update_ema(get_raw_online(online, is_ddp), teacher, args.ema_decay)

                    synchronize()
                    dt = time.time() - t0
                    global_step += 1
                    if global_step > TIMING_WARMUP:
                        total_train_time += dt

                    if rank == 0 and global_step % args.log_steps == 0:
                        n = micro_steps
                        tok_s = int(tokens_per_step / dt)
                        lr_now = scheduler.get_last_lr()[0]
                        pbar.set_postfix(
                            loss="%0.4f" % (run_loss / n),
                            cpt="%0.4f" % (run_cpt / n),
                            jepa="%0.4f" % (run_jepa / n),
                            red="%0.4f" % (run_red / n),
                            tok_s="{:,}".format(tok_s),
                            lr="%0.2e" % lr_now,
                        )
                        if train_wandb is not None:
                            train_wandb.log({
                                "train/loss": run_loss / n,
                                "train/loss_cpt": run_cpt / n,
                                "train/loss_jepa": run_jepa / n,
                                "train/loss_red": run_red / n,
                                "train/lr": lr_now,
                                "train/tok_per_s": tok_s,
                            }, step=global_step)
                        run_loss = run_cpt = run_jepa = run_red = 0.0
                        micro_steps = 0

                    should_eval_periodic = global_micro_step >= next_eval_target
                    should_eval_exact = False
                    while pending_eval_at_steps and global_micro_step >= pending_eval_at_steps[0]:
                        should_eval_exact = True
                        pending_eval_at_steps.pop(0)

                    if should_eval_periodic or should_eval_exact:
                        if is_ddp:
                            dist.barrier()
                        if rank == 0:
                            reason = []
                            if should_eval_periodic:
                                reason.append("periodic")
                            if should_eval_exact:
                                reason.append("explicit")
                            logger.info(
                                "Starting inline evaluation at micro_step=%d (%s)",
                                global_micro_step,
                                "+".join(reason),
                            )
                            raw_online = get_raw_online(online, is_ddp)
                            prev_training = raw_online.training
                            prev_pred_training = (predictor.module if isinstance(predictor, DDP) else predictor).training
                            raw_online.eval()
                            (predictor.module if isinstance(predictor, DDP) else predictor).eval()
                            eval_tokenizer = stage2.AutoTokenizer.from_pretrained(
                                args.tokenizer_name,
                                local_files_only=args.local_files_only,
                            )
                            event_embeddings = encode_event_embeddings_in_memory(
                                raw_online, eval_tokenizer, encoder, event_ids, event_texts, args, device
                            )
                            eval_result = run_inline_benchmark(event_embeddings, eval_rows_cache, args)
                            metric_value = resolve_selection_metric(eval_result, args.selection_metric)
                            record = {
                                "label": "inline-%d" % global_micro_step,
                                "checkpoint_path": "in_memory@micro_step_%d" % global_micro_step,
                                "metric_name": args.selection_metric,
                                "metric_value": metric_value,
                                "step": global_micro_step,
                                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "aggregate": eval_result["aggregate"],
                            }
                            eval_history.append(record)
                            improved = (best_metric is None) or (metric_value > best_metric + args.min_delta)
                            if improved:
                                best_metric = metric_value
                                best_record = record
                                bad_periods = 0
                                maybe_save_best(output_dir, raw_online, predictor, mask_embedding, optimizer, scheduler, tokenizer, global_step, global_micro_step, args)
                            else:
                                bad_periods += 1
                            log_eval_result(eval_result, args.selection_metric, best_record, "micro_step_%d" % global_micro_step)
                            log_eval_to_wandb(eval_wandb, args.selection_metric, global_micro_step, eval_result, best_record)
                            write_state(state_path, {
                                "selection_metric": args.selection_metric,
                                "best_metric": best_metric,
                                "best_record": best_record,
                                "bad_periods": bad_periods,
                                "global_step": global_step,
                                "global_micro_step": global_micro_step,
                                "eval_history": eval_history,
                            })
                            if prev_training:
                                raw_online.train()
                            if prev_pred_training:
                                (predictor.module if isinstance(predictor, DDP) else predictor).train()
                            if bad_periods >= args.early_stop_patience:
                                early_stopped = True
                        if is_ddp:
                            flag = torch.tensor([1 if early_stopped else 0], device=device)
                            dist.broadcast(flag, src=0)
                            early_stopped = bool(flag.item())
                            dist.barrier()
                        while global_micro_step >= next_eval_target:
                            next_eval_target += args.eval_every_steps
                        if early_stopped:
                            break
            if early_stopped:
                break

        if rank == 0:
            final_dir = output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            raw = get_raw_online(online, is_ddp)
            raw.save_pretrained(final_dir)
            torch.save((predictor.module if isinstance(predictor, DDP) else predictor).state_dict(), final_dir / "predictor.pt")
            torch.save(mask_embedding.state_dict(), final_dir / "mask_embedding.pt")
            tokenizer.save_pretrained(final_dir)
            logger.info("Final model saved -> %s", final_dir)
            logger.info("Total training time (excl. first %d steps): %.2fm", TIMING_WARMUP, total_train_time / 60.0)

            raw.eval()
            eval_tokenizer = stage2.AutoTokenizer.from_pretrained(
                args.tokenizer_name,
                local_files_only=args.local_files_only,
            )
            event_embeddings = encode_event_embeddings_in_memory(
                raw, eval_tokenizer, encoder, event_ids, event_texts, args, device
            )
            eval_result = run_inline_benchmark(event_embeddings, eval_rows_cache, args)
            metric_value = resolve_selection_metric(eval_result, args.selection_metric)
            record = {
                "label": "final",
                "checkpoint_path": str(final_dir),
                "metric_name": args.selection_metric,
                "metric_value": metric_value,
                "step": global_micro_step,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "aggregate": eval_result["aggregate"],
            }
            eval_history.append(record)
            if (best_metric is None) or (metric_value > best_metric + args.min_delta):
                best_metric = metric_value
                best_record = record
                maybe_save_best(output_dir, raw, predictor, mask_embedding, optimizer, scheduler, tokenizer, global_step, global_micro_step, args)
            log_eval_result(eval_result, args.selection_metric, best_record, "final")
            log_eval_to_wandb(eval_wandb, args.selection_metric, global_micro_step, eval_result, best_record)
            write_state(state_path, {
                "selection_metric": args.selection_metric,
                "best_metric": best_metric,
                "best_record": best_record,
                "bad_periods": bad_periods,
                "early_stopped": early_stopped,
                "global_step": global_step,
                "global_micro_step": global_micro_step,
                "eval_history": eval_history,
            })

            if best_record is not None:
                if train_wandb is not None:
                    train_wandb.summary["best_checkpoint_label"] = best_record["label"]
                    train_wandb.summary["best_metric_name"] = best_record["metric_name"]
                    train_wandb.summary["best_metric_value"] = best_record["metric_value"]
                    train_wandb.summary["best_checkpoint_step"] = best_record["step"]
                if eval_wandb is not None:
                    eval_wandb.summary["best_checkpoint_label"] = best_record["label"]
                    eval_wandb.summary["best_metric_name"] = best_record["metric_name"]
                    eval_wandb.summary["best_metric_value"] = best_record["metric_value"]
                    eval_wandb.summary["best_checkpoint_step"] = best_record["step"]
            if train_wandb is not None:
                train_wandb.finish()
            if eval_wandb is not None:
                eval_wandb.finish()
    finally:
        if is_ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
