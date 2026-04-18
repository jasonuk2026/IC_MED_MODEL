#!/usr/bin/env python3

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import train_stage2 as stage2


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Synchronous stage2 training with inline periodic evaluation and early stopping.",
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
    p.add_argument("--wandb_project", default=None,
                   help="Training W&B project from the stage2 loop.")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)

    p.add_argument("--eval_every_steps", type=int, required=True,
                   help="Evaluate every N actual micro-batch steps.")
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--selection_metric", default="test_macro_auc")

    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--extract_script", default="01_gen_meta/extract_event_emb.py")
    p.add_argument("--benchmark_script", default="benchmark_foundation_simple_classifier.py")
    p.add_argument("--extract_unique_events_path", default="data/01_outputs/unique_events.parquet")
    p.add_argument("--extract_output_prefix", default="stage2_periodic_eval")
    p.add_argument("--encoder", default="qwen3")
    p.add_argument("--tokenizer_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--pooling_mode", default="mean", choices=["mean", "suffix_only"])
    p.add_argument("--append_token_name", default=None)
    p.add_argument("--append_token_text", default=None)
    p.add_argument("--pool_max_tokens", type=int, default=None)
    p.add_argument("--extract_batch_size", type=int, default=256)
    p.add_argument("--extract_max_length", type=int, default=256)
    p.add_argument("--extract_preview_tokenization_n", type=int, default=0)
    p.add_argument("--extract_bf16", action="store_true")
    p.add_argument("--extract_fp16", action="store_true")

    p.add_argument("--eval_data_dir", default="data/llm_eval_data_ixc")
    p.add_argument("--benchmark_tasks", nargs="+", default=[
        "new_acutemi",
        "new_celiac",
        "new_hyperlipidemia",
        "new_hypertension",
        "new_lupus",
        "new_pancan",
    ])
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
    p.add_argument("--benchmark_output_root", default="output/stage2_periodic_benchmark")
    p.add_argument("--eval_wandb_project", default=None,
                   help="Optional separate W&B project for periodic eval metrics.")
    p.add_argument("--eval_wandb_run_name", default=None)
    p.add_argument("--eval_wandb_tags", nargs="+", default=None)
    p.add_argument("--state_path", default=None)
    return p.parse_args()


def run_command(cmd, cwd):
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def maybe_init_eval_wandb(args, output_dir):
    if args.eval_wandb_project is None:
        return None
    import wandb
    run_name = args.eval_wandb_run_name or ("stage2-inline-eval-%s" % Path(output_dir).name)
    return wandb.init(
        project=args.eval_wandb_project,
        name=run_name,
        tags=args.eval_wandb_tags,
        config=vars(args),
        dir=str(output_dir),
        settings=wandb.Settings(console="wrap"),
    )


def log_eval_to_wandb(wandb_run, record, best_record):
    if wandb_run is None:
        return
    summary_df = pd.read_csv(record["summary_path"])
    if summary_df.empty:
        return
    top_row = summary_df.iloc[0].to_dict()
    payload = {
        "eval/current_metric": record["metric_value"],
        "eval/is_best": 1.0 if best_record and record["checkpoint_path"] == best_record["checkpoint_path"] else 0.0,
    }
    for key, value in top_row.items():
        if key == "embedding_dir":
            continue
        try:
            payload["eval/%s" % key] = float(value)
        except (TypeError, ValueError):
            continue
    wandb_run.log(payload, step=record["step"] if record["step"] is not None else None)
    if best_record is not None:
        wandb_run.summary["best_checkpoint_path"] = best_record["checkpoint_path"]
        wandb_run.summary["best_checkpoint_label"] = best_record["label"]
        wandb_run.summary["best_metric_name"] = best_record["metric_name"]
        wandb_run.summary["best_metric_value"] = best_record["metric_value"]
        if best_record.get("step") is not None:
            wandb_run.summary["best_checkpoint_step"] = best_record["step"]


def write_state(state_path, payload):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(state_path), "w") as f:
        json.dump(payload, f, indent=2)


def benchmark_run_dir(output_root, before_dirs):
    after_dirs = set()
    if output_root.exists():
        after_dirs = set([p for p in output_root.iterdir() if p.is_dir()])
    new_dirs = sorted(after_dirs - before_dirs, key=lambda p: p.stat().st_mtime)
    if not new_dirs:
        raise RuntimeError("Benchmark finished but no new output directory was created under %s" % output_root)
    return new_dirs[-1]


def evaluate_checkpoint(repo_root, args, checkpoint_path, label):
    embedding_dir_name = "%s_%s" % (args.extract_output_prefix, label)
    embedding_output_dir = repo_root / "data" / "01_outputs" / embedding_dir_name

    extract_cmd = [
        args.python_executable,
        str(repo_root / args.extract_script),
        "--encoder", args.encoder,
        "--model_name", args.model_name,
        "--model_path", str(checkpoint_path),
        "--tokenizer_name", args.tokenizer_name,
        "--unique_events_path", args.extract_unique_events_path,
        "--output_dir", str(embedding_output_dir),
        "--pooling_mode", args.pooling_mode,
        "--batch_size", str(args.extract_batch_size),
        "--max_length", str(args.extract_max_length),
        "--preview_tokenization_n", str(args.extract_preview_tokenization_n),
    ]
    if args.append_token_name:
        extract_cmd.extend(["--append_token_name", args.append_token_name])
    if args.append_token_text:
        extract_cmd.extend(["--append_token_text", args.append_token_text])
    if args.pool_max_tokens is not None:
        extract_cmd.extend(["--pool_max_tokens", str(args.pool_max_tokens)])
    if args.extract_bf16:
        extract_cmd.append("--bf16")
    if args.extract_fp16:
        extract_cmd.append("--fp16")
    if args.local_files_only:
        extract_cmd.append("--local_files_only")
    run_command(extract_cmd, repo_root)

    benchmark_output_root = repo_root / args.benchmark_output_root
    before_dirs = set()
    if benchmark_output_root.exists():
        before_dirs = set([p for p in benchmark_output_root.iterdir() if p.is_dir()])

    benchmark_cmd = [
        args.python_executable,
        str(repo_root / args.benchmark_script),
        "--embedding_dirs", embedding_dir_name,
        "--eval_data_dir", args.eval_data_dir,
        "--train_split", args.benchmark_train_split,
        "--test_split", args.benchmark_test_split,
        "--max_events", str(args.benchmark_max_events),
        "--truncate_side", args.benchmark_truncate_side,
        "--batch_size", str(args.benchmark_batch_size),
        "--eval_batch_size", str(args.benchmark_eval_batch_size),
        "--epochs", str(args.benchmark_epochs),
        "--lr", str(args.benchmark_lr),
        "--weight_decay", str(args.benchmark_weight_decay),
        "--dropout", str(args.benchmark_dropout),
        "--hidden_dim", str(args.benchmark_hidden_dim),
        "--early_stop_patience", str(args.benchmark_classifier_patience),
        "--num_workers", str(args.benchmark_num_workers),
        "--seed", str(args.benchmark_seed),
        "--device", args.benchmark_device,
        "--output_dir", args.benchmark_output_root,
    ]
    if args.benchmark_tasks:
        benchmark_cmd.extend(["--tasks"] + list(args.benchmark_tasks))
    run_command(benchmark_cmd, repo_root)

    run_dir = benchmark_run_dir(benchmark_output_root, before_dirs)
    summary_path = run_dir / "summary.csv"
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise RuntimeError("Benchmark summary.csv is empty: %s" % summary_path)
    if args.selection_metric not in summary_df.columns:
        raise KeyError("Selection metric %r not found in %s" % (args.selection_metric, summary_path))
    metric_value = float(summary_df.iloc[0][args.selection_metric])
    return {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "benchmark_run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "metric_name": args.selection_metric,
        "metric_value": metric_value,
    }


def get_raw_online(online, is_ddp):
    m = online.module if is_ddp else online
    return getattr(m, "_orig_mod", m)


def maybe_run_eval(
    *,
    repo_root,
    args,
    output_dir,
    online,
    predictor,
    mask_embedding,
    optimizer,
    scheduler,
    global_step,
    global_micro_step,
    best_metric,
    best_record,
    bad_periods,
    eval_history,
    eval_wandb_run,
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    stage2.save_checkpoint(
        output_dir,
        global_step,
        global_micro_step,
        online,
        predictor,
        mask_embedding,
        optimizer,
        scheduler,
        args,
    )
    ckpt_path = output_dir / ("checkpoint-%d" % global_micro_step)
    record = evaluate_checkpoint(repo_root, args, ckpt_path, "checkpoint-%d" % global_micro_step)
    record["step"] = global_micro_step
    record["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    eval_history.append(record)

    metric_value = record["metric_value"]
    improved = (best_metric is None) or (metric_value > best_metric + args.min_delta)
    if improved:
        best_metric = metric_value
        best_record = record
        bad_periods = 0
        logger.info("New best metric: %s=%.6f at micro_step=%d",
                    args.selection_metric, metric_value, global_micro_step)
    else:
        bad_periods += 1
        logger.info(
            "No improvement at micro_step=%d: current=%.6f best=%.6f bad_periods=%d/%d",
            global_micro_step,
            metric_value,
            best_metric,
            bad_periods,
            args.early_stop_patience,
        )
    log_eval_to_wandb(eval_wandb_run, record, best_record)
    return best_metric, best_record, bad_periods


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
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

    eval_wandb_run = maybe_init_eval_wandb(args, output_dir) if rank == 0 else None

    use_wandb = (args.wandb_project is not None) and (rank == 0)
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or ("stage2-%s" % datetime.now().strftime("%m%d-%H%M")),
            tags=args.wandb_tags,
            config=vars(args),
        )

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
        collate_fn=stage2.partial(
            stage2.collate_fn,
            eos_token_id=eos_token_id,
            num_mask_events=args.num_mask_events,
        ),
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
    eval_history = []
    next_eval_target = ((global_micro_step // args.eval_every_steps) + 1) * args.eval_every_steps
    TIMING_WARMUP = 10
    total_train_time = 0.0
    tokens_per_step = None
    early_stopped = False

    online.train()
    predictor.train()

    try:
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
                        if use_wandb:
                            wandb_run.log({
                                "train/loss": run_loss / n,
                                "train/loss_cpt": run_cpt / n,
                                "train/loss_jepa": run_jepa / n,
                                "train/loss_red": run_red / n,
                                "train/lr": lr_now,
                                "train/tok_per_s": tok_s,
                            }, step=global_step)
                        run_loss = run_cpt = run_jepa = run_red = 0.0
                        micro_steps = 0

                    should_eval_now = global_micro_step >= next_eval_target
                    if should_eval_now:
                        if is_ddp:
                            dist.barrier()
                        if rank == 0:
                            logger.info("Starting synchronous periodic evaluation at micro_step=%d", global_micro_step)
                            best_metric, best_record, bad_periods = maybe_run_eval(
                                repo_root=repo_root,
                                args=args,
                                output_dir=output_dir,
                                online=online,
                                predictor=predictor,
                                mask_embedding=mask_embedding,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                global_micro_step=global_micro_step,
                                best_metric=best_metric,
                                best_record=best_record,
                                bad_periods=bad_periods,
                                eval_history=eval_history,
                                eval_wandb_run=eval_wandb_run,
                            )
                            write_state(state_path, {
                                "selection_metric": args.selection_metric,
                                "best_metric": best_metric,
                                "best_record": best_record,
                                "bad_periods": bad_periods,
                                "global_step": global_step,
                                "global_micro_step": global_micro_step,
                                "eval_history": eval_history,
                            })
                            if bad_periods >= args.early_stop_patience:
                                early_stopped = True
                        if is_ddp:
                            flag = torch.tensor([1 if early_stopped else 0], device=device)
                            dist.broadcast(flag, src=0)
                            early_stopped = bool(flag.item())
                            dist.barrier()
                        next_eval_target += args.eval_every_steps
                        online.train()
                        predictor.train()
                        if early_stopped:
                            break

            if early_stopped:
                break

        if rank == 0:
            final_dir = output_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            raw = get_raw_online(online, is_ddp)
            raw.save_pretrained(final_dir)
            torch.save((predictor.module if is_ddp else predictor).state_dict(), final_dir / "predictor.pt")
            torch.save(mask_embedding.state_dict(), final_dir / "mask_embedding.pt")
            tokenizer.save_pretrained(final_dir)
            logger.info("Final model saved -> %s", final_dir)
            logger.info("Total training time (excl. first %d steps): %.2fm", TIMING_WARMUP, total_train_time / 60.0)
            record = evaluate_checkpoint(repo_root, args, final_dir, "final")
            record["step"] = global_micro_step
            record["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            eval_history.append(record)
            metric_value = record["metric_value"]
            if (best_metric is None) or (metric_value > best_metric + args.min_delta):
                best_metric = metric_value
                best_record = record
            log_eval_to_wandb(eval_wandb_run, record, best_record)
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
                logger.info("Best checkpoint: %s (%s=%.6f)",
                            best_record["checkpoint_path"], best_record["metric_name"], best_record["metric_value"])
            if use_wandb:
                wandb_run.finish()
            if eval_wandb_run is not None:
                eval_wandb_run.finish()
    finally:
        if is_ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
