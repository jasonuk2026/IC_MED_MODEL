#!/usr/bin/env python3

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run stage2 training with periodic event-embedding extraction, downstream benchmark, and early stopping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train_output_dir", required=True,
                   help="Stage2 output_dir. Checkpoints are monitored here.")
    p.add_argument("--eval_every_steps", type=int, required=True,
                   help="Counted in actual micro-batch steps. Injected into train_stage2.py as --save_micro_steps.")
    p.add_argument("--early_stop_patience", type=int, default=3,
                   help="Stop if metric does not improve for this many consecutive evaluation periods.")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Minimum metric improvement required to reset patience.")
    p.add_argument("--selection_metric", default="test_macro_auc",
                   help="Column name from benchmark summary.csv used for early stopping.")
    p.add_argument("--monitor_interval_sec", type=int, default=30)
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--train_launcher", default="python", choices=["python", "torchrun"])
    p.add_argument("--nproc_per_node", type=int, default=1)
    p.add_argument("--stage2_script", default="train_stage2.py")
    p.add_argument("--extract_script", default="01_gen_meta/extract_event_emb.py")
    p.add_argument("--benchmark_script", default="benchmark_foundation_simple_classifier.py")

    p.add_argument("--extract_unique_events_path", default="data/01_outputs/unique_events.parquet")
    p.add_argument("--extract_output_prefix", default="stage2_periodic_eval")
    p.add_argument("--encoder", default="qwen3")
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
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
    p.add_argument("--local_files_only", action="store_true")

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
    p.add_argument("--wandb_project", default=None,
                   help="Optional W&B project for periodic evaluation metrics logged by this wrapper.")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)

    p.add_argument("--terminate_timeout_sec", type=int, default=120)
    p.add_argument("--state_path", default=None,
                   help="Optional JSON path for evaluation history. Defaults to <train_output_dir>/periodic_eval_state.json")
    p.add_argument("stage2_args", nargs=argparse.REMAINDER,
                   help="Arguments forwarded to train_stage2.py. Prefix with -- in the shell command.")
    return p.parse_args()


def normalize_forwarded_args(args_list):
    if args_list and args_list[0] == "--":
        return args_list[1:]
    return args_list


def checkpoint_step_from_name(path):
    name = path.name
    if not name.startswith("checkpoint-"):
        return None
    suffix = name.split("-", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def sorted_checkpoints(output_dir):
    ckpts = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        step = checkpoint_step_from_name(path)
        if step is not None:
            trainer_state = path / "trainer_state.pt"
            config_json = path / "config.json"
            ready_marker = path / ".ready"
            if not (trainer_state.exists() and config_json.exists() and ready_marker.exists()):
                logger.info("Skipping incomplete checkpoint directory: %s", path)
                continue
            ckpts.append((step, path))
    ckpts.sort()
    return ckpts


def run_command(cmd, cwd):
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


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

    if not (embedding_output_dir / "embeddings.npy").exists():
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
    else:
        logger.info("Embedding cache exists, skipping extraction: %s", embedding_output_dir)

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
    if not summary_path.exists():
        raise RuntimeError("Benchmark output missing summary.csv: %s" % summary_path)
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise RuntimeError("Benchmark summary.csv is empty: %s" % summary_path)
    if args.selection_metric not in summary_df.columns:
        raise KeyError("Selection metric %r not found in %s" % (args.selection_metric, summary_path))

    metric_value = float(summary_df.iloc[0][args.selection_metric])
    logger.info(
        "Evaluation complete for %s: %s=%.6f (summary=%s)",
        label,
        args.selection_metric,
        metric_value,
        summary_path,
    )
    return {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "embedding_dir_name": embedding_dir_name,
        "benchmark_run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "metric_name": args.selection_metric,
        "metric_value": metric_value,
    }


def terminate_process_group(proc, timeout_sec):
    if proc.poll() is not None:
        return
    logger.warning("Stopping training process group ...")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)

    logger.warning("Process group did not exit after SIGTERM; sending SIGKILL")
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def write_state(state_path, payload):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(state_path), "w") as f:
        json.dump(payload, f, indent=2)


def maybe_init_wandb(args, train_output_dir):
    if args.wandb_project is None:
        return None
    import wandb
    run_name = args.wandb_run_name or ("stage2-periodic-eval-%s" % train_output_dir.name)
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        tags=args.wandb_tags,
        config=vars(args),
        dir=str(train_output_dir),
        settings=wandb.Settings(console="wrap"),
    )


def log_eval_to_wandb(wandb_run, record, best_record):
    if wandb_run is None:
        return
    summary_df = pd.read_csv(record["summary_path"])
    if summary_df.empty:
        return
    top_row = summary_df.iloc[0].to_dict()
    step = record["step"] if record["step"] is not None else None
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
    if step is None:
        wandb_run.log(payload)
    else:
        wandb_run.log(payload, step=step)

    if best_record is not None:
        wandb_run.summary["best_checkpoint_path"] = best_record["checkpoint_path"]
        wandb_run.summary["best_checkpoint_label"] = best_record["label"]
        wandb_run.summary["best_metric_name"] = best_record["metric_name"]
        wandb_run.summary["best_metric_value"] = best_record["metric_value"]
        if best_record.get("step") is not None:
            wandb_run.summary["best_checkpoint_step"] = best_record["step"]


def build_train_command(repo_root, args):
    forwarded = normalize_forwarded_args(list(args.stage2_args))
    train_output_dir = str(Path(args.train_output_dir))
    stage2_script_path = str(repo_root / args.stage2_script)

    if args.train_launcher == "torchrun":
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=%d" % args.nproc_per_node,
            stage2_script_path,
        ]
    else:
        cmd = [args.python_executable, stage2_script_path]

    cmd.extend(forwarded)
    cmd.extend([
        "--output_dir", train_output_dir,
        "--save_micro_steps", str(args.eval_every_steps),
    ])
    if args.local_files_only:
        cmd.append("--local_files_only")
    return cmd


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    train_output_dir = Path(args.train_output_dir)
    train_output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path) if args.state_path else (train_output_dir / "periodic_eval_state.json")

    train_cmd = build_train_command(repo_root, args)
    logger.info("Train output dir: %s", train_output_dir)
    logger.info("Early stopping: patience=%d min_delta=%.6f metric=%s",
                args.early_stop_patience, args.min_delta, args.selection_metric)
    wandb_run = maybe_init_wandb(args, train_output_dir)

    proc = subprocess.Popen(
        train_cmd,
        cwd=str(repo_root),
        start_new_session=True,
    )

    evaluated_steps = set()
    eval_history = []
    best_metric = None
    best_record = None
    bad_periods = 0
    early_stopped = False

    try:
        while True:
            for step, ckpt_path in sorted_checkpoints(train_output_dir):
                if step in evaluated_steps:
                    continue
                logger.info("Discovered new checkpoint for evaluation: step=%d path=%s", step, ckpt_path)
                record = evaluate_checkpoint(repo_root, args, ckpt_path, "checkpoint-%d" % step)
                record["step"] = step
                record["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                eval_history.append(record)
                evaluated_steps.add(step)

                metric_value = record["metric_value"]
                improved = (best_metric is None) or (metric_value > best_metric + args.min_delta)
                if improved:
                    best_metric = metric_value
                    best_record = record
                    bad_periods = 0
                    logger.info("New best metric: %s=%.6f at step=%d", args.selection_metric, metric_value, step)
                else:
                    bad_periods += 1
                    logger.info(
                        "No improvement at step=%d: current=%.6f best=%.6f bad_periods=%d/%d",
                        step,
                        metric_value,
                        best_metric,
                        bad_periods,
                        args.early_stop_patience,
                    )

                write_state(state_path, {
                    "train_command": train_cmd,
                    "selection_metric": args.selection_metric,
                    "best_metric": best_metric,
                    "best_record": best_record,
                    "bad_periods": bad_periods,
                    "early_stopped": early_stopped,
                    "evaluated_steps": sorted(list(evaluated_steps)),
                    "eval_history": eval_history,
                })
                log_eval_to_wandb(wandb_run, record, best_record)

                if bad_periods >= args.early_stop_patience:
                    early_stopped = True
                    logger.warning(
                        "Early stopping triggered: %d consecutive evaluation periods without improvement.",
                        bad_periods,
                    )
                    terminate_process_group(proc, args.terminate_timeout_sec)
                    break

            if early_stopped:
                break

            if proc.poll() is not None:
                logger.info("Training process exited with code %s", proc.returncode)
                break

            time.sleep(args.monitor_interval_sec)

        final_dir = train_output_dir / "final"
        if final_dir.exists() and "final" not in evaluated_steps:
            logger.info("Evaluating final model directory: %s", final_dir)
            record = evaluate_checkpoint(repo_root, args, final_dir, "final")
            record["step"] = None
            record["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            eval_history.append(record)
            evaluated_steps.add("final")
            metric_value = record["metric_value"]
            if (best_metric is None) or (metric_value > best_metric + args.min_delta):
                best_metric = metric_value
                best_record = record
            log_eval_to_wandb(wandb_run, record, best_record)

        write_state(state_path, {
            "train_command": train_cmd,
            "selection_metric": args.selection_metric,
            "best_metric": best_metric,
            "best_record": best_record,
            "bad_periods": bad_periods,
            "early_stopped": early_stopped,
            "training_returncode": proc.poll(),
            "evaluated_steps": sorted([x for x in evaluated_steps if isinstance(x, int)]),
            "eval_history": eval_history,
        })

        if best_record is not None:
            logger.info(
                "Best checkpoint: %s (%s=%.6f)",
                best_record["checkpoint_path"],
                best_record["metric_name"],
                best_record["metric_value"],
            )
            if wandb_run is not None:
                wandb_run.summary["best_checkpoint_path"] = best_record["checkpoint_path"]
                wandb_run.summary["best_checkpoint_label"] = best_record["label"]
                wandb_run.summary["best_metric_name"] = best_record["metric_name"]
                wandb_run.summary["best_metric_value"] = best_record["metric_value"]
                if best_record.get("step") is not None:
                    wandb_run.summary["best_checkpoint_step"] = best_record["step"]
        logger.info("State written to %s", state_path)

        if proc.poll() not in (0, None) and not early_stopped:
            raise subprocess.CalledProcessError(proc.poll(), train_cmd)
    finally:
        if proc.poll() is None:
            terminate_process_group(proc, args.terminate_timeout_sec)
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
