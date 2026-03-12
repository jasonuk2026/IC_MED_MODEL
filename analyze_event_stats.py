"""
Summarize n_events_in_history statistics across all first-diagnosis tasks.

Reads the CSVs produced by extract_new_diagnosis.py and prints:
  - Per-task distribution of input event counts (overall + by label + by split)
  - A compact cross-task comparison table

Usage:
    python analyze_event_stats.py
    python analyze_event_stats.py --input_dir ./data/llm_inputs/new_diagnosis
    python analyze_event_stats.py --strategy last
"""

import argparse
import glob
import os

import pandas as pd


PERCENTILES = [0, 10, 25, 50, 75, 90, 95, 100]


def pct_label(p):
    if p == 0:   return "min"
    if p == 100: return "max"
    return f"p{p}"


def describe(series: pd.Series) -> pd.Series:
    stats = {pct_label(p): series.quantile(p / 100) for p in PERCENTILES}
    stats["mean"] = series.mean()
    stats["std"]  = series.std()
    stats["n"]    = len(series)
    return pd.Series(stats)


def load_all(input_dir: str, strategy: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, f"*_{strategy}.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=["patient_id", "task", "split", "label", "n_events_in_history"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./data/llm_inputs/new_diagnosis")
    parser.add_argument("--strategy",  default="all", choices=["first", "last", "all"],
                        help="Which prediction strategy CSVs to read.")
    args = parser.parse_args()

    df = load_all(args.input_dir, args.strategy)
    col = "n_events_in_history"

    # ------------------------------------------------------------------ #
    # 1. Cross-task summary table
    # ------------------------------------------------------------------ #
    print_section("Cross-task summary  (n_events_in_history)")
    summary = df.groupby("task")[col].apply(describe).unstack()
    summary = summary[["n", "mean", "std", "min", "p25", "p50", "p75", "p90", "p95", "max"]]
    summary["n"] = summary["n"].astype(int)
    with pd.option_context("display.float_format", "{:.1f}".format, "display.max_columns", 20, "display.width", 120):
        print(summary.to_string())

    # ------------------------------------------------------------------ #
    # 2. Per-task breakdown by label (positive vs negative)
    # ------------------------------------------------------------------ #
    print_section("Per-task breakdown by label  (median | mean)")
    rows = []
    for task, g in df.groupby("task"):
        for label_val, lg in g.groupby("label"):
            rows.append({
                "task":   task,
                "label":  "positive" if label_val else "negative",
                "n":      len(lg),
                "median": lg[col].median(),
                "mean":   lg[col].mean(),
                "p90":    lg[col].quantile(0.90),
            })
    tbl = pd.DataFrame(rows).set_index(["task", "label"])
    with pd.option_context("display.float_format", "{:.1f}".format, "display.width", 100):
        print(tbl.to_string())

    # ------------------------------------------------------------------ #
    # 3. Per-task breakdown by split (train / val / test)
    # ------------------------------------------------------------------ #
    print_section("Per-task breakdown by split  (median | mean)")
    rows = []
    for task, g in df.groupby("task"):
        for split_val, sg in g.groupby("split"):
            rows.append({
                "task":   task,
                "split":  split_val,
                "n":      len(sg),
                "median": sg[col].median(),
                "mean":   sg[col].mean(),
                "p90":    sg[col].quantile(0.90),
            })
    tbl = pd.DataFrame(rows).set_index(["task", "split"])
    with pd.option_context("display.float_format", "{:.1f}".format, "display.width", 100):
        print(tbl.to_string())

    # ------------------------------------------------------------------ #
    # 4. Truncation impact: how many rows hit the max_events cap?
    # ------------------------------------------------------------------ #
    print_section("Rows at max-events cap (200)")
    capped = df[df[col] >= 200].groupby("task").size().rename("n_capped")
    total  = df.groupby("task").size().rename("n_total")
    cap_tbl = pd.concat([total, capped], axis=1).fillna(0)
    cap_tbl["n_capped"] = cap_tbl["n_capped"].astype(int)
    cap_tbl["pct_capped"] = 100 * cap_tbl["n_capped"] / cap_tbl["n_total"]
    with pd.option_context("display.float_format", "{:.1f}".format):
        print(cap_tbl.to_string())

    print()


if __name__ == "__main__":
    main()
