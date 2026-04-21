"""Quantitative summary analysis for multi-robot simulation results.

Reads all termination_summary.txt files under a given run directory and
produces aggregate statistics (mean, std, min, max) grouped by map_tag
and overall.

Usage:
    python -m analysis.quantitative_summary --run_dir results/quantitative/agent_3
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_summary(path: str) -> Optional[dict]:
    """Parse a termination_summary.txt file into a dict.

    Returns None if the file cannot be read or is malformed.
    """
    data = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                data[key.strip()] = val.strip()
    except OSError:
        return None

    # Type coercions
    def _float(k):
        v = data.get(k, "NA")
        try:
            return float(v)
        except (ValueError, TypeError):
            return float("nan")

    def _int(k):
        v = data.get(k, "None")
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    return {
        "map_tag":                 data.get("map_tag", "unknown"),
        "episode_index":           _int("episode_index"),
        "stopped":                 data.get("stopped", "False").lower() == "true",
        "stop_step":               _int("stop_step"),
        "stop_reason":             data.get("stop_reason", "None"),
        "free_coverage":           _float("free_coverage"),
        "covered_free_cells":      _int("covered_free_cells"),
        "total_gt_free_cells":     _int("total_gt_free_cells"),
        "belief_free_cells":       _int("belief_free_cells"),
        "cbf_total_steps":         _int("cbf_total_steps"),
        "cbf_obs_violation_rate":  _float("cbf_obs_violation_rate"),
        "cbf_avoid_violation_rate":_float("cbf_avoid_violation_rate"),
        "cbf_conn_violation_rate": _float("cbf_conn_violation_rate"),
    }


def collect_summaries(run_dir: str) -> pd.DataFrame:
    """Walk run_dir, parse every termination_summary.txt, return DataFrame."""
    records = []
    for entry in sorted(os.listdir(run_dir)):
        ep_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(ep_dir):
            continue
        summary_path = os.path.join(ep_dir, "termination_summary.txt")
        if not os.path.isfile(summary_path):
            continue
        rec = parse_summary(summary_path)
        if rec is not None:
            rec["episode_dir"] = entry
            records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "free_coverage",
    "cbf_obs_violation_rate",
    "cbf_avoid_violation_rate",
    "cbf_conn_violation_rate",
    "stop_step",
    "cbf_total_steps",
]


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean/std/min/max for NUMERIC_COLS across all rows in df."""
    rows = []
    for col in NUMERIC_COLS:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({
            "metric": col,
            "count":  len(vals),
            "mean":   vals.mean() if len(vals) else float("nan"),
            "std":    vals.std(ddof=1) if len(vals) > 1 else float("nan"),
            "min":    vals.min() if len(vals) else float("nan"),
            "max":    vals.max() if len(vals) else float("nan"),
        })
    return pd.DataFrame(rows)


def stop_reason_counts(df: pd.DataFrame) -> pd.Series:
    return df["stop_reason"].value_counts(dropna=False)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def report(df: pd.DataFrame, out_dir: str) -> None:
    out_dir = os.path.join(out_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Overall ----
    print_section("OVERALL STATISTICS")
    overall = compute_stats(df)
    print(overall.to_string(index=False))
    overall.to_csv(os.path.join(out_dir, "stats_overall.csv"), index=False)

    print("\n-- Stop reason distribution (overall) --")
    rc = stop_reason_counts(df)
    print(rc.to_string())

    # ---- Per map_tag ----
    for tag, grp in df.groupby("map_tag"):
        print_section(f"MAP TAG: {tag}  (n={len(grp)})")
        stats = compute_stats(grp)
        print(stats.to_string(index=False))
        stats.to_csv(os.path.join(out_dir, f"stats_{tag}.csv"), index=False)

        print(f"\n-- Stop reason distribution ({tag}) --")
        print(stop_reason_counts(grp).to_string())

    # ---- Per-episode table ----
    cols_to_show = [
        "episode_dir", "map_tag", "stop_reason", "stop_step",
        "free_coverage",
        "cbf_obs_violation_rate",
        "cbf_avoid_violation_rate",
        "cbf_conn_violation_rate",
    ]
    per_ep = df[[c for c in cols_to_show if c in df.columns]]
    print_section("PER-EPISODE TABLE")
    print(per_ep.to_string(index=False))
    per_ep.to_csv(os.path.join(out_dir, "per_episode.csv"), index=False)

    print(f"\n[analysis] CSVs saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quantitative summary for simulation results")
    parser.add_argument(
        "--run_dir",
        default="results/quantitative/agent_3",
        help="Directory containing episode subdirectories (default: results/quantitative/agent_3)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Root output directory (default: <run_dir>); CSVs are saved under <out_dir>/summary/",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir

    if not os.path.isdir(run_dir):
        print(f"[error] run_dir does not exist: {run_dir}")
        return

    df = collect_summaries(run_dir)
    if df.empty:
        print(f"[error] No termination_summary.txt files found under: {run_dir}")
        return

    print(f"[analysis] Found {len(df)} episodes in: {run_dir}")
    report(df, out_dir)


if __name__ == "__main__":
    main()
