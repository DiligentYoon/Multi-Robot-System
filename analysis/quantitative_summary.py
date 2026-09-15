"""Quantitative summary: two-stage aggregation by agent count and seed.

Stage 1: average episodes within each (agent[, map], seed) group.
Stage 2: mean / std across seeds.

Usage:
    python -m analysis.quantitative_summary --root results/quantitative
    python -m analysis.quantitative_summary --root results/quantitative --per_map
"""

import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

DIR_RE   = re.compile(r"^(?P<map_tag>.+)_seed_(?P<seed>\d+)_(?P<idx>\d+)$")
AGENT_RE = re.compile(r"agent_(?P<n>\d+)")

NUMERIC_COLS = [
    "free_coverage",
    "cbf_feasible_rate",
    "cbf_obs_violation_rate",
    "cbf_avoid_violation_rate",
    "cbf_conn_violation_rate",
    "stop_step",
    "cbf_total_steps",
]

RATE_COLS = ["success_rate", "obstacle_collision_rate",
             "robot_collision_rate", "frozen_rate", "timeout_rate"]

REASON_MAP = {
    "goal":               "success_rate",
    "obstacle_collision": "obstacle_collision_rate",
    "robot_collision":    "robot_collision_rate",
    "frozen":             "frozen_rate",
    "timeout":            "timeout_rate",
}


def parse_summary(path: str) -> Optional[dict]:
    """Parse one termination_summary.txt into a dict, or None if unreadable."""
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

    def _float(k):
        try:
            return float(data.get(k, "NA"))
        except (ValueError, TypeError):
            return float("nan")

    def _int(k):
        try:
            return int(data.get(k, "None"))
        except (ValueError, TypeError):
            return None

    rec = {
        "map_tag":       data.get("map_tag", "unknown"),
        "seed":          _int("seed"),
        "num_agent":     _int("num_agent"),
        "episode_index": _int("episode_index"),
        "stopped":       data.get("stopped", "False").lower() == "true",
        "stop_step":     _int("stop_step"),
        "stop_reason":   data.get("stop_reason", "None"),
    }
    for c in NUMERIC_COLS:
        if c not in rec:
            rec[c] = _float(c)
    rec["covered_free_cells"]  = _int("covered_free_cells")
    rec["total_gt_free_cells"] = _int("total_gt_free_cells")
    return rec


def collect(root: str) -> pd.DataFrame:
    """Walk root/agent_*/<episode_dir>/termination_summary.txt and collect all."""
    records = []
    for agent_dir in sorted(os.listdir(root)):
        agent_path = os.path.join(root, agent_dir)
        if not os.path.isdir(agent_path):
            continue
        m_agent = AGENT_RE.search(agent_dir)
        agent_fallback = int(m_agent.group("n")) if m_agent else None

        for ep_dir in sorted(os.listdir(agent_path)):
            ep_path = os.path.join(agent_path, ep_dir)
            summary = os.path.join(ep_path, "termination_summary.txt")
            if not os.path.isfile(summary):
                continue
            rec = parse_summary(summary)
            if rec is None:
                continue

            # Fall back to directory names when the fields are absent
            # from the summary file (older runs).
            if rec["num_agent"] is None:
                rec["num_agent"] = agent_fallback
            m_dir = DIR_RE.match(ep_dir)
            if rec["seed"] is None and m_dir:
                rec["seed"] = int(m_dir.group("seed"))
            if rec["map_tag"] in ("unknown", "") and m_dir:
                rec["map_tag"] = m_dir.group("map_tag")

            rec["episode_dir"] = ep_dir
            records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Expand stop_reason into one-hot rate columns
    for col in RATE_COLS:
        df[col] = 0.0
    for reason, col in REASON_MAP.items():
        df.loc[df["stop_reason"] == reason, col] = 1.0
    return df


# ---------------------------------------------------------------------------
# Two-stage aggregation
# ---------------------------------------------------------------------------

METRICS = NUMERIC_COLS + RATE_COLS


def stage1_per_seed(df: pd.DataFrame, group_keys) -> pd.DataFrame:
    """Average episodes within each (agent[, map], seed) group."""
    agg = {m: "mean" for m in METRICS if m in df.columns}
    out = df.groupby(group_keys, dropna=False).agg(agg).reset_index()
    out["n_episodes"] = (df.groupby(group_keys, dropna=False)
                           .size().reset_index(drop=True))
    return out


def stage2_across_seeds(per_seed: pd.DataFrame, group_keys) -> pd.DataFrame:
    """Mean / std across seeds, plus seed and episode counts."""
    rows = []
    for keys, grp in per_seed.groupby(group_keys, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_keys, keys))
        base["n_seeds"]    = len(grp)
        base["n_episodes"] = int(grp["n_episodes"].sum())
        for m in METRICS:
            if m not in grp.columns:
                continue
            v = pd.to_numeric(grp[m], errors="coerce").dropna()
            base[f"{m}_mean"] = v.mean() if len(v) else np.nan
            base[f"{m}_std"]  = v.std(ddof=1) if len(v) > 1 else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def format_pm(df: pd.DataFrame, group_keys, metrics=None) -> pd.DataFrame:
    """Render metrics as 'mean ± std' strings."""
    metrics = metrics or METRICS
    out = df[group_keys + ["n_seeds", "n_episodes"]].copy()
    for m in metrics:
        mc, sc = f"{m}_mean", f"{m}_std"
        if mc not in df.columns:
            continue
        out[m] = [
            "NA" if pd.isna(mu) else
            (f"{mu:.3f}" if pd.isna(sd) else f"{mu:.3f} ± {sd:.3f}")
            for mu, sd in zip(df[mc], df[sc])
        ]
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def report(df: pd.DataFrame, out_dir: str, per_map: bool):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Primary: by agent count, maps pooled ----
    per_seed = stage1_per_seed(df, ["num_agent", "seed"])
    final    = stage2_across_seeds(per_seed, ["num_agent"])

    section("PER-SEED AGGREGATE (Stage 1)")
    print(per_seed.round(4).to_string(index=False))
    per_seed.to_csv(os.path.join(out_dir, "per_seed.csv"), index=False)

    section("BY AGENT COUNT - mean +/- std over seeds (Stage 2)")
    print(format_pm(final, ["num_agent"]).to_string(index=False))
    final.to_csv(os.path.join(out_dir, "by_agent.csv"), index=False)

    if (final["n_seeds"] < 3).any():
        print("\n[warn] Some groups have fewer than 3 seeds. "
              "Treat std as indicative only (insufficient degrees of freedom).")

    # ---- Secondary: agent x map, for trend inspection ----
    if per_map:
        per_seed_map = stage1_per_seed(df, ["num_agent", "map_tag", "seed"])
        final_map    = stage2_across_seeds(per_seed_map, ["num_agent", "map_tag"])
        section("BY AGENT x MAP (secondary - trend inspection)")
        print(format_pm(final_map, ["num_agent", "map_tag"]).to_string(index=False))
        final_map.to_csv(os.path.join(out_dir, "by_agent_map.csv"), index=False)

    # ---- Termination reason distribution ----
    section("STOP REASON DISTRIBUTION")
    ct = pd.crosstab([df["num_agent"], df["seed"]], df["stop_reason"])
    print(ct.to_string())
    ct.to_csv(os.path.join(out_dir, "stop_reason_counts.csv"))

    # ---- stop_step restricted to successful episodes ----
    succ = df[df["stop_reason"] == "goal"]
    if not succ.empty:
        section("stop_step - successful (goal) episodes only")
        ps = stage1_per_seed(succ, ["num_agent", "seed"])
        fs = stage2_across_seeds(ps, ["num_agent"])
        print(format_pm(fs, ["num_agent"], ["stop_step"]).to_string(index=False))
        fs.to_csv(os.path.join(out_dir, "by_agent_success_only.csv"), index=False)
    else:
        print("\n[info] No episodes ended with 'goal'; "
              "skipping the conditional stop_step table.")

    # ---- Raw per-episode table ----
    df.to_csv(os.path.join(out_dir, "per_episode.csv"), index=False)
    print(f"\n[analysis] CSVs saved to: {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description="Quantitative summary aggregated by agent count and seed")
    p.add_argument("--root", default="results/quantitative",
                   help="Root directory containing the agent_* subdirectories")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: <root>/summary)")
    p.add_argument("--per_map", action="store_true",
                   help="Also emit the agent x map secondary table")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        print(f"[error] root does not exist: {args.root}")
        return

    df = collect(args.root)
    if df.empty:
        print(f"[error] No termination_summary.txt found under: {args.root}")
        return

    missing = df["seed"].isna() | df["num_agent"].isna()
    if missing.any():
        print(f"[warn] Dropping {int(missing.sum())} records "
              f"with unresolved seed/num_agent")
        df = df[~missing]

    df["num_agent"] = df["num_agent"].astype(int)
    df["seed"]      = df["seed"].astype(int)

    print(f"[analysis] {len(df)} episodes | "
          f"agents={sorted(df['num_agent'].unique())} | "
          f"seeds={sorted(df['seed'].unique())} | "
          f"maps={sorted(df['map_tag'].unique())}")

    report(df, args.out_dir or os.path.join(args.root, "summary"), args.per_map)


if __name__ == "__main__":
    main()