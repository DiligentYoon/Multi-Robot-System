"""Quantitative summary: two-stage aggregation by agent count and seed.

Stage 1: average episodes within each (agent[, map], seed) group.
Stage 2: mean / std across seeds.

Continuous metrics (coverage, stop_step, CBF rates) are reported as
mean +/- std across seeds. Episode outcomes (success, collisions, frozen,
timeout) are binomial and are reported instead as counts with Wilson score
intervals, which stay inside [0, 1] and behave sensibly at 0% and 100%.

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
    "conn_margin_min",
    "conn_margin_mean",
    "conn_margin_p05",
    "fce_mean",
    "fce_p95",
    "fce_max",
    "fce_n_updates",
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

CSV_RE = re.compile(r"^agent_(?P<idx>\d+)_log\.csv$")

# Barrier columns to summarise from the per-agent CSV logs.
# Each yields <name>_min / <name>_mean / <name>_p05 after the agent axis
# has been reduced with a minimum at every step.
BARRIER_COLS = [("agent_conn", "conn_margin")]

# Control columns used for the Filter Correction Effort (FCE) metric.
FCE_COLS = ("a_nom", "a_safe", "w_nom", "w_safe")


def load_control_effort(ep_dir: str, a_max: float = 0.1, w_max: float = 1.0) -> dict:
    """Compute Filter Correction Effort (FCE) from the per-agent CSV logs.

    Inputs are normalised by their actuation limits so that linear and
    angular components are commensurable:

        u_bar = [a / a_max, w / w_max]
        FCE   = mean over all agent-wise control updates of
                ||u_bar_safe - u_bar_nom||_2

    Averaging over control updates (rather than summing) makes the metric
    independent of episode length, so it stays comparable across runs that
    terminate at different steps.

    Returns:
        Mapping with the mean, the 95th percentile and the maximum of the
        per-update correction, plus the number of updates. NaN when the logs
        are missing or the columns are absent.
    """
    out = {"fce_mean": float("nan"), "fce_p95": float("nan"),
           "fce_max": float("nan"), "fce_n_updates": float("nan")}

    if not (a_max and w_max):
        return out

    try:
        csv_files = [f for f in os.listdir(ep_dir) if CSV_RE.match(f)]
    except OSError:
        return out
    if not csv_files:
        return out

    diffs = []
    for cf in sorted(csv_files):
        try:
            df = pd.read_csv(os.path.join(ep_dir, cf), skipinitialspace=True)
        except (OSError, pd.errors.ParserError):
            continue
        df.columns = df.columns.str.strip()
        if not all(c in df.columns for c in FCE_COLS):
            continue

        da = (df["a_safe"].to_numpy(dtype=float)
              - df["a_nom"].to_numpy(dtype=float)) / a_max
        dw = (df["w_safe"].to_numpy(dtype=float)
              - df["w_nom"].to_numpy(dtype=float)) / w_max
        diffs.append(np.sqrt(da * da + dw * dw))

    if not diffs:
        return out

    # Agents may log different numbers of steps, so pool the updates rather
    # than stacking them into a rectangular array.
    all_diff = np.concatenate(diffs)
    all_diff = all_diff[np.isfinite(all_diff)]
    if all_diff.size == 0:
        return out

    out["fce_mean"]      = float(np.mean(all_diff))
    out["fce_p95"]       = float(np.percentile(all_diff, 95))
    out["fce_max"]       = float(np.max(all_diff))
    out["fce_n_updates"] = float(all_diff.size)
    return out


def load_barrier_margins(ep_dir: str) -> dict:
    """Summarise barrier margins from the per-agent CSV logs of one episode.

    For each barrier column the agent axis is reduced with a minimum at every
    step (the worst-off agent), and that series is then reduced over time in
    three ways: its minimum (worst instant in the episode), its mean (typical
    margin) and its 5th percentile (worst instant, robust to single-step
    spikes).

    Returns:
        Mapping of metric name -> float. NaN when the logs are missing or the
        column is absent, so that episodes without CSVs still aggregate.
    """
    out = {}
    for _, name in BARRIER_COLS:
        out[f"{name}_min"]  = float("nan")
        out[f"{name}_mean"] = float("nan")
        out[f"{name}_p05"]  = float("nan")

    try:
        csv_files = [f for f in os.listdir(ep_dir) if CSV_RE.match(f)]
    except OSError:
        return out
    if not csv_files:
        return out

    dfs = []
    for cf in sorted(csv_files):
        try:
            df = pd.read_csv(os.path.join(ep_dir, cf), skipinitialspace=True)
        except (OSError, pd.errors.ParserError):
            continue
        df.columns = df.columns.str.strip()
        dfs.append(df)
    if not dfs:
        return out

    for col, name in BARRIER_COLS:
        series = [d[col].to_numpy(dtype=float) for d in dfs if col in d.columns]
        if not series:
            continue

        # Logs may differ in length if an agent terminated early; truncate to
        # the shortest so the agent axis stays rectangular.
        min_len = min(len(s) for s in series)
        if min_len == 0:
            continue
        mat = np.stack([s[:min_len] for s in series], axis=0)

        worst_over_agents = np.nanmin(mat, axis=0)     # min over agents, per step
        if np.all(np.isnan(worst_over_agents)):
            continue

        out[f"{name}_min"]  = float(np.nanmin(worst_over_agents))
        out[f"{name}_mean"] = float(np.nanmean(worst_over_agents))
        out[f"{name}_p05"]  = float(np.nanpercentile(worst_over_agents, 5))

    return out


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


def _iter_episode_dirs(root: str):
    """Yield (episode_dir_path, agent_fallback) for both layouts:
      (a) root/agent_*/<episode_dir>/     - multiple agent counts
      (b) root/<episode_dir>/             - a single agent_* directory
    """
    root_agent = AGENT_RE.search(os.path.basename(os.path.normpath(root)))
    root_agent = int(root_agent.group("n")) if root_agent else None

    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue

        # Layout (b): this entry is itself an episode directory
        if os.path.isfile(os.path.join(path, "termination_summary.txt")):
            yield path, root_agent
            continue

        # Layout (a): this entry is an agent_* directory
        m = AGENT_RE.search(entry)
        agent_fallback = int(m.group("n")) if m else root_agent
        for ep in sorted(os.listdir(path)):
            ep_path = os.path.join(path, ep)
            if os.path.isfile(os.path.join(ep_path, "termination_summary.txt")):
                yield ep_path, agent_fallback


def collect(root: str) -> pd.DataFrame:
    """Collect every termination_summary.txt under root (either layout)."""
    records = []
    for ep_path, agent_fallback in _iter_episode_dirs(root):
        rec = parse_summary(os.path.join(ep_path, "termination_summary.txt"))
        if rec is None:
            continue

        ep_dir = os.path.basename(ep_path)

        # Fall back to directory names when the fields are absent
        # from the summary file (older runs).
        if rec["num_agent"] is None:
            rec["num_agent"] = agent_fallback
        m_dir = DIR_RE.match(ep_dir)
        if rec["seed"] is None and m_dir:
            rec["seed"] = int(m_dir.group("seed"))
        if rec["map_tag"] in ("unknown", "") and m_dir:
            rec["map_tag"] = m_dir.group("map_tag")

        if rec["map_tag"] in ("unknown", "") and m_dir:
            rec["map_tag"] = m_dir.group("map_tag")

        rec.update(load_barrier_margins(ep_path))   # from agent_*_log.csv
        rec.update(load_control_effort(ep_path))

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
    g = df.groupby(group_keys, dropna=False)
    out = g.agg(agg)
    out["n_episodes"] = g.size()      # index-aligned, not positional
    return out.reset_index()


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


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

# Continuous metrics: reported as mean +/- std across seeds.
# Outcome rates are excluded here on purpose - see OUTCOME_SPEC below.
#   kind: 'pct'   -> shown as a percentage
#         'float' -> shown with 3 decimals
#         'step'  -> shown as an integer step count
METRIC_SPEC = [
    ("free_coverage",            "Free coverage",        "pct"),
    ("cbf_feasible_rate",        "QP feasible",          "pct"),
    ("cbf_obs_violation_rate",   "CBF viol. (obstacle)", "pct"),
    ("cbf_avoid_violation_rate", "CBF viol. (agent)",    "pct"),
    ("cbf_conn_violation_rate",  "CBF viol. (conn.)",    "pct"),
    ("conn_margin_min",          "Conn margin (min)",    "float"),
    ("conn_margin_p05",          "Conn margin (p05)",    "float"),
    ("conn_margin_mean",         "Conn margin (mean)",   "float"),
    ("fce_mean", "Filter correction (FCE)", "float"),
    ("stop_step",                "Stop step",            "step"),
]

# Episode outcome categories, counted per episode in display order.
OUTCOME_SPEC = [
    ("goal",               "Success"),
    ("obstacle_collision", "Obstacle collision"),
    ("robot_collision",    "Robot collision"),
    ("frozen",             "Frozen"),
    ("timeout",            "Timeout"),
]


def _fmt_key(key, val):
    """Format a group key, rendering whole-valued floats as integers."""
    if isinstance(val, (int, np.integer)):
        return f"{key}={int(val)}"
    if isinstance(val, (float, np.floating)) and float(val).is_integer():
        return f"{key}={int(val)}"
    return f"{key}={val}"


def _fmt_cell(mu, sd, kind):
    """Render a single 'mean +/- std' cell with a fixed width."""
    if pd.isna(mu):
        return "NA"
    if kind == "pct":
        mu, sd = mu * 100.0, (sd * 100.0 if not pd.isna(sd) else sd)
        return f"{mu:6.2f}%" if pd.isna(sd) else f"{mu:6.2f} +/- {sd:5.2f} %"
    if kind == "step":
        return f"{mu:7.0f}" if pd.isna(sd) else f"{mu:7.0f} +/- {sd:5.0f}  "
    return f"{mu:7.3f}" if pd.isna(sd) else f"{mu:7.3f} +/- {sd:5.3f}  "


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion.

    Preferred over a seed-wise standard deviation for outcome rates: it is
    computed from the episode counts directly, stays inside [0, 1], and does
    not collapse to a misleading value when every seed reports the same
    outcome.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def print_metric_table(df: pd.DataFrame, group_keys, metrics=None,
                       label_width: int = 22) -> None:
    """Print continuous metrics as rows and groups as columns.

    Uses plain ASCII '+/-' so that column alignment holds in any terminal;
    the '±' glyph is double-width in many fonts and breaks padding.
    """
    metrics = metrics or [m for m in METRIC_SPEC
                          if f"{m[0]}_mean" in df.columns]

    headers = [" ".join(_fmt_key(k, row[k]) for k in group_keys)
               for _, row in df.iterrows()]

    col_w = max(24, max((len(h) for h in headers), default=0) + 2)

    line = "-" * (label_width + col_w * len(headers))
    print(line)
    print(" " * label_width + "".join(h.ljust(col_w) for h in headers))
    print(line)

    for key, label, kind in metrics:
        mc, sc = f"{key}_mean", f"{key}_std"
        if mc not in df.columns:
            continue
        cells = [_fmt_cell(row[mc], row.get(sc, np.nan), kind)
                 for _, row in df.iterrows()]
        print(label.ljust(label_width) + "".join(c.ljust(col_w) for c in cells))

    print(line)
    counts = [f"{int(r['n_seeds'])} seeds / {int(r['n_episodes'])} eps"
              for _, r in df.iterrows()]
    print("(sample size)".ljust(label_width) + "".join(c.ljust(col_w) for c in counts))
    print(line)


def print_outcome_table(df: pd.DataFrame, group_keys,
                        label_width: int = 22) -> None:
    """Print episode outcome counts with proportions and Wilson intervals."""
    groups = list(df.groupby(group_keys, dropna=False))

    headers = []
    for keys, _ in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        headers.append(" ".join(_fmt_key(k, v) for k, v in zip(group_keys, keys)))

    col_w = max(32, max((len(h) for h in headers), default=0) + 2)
    line = "-" * (label_width + col_w * len(headers))

    print(line)
    print(" " * label_width + "".join(h.ljust(col_w) for h in headers))
    print(line)

    for reason, label in OUTCOME_SPEC:
        cells = []
        for _, grp in groups:
            n = len(grp)
            k = int((grp["stop_reason"] == reason).sum())
            lo, hi = wilson_ci(k, n)
            cells.append(f"{k:3d}/{n:<3d} {k / n * 100:6.2f}% "
                         f"[{lo * 100:5.1f},{hi * 100:5.1f}]")
        print(label.ljust(label_width) + "".join(c.ljust(col_w) for c in cells))

    print(line)

    known = {r for r, _ in OUTCOME_SPEC}
    other_counts = [int((~grp["stop_reason"].isin(known)).sum())
                    for _, grp in groups]
    if any(c > 0 for c in other_counts):
        cells = [f"{c:3d}/{len(grp):<3d}"
                 for c, (_, grp) in zip(other_counts, groups)]
        print("(other / unknown)".ljust(label_width)
              + "".join(c.ljust(col_w) for c in cells))
        print(line)

    print("  Intervals are 95% Wilson score intervals over episodes.")


def print_per_seed_table(per_seed: pd.DataFrame, group_keys) -> None:
    """Stage-1 table: one row per (group, seed), compact numeric columns."""
    show = [("free_coverage", "cover"), ("success_rate", "succ"),
            ("frozen_rate", "froz"), ("timeout_rate", "tout"),
            ("obstacle_collision_rate", "obs"), ("robot_collision_rate", "rob"),
            ("stop_step", "steps")]
    present = [(c, n) for c, n in show if c in per_seed.columns]
    cols = group_keys + ["n_episodes"] + [c for c, _ in present]
    out = per_seed[cols].copy()
    # Assign names positionally to avoid collisions with existing columns.
    out.columns = group_keys + ["n_eps"] + [n for _, n in present]
    for _, n in present:
        out[n] = out[n].map(lambda v: f"{v:.3f}" if pd.notna(v) else "NA")
    print(out.to_string(index=False))


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
    print_per_seed_table(per_seed, ["num_agent", "seed"])
    per_seed.to_csv(os.path.join(out_dir, "per_seed.csv"), index=False)

    section("BY AGENT COUNT - mean +/- std over seeds (Stage 2)")
    print_metric_table(final, ["num_agent"])
    final.to_csv(os.path.join(out_dir, "by_agent.csv"), index=False)

    if (final["n_seeds"] < 3).any():
        print("[warn] Some groups have fewer than 3 seeds. "
              "Treat the continuous-metric std as indicative only.")

    # ---- Episode outcomes: counts + Wilson intervals ----
    section("EPISODE OUTCOMES BY AGENT COUNT")
    print_outcome_table(df, ["num_agent"])

    # ---- Secondary: agent x map ----
    if per_map:
        per_seed_map = stage1_per_seed(df, ["num_agent", "map_tag", "seed"])
        final_map    = stage2_across_seeds(per_seed_map, ["num_agent", "map_tag"])
        section("BY AGENT x MAP (secondary - trend inspection)")
        print_metric_table(final_map, ["num_agent", "map_tag"])
        final_map.to_csv(os.path.join(out_dir, "by_agent_map.csv"), index=False)

        section("EPISODE OUTCOMES BY AGENT x MAP")
        print_outcome_table(df, ["num_agent", "map_tag"])

    # ---- Termination reason distribution ----
    section("STOP REASON DISTRIBUTION")
    ct = pd.crosstab([df["num_agent"], df["seed"]], df["stop_reason"])
    ct["total"] = ct.sum(axis=1)
    print(ct.to_string())
    ct.to_csv(os.path.join(out_dir, "stop_reason_counts.csv"))

    # ---- stop_step restricted to successful episodes ----
    succ = df[df["stop_reason"] == "goal"]
    if not succ.empty:
        section("stop_step - successful (goal) episodes only")
        ps = stage1_per_seed(succ, ["num_agent", "seed"])
        fs = stage2_across_seeds(ps, ["num_agent"])
        print_metric_table(fs, ["num_agent"],
                           metrics=[("stop_step", "Stop step", "step")])
        fs.to_csv(os.path.join(out_dir, "by_agent_success_only.csv"), index=False)
    else:
        print("\n[info] No episodes ended with 'goal'; "
              "skipping the conditional stop_step table.")

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
                   help="Also emit the agent x map secondary tables")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        print(f"[error] root does not exist: {args.root}")
        return

    df = collect(args.root)
    if df.empty:
        print(f"[error] No termination_summary.txt found under: {args.root}\n"
              f"        Expected either <root>/agent_*/<episode>/ "
              f"or <root>/<episode>/")
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