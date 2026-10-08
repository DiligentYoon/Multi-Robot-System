"""Per-agent time-series plots for a single simulation episode.

Reads agent_*_log.csv from one episode directory and writes one figure per
PNG file:

  agent_<i>/obs_avoid.png            obstacle avoidance barrier
  agent_<i>/agent_avoid.png          agent-agent avoidance barrier
  agent_<i>/agent_conn.png           connectivity barrier
  agent_<i>/a_nom_vs_a_safe.png      nominal vs. safe linear acceleration
  agent_<i>/w_nom_vs_w_safe.png      nominal vs. safe angular velocity
  aggregate/<barrier>_mean.png       barrier averaged over agents
  aggregate/<barrier>_min.png        barrier minimised over agents

This is a single-case inspection tool, so the episode directory is given
directly and no aggregation across seeds or maps is performed. The seed is
already encoded in the directory name (e.g. i_shape_seed_64_034).

Usage:
    python -m analysis.plot_timeseries --ep_dir results/quantitative/agent_5/i_shape_seed_64_034
    python -m analysis.plot_timeseries --ep_dir <path> --agents 0 1 2
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode_dfs(ep_dir: str) -> dict[int, pd.DataFrame]:
    """Load all agent_i_log.csv files from an episode directory.

    Returns:
        Mapping of agent_idx -> DataFrame. Empty dict if no CSV files found.
    """
    csv_files = sorted(
        f for f in os.listdir(ep_dir) if re.match(r"agent_\d+_log\.csv", f)
    )
    result = {}
    for cf in csv_files:
        agent_idx = int(re.search(r"agent_(\d+)_log", cf).group(1))
        df = pd.read_csv(os.path.join(ep_dir, cf), skipinitialspace=True)
        df.columns = df.columns.str.strip()
        result[agent_idx] = df
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# (csv column, y-axis label, plot title, output file stem)
CBF_ROWS = [
    ("obs_avoid",   "obs_avoid (m²)",   "Obstacle avoidance barrier",   "obs_avoid"),
    ("agent_avoid", "agent_avoid (m²)", "Agent-agent avoidance barrier", "agent_avoid"),
    ("agent_conn",  "agent_conn (m²)",  "Connectivity barrier",         "agent_conn"),
]

# (nominal column, safe column, y-axis label, plot title, output file stem)
CTRL_ROWS = [
    ("a_nom", "a_safe", "Linear accel (m/s²)",
     "Nominal vs. Safe linear acceleration", "a_nom_vs_a_safe"),
    ("w_nom", "w_safe", "Angular vel (rad/s)",
     "Nominal vs. Safe angular velocity",    "w_nom_vs_w_safe"),
]

FIGSIZE = (8, 3.5)
DPI = 120


def _new_figure(title: str):
    """Create a single-panel figure with a consistent size and title."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title, fontsize=12)
    return fig, ax


def _save(fig, out_path: str) -> None:
    """Write the figure to disk and close it."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


def _save_na(out_path: str, title: str, reason: str) -> None:
    """Write a placeholder figure when the required column is missing."""
    fig, ax = _new_figure(title)
    ax.text(0.5, 0.5, reason, transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, out_path)


def plot_barrier(steps: np.ndarray, vals: np.ndarray,
                 ylabel: str, title: str, out_path: str) -> None:
    """Plot one CBF barrier trace, shading the region below zero."""
    fig, ax = _new_figure(title)
    ax.plot(steps, vals, color="tab:blue", linewidth=0.9, label=ylabel)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8,
               label="h=0 (safety boundary)")
    ax.fill_between(steps, vals, 0, where=(vals < 0),
                    color="red", alpha=0.15, label="violation")
    ax.set_xlabel("timesteps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    _save(fig, out_path)


def plot_control(steps: np.ndarray, nom_vals: np.ndarray, safe_vals: np.ndarray,
                 ylabel: str, title: str, out_path: str) -> None:
    """Plot one nominal-vs-safe control trace."""
    fig, ax = _new_figure(title)
    ax.plot(steps, nom_vals,  color="tab:blue", linewidth=0.9, label="nominal")
    ax.plot(steps, safe_vals, color="tab:red",  linewidth=0.9, label="safe (CBF)")
    ax.set_xlabel("timesteps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    _save(fig, out_path)


def plot_agent(agent_idx: int, df: pd.DataFrame,
               agent_dir: str, episode_name: str) -> None:
    steps = df["step"].to_numpy()
    prefix = f"Agent {agent_idx}"          # episode_name 제거

    for col, ylabel, title, stem in CBF_ROWS:
        out_path = os.path.join(agent_dir, f"{stem}.png")
        if col in df.columns:
            plot_barrier(steps, df[col].to_numpy(dtype=float),
                         ylabel, f"{prefix} — {title}", out_path)
        else:
            _save_na(out_path, f"{prefix} — {title}",
                     f"N/A ('{col}' not in CSV)")

    for nom_col, safe_col, ylabel, title, stem in CTRL_ROWS:
        out_path = os.path.join(agent_dir, f"{stem}.png")
        if nom_col in df.columns and safe_col in df.columns:
            plot_control(steps,
                         df[nom_col].to_numpy(dtype=float),
                         df[safe_col].to_numpy(dtype=float),
                         ylabel, f"{prefix} — {title}", out_path)
        else:
            _save_na(out_path, f"{prefix} — {title}",
                     f"N/A ('{nom_col}'/'{safe_col}' not in CSV)")


def plot_agent_axis_aggregate(agent_dfs: dict[int, pd.DataFrame],
                              agg_dir: str, episode_name: str,
                              mode: str = "mean") -> None:
    """Write one figure per barrier, aggregated across the agent axis.

    Args:
        agent_dfs:    Mapping of agent_idx -> DataFrame.
        agg_dir:      Directory the aggregate PNG files are written to.
        episode_name: Episode directory name used in figure titles.
        mode:         "mean" averages over agents at each step; "min" takes
                      the minimum, i.e. the worst-case agent.
    """
    if mode not in ("mean", "min"):
        raise ValueError(f"mode must be 'mean' or 'min', got: {mode}")

    # Use the first agent as the reference time axis.
    # This assumes all agent logs share the same step sequence.
    first_idx = sorted(agent_dfs.keys())[0]
    steps = agent_dfs[first_idx]["step"].to_numpy()

    mode_name = "Agent-axis mean" if mode == "mean" else "Agent-axis minimum"

    for col, ylabel, title, stem in CBF_ROWS:
        out_path = os.path.join(agg_dir, f"{stem}_{mode}.png")
        full_title = f"{mode_name} — {title}"   

        vals_per_agent = [df[col].to_numpy(dtype=float)
                          for _, df in sorted(agent_dfs.items())
                          if col in df.columns]

        if not vals_per_agent:
            _save_na(out_path, full_title, f"N/A ('{col}' not in CSV)")
            continue

        # Logs may differ in length if an agent terminated early; truncate
        # to the shortest so the agent axis stays rectangular.
        min_len = min(len(v) for v in vals_per_agent)
        value_mat = np.stack([v[:min_len] for v in vals_per_agent], axis=0)

        agg_vals = (np.mean(value_mat, axis=0) if mode == "mean"
                    else np.min(value_mat, axis=0))

        plot_barrier(steps[:min_len], agg_vals,
                     f"{ylabel} ({mode})", full_title, out_path)


def plot_episode(ep_dir: str, plot_dir: str,
                 agents: list[int] | None = None) -> None:
    """Write all figures for one episode, one plot per PNG file.

    Args:
        ep_dir:   Path to the episode directory containing agent CSV logs.
        plot_dir: Directory the PNG files are written to.
        agents:   Agent indices to plot. None means all agents.
    """
    episode_name = os.path.basename(ep_dir)
    agent_dfs = load_episode_dfs(ep_dir)

    if not agent_dfs:
        print(f"[error] No agent_*_log.csv found in: {ep_dir}")
        return

    selected = {idx: df for idx, df in agent_dfs.items()
                if agents is None or idx in agents}

    if not selected:
        print(f"[error] None of the requested agents {agents} are present. "
              f"Available: {sorted(agent_dfs.keys())}")
        return

    print(f"[plot] Episode: {episode_name} | agents: {sorted(selected.keys())}")

    for idx, df in sorted(selected.items()):
        plot_agent(idx, df, os.path.join(plot_dir, f"agent_{idx}"), episode_name)

    agg_dir = os.path.join(plot_dir, "aggregate")
    plot_agent_axis_aggregate(selected, agg_dir, episode_name, mode="mean")
    plot_agent_axis_aggregate(selected, agg_dir, episode_name, mode="min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for plot_timeseries."""
    parser = argparse.ArgumentParser(
        description="Generate per-agent time-series plots for a single episode"
    )
    parser.add_argument(
        "--ep_dir",
        required=True,
        help="Path to one episode directory containing agent_*_log.csv "
             "(e.g. results/quantitative/agent_5/i_shape_seed_64_034)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Agent indices to plot (default: all agents)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <ep_dir>/timeseries_plots)",
    )
    args = parser.parse_args()

    ep_dir = os.path.normpath(args.ep_dir)
    if not os.path.isdir(ep_dir):
        print(f"[error] episode directory does not exist: {ep_dir}")
        return

    out_dir = args.out_dir or os.path.join(ep_dir, "timeseries_plots")
    plot_episode(ep_dir, out_dir, agents=args.agents)


if __name__ == "__main__":
    main()