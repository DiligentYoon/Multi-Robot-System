"""Per-agent time-series plots for multi-robot simulation results.

For each episode directory, reads agent_i_log.csv files and produces
5-row time-series figures (one PNG per agent):
  row 0 — obs_avoid barrier
  row 1 — agent_avoid barrier
  row 2 — agent_conn barrier
  row 3 — a_nom vs a_safe (linear acceleration)
  row 4 — w_nom vs w_safe (angular velocity)

Saved to: <out_dir>/<episode_dir>/timeseries_plots/agent_<i>_timeseries.png

Usage:
    python -m analysis.plot_timeseries --run_dir results/quantitative/agent_3
    python -m analysis.plot_timeseries --run_dir results/quantitative/agent_3 --episode i_shape_011
    python -m analysis.plot_timeseries --run_dir results/quantitative/agent_3 --agents 0 1 2
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

CBF_ROWS = [
    ("obs_avoid",   "obs_avoid (m²)",   "Obstacle avoidance barrier"),
    ("agent_avoid", "agent_avoid (m²)", "Agent-agent avoidance barrier"),
    ("agent_conn",  "agent_conn (m²)",  "Connectivity barrier"),
]

CTRL_ROWS = [
    ("a_nom", "a_safe", "Linear accel (m/s²)", "Nominal vs. Safe linear acceleration"),
    ("w_nom", "w_safe", "Angular vel (rad/s)",  "Nominal vs. Safe angular velocity"),
]


def _plot_barrier_row(ax: plt.Axes, steps: np.ndarray, vals: np.ndarray,
                      ylabel: str, title: str) -> None:
    ax.plot(steps, vals, color="tab:blue", linewidth=0.9, label=ylabel)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8,
               label="h=0 (safety boundary)")
    ax.fill_between(steps, vals, 0, where=(vals < 0),
                    color="red", alpha=0.15, label="violation")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


def _plot_control_row(ax: plt.Axes, steps: np.ndarray,
                      nom_vals: np.ndarray, safe_vals: np.ndarray,
                      ylabel: str, title: str) -> None:
    ax.plot(steps, nom_vals,  color="tab:blue", linewidth=0.9, label="nominal")
    ax.plot(steps, safe_vals, color="tab:red",  linewidth=0.9, label="safe (CBF)")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


def plot_agent(agent_idx: int, df: pd.DataFrame,
               out_path: str, episode_name: str) -> None:
    """Generate and save a 5-row time-series figure for a single agent.

    Args:
        agent_idx:    Integer index of the agent.
        df:           DataFrame loaded from agent_i_log.csv.
        out_path:     Full path for the output PNG file.
        episode_name: Episode directory name used in the figure title.
    """
    steps = df["step"].to_numpy()

    fig, axes = plt.subplots(5, 1, figsize=(8, 15), sharex=True)
    fig.suptitle(f"{episode_name} — Agent {agent_idx}", fontsize=11)

    # --- rows 0-2: CBF barrier values ---
    for row_idx, (col, ylabel, title) in enumerate(CBF_ROWS):
        ax = axes[row_idx]
        if col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            _plot_barrier_row(ax, steps, vals, ylabel, title)
        else:
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "N/A (column not in CSV)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray")

    # --- rows 3-4: control inputs ---
    for row_idx, (nom_col, safe_col, ylabel, title) in enumerate(CTRL_ROWS, start=3):
        ax = axes[row_idx]
        nom_vals  = df[nom_col].to_numpy(dtype=float)
        safe_vals = df[safe_col].to_numpy(dtype=float)
        _plot_control_row(ax, steps, nom_vals, safe_vals, ylabel, title)

    axes[-1].set_xlabel("Step", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


def plot_episode(ep_dir: str, out_dir: str,
                 agents: list[int] | None = None) -> None:
    """Plot time-series figures for all (or specified) agents in an episode.

    Saved to: <out_dir>/<episode_name>/timeseries_plots/agent_<i>_timeseries.png

    Args:
        ep_dir:  Path to the episode directory containing agent CSV logs.
        out_dir: Root output directory.
        agents:  List of agent indices to plot. None means all agents.
    """
    episode_name = os.path.basename(ep_dir)
    agent_dfs = load_episode_dfs(ep_dir)

    if not agent_dfs:
        print(f"[skip] No agent CSV files found in: {ep_dir}")
        return

    plot_dir = os.path.join(out_dir, episode_name, "timeseries_plots")

    for idx, df in sorted(agent_dfs.items()):
        if agents is not None and idx not in agents:
            continue
        out_path = os.path.join(plot_dir, f"agent_{idx}_timeseries.png")
        plot_agent(idx, df, out_path, episode_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for plot_timeseries."""
    parser = argparse.ArgumentParser(
        description="Generate per-agent time-series plots from simulation CSV logs"
    )
    parser.add_argument(
        "--run_dir",
        default="results/quantitative/agent_3",
        help="Directory containing episode subdirectories (default: results/quantitative/agent_3)",
    )
    parser.add_argument(
        "--episode",
        default=None,
        help="Plot only this episode subdirectory name (default: all episodes)",
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
        help="Root output directory (default: <run_dir>/plots)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "plots")

    if not os.path.isdir(run_dir):
        print(f"[error] run_dir does not exist: {run_dir}")
        return

    if args.episode:
        ep_dir = os.path.join(run_dir, args.episode)
        if not os.path.isdir(ep_dir):
            print(f"[error] episode directory not found: {ep_dir}")
            return
        plot_episode(ep_dir, out_dir, agents=args.agents)
    else:
        for entry in sorted(os.listdir(run_dir)):
            ep_dir = os.path.join(run_dir, entry)
            if os.path.isdir(ep_dir):
                plot_episode(ep_dir, out_dir, agents=args.agents)


if __name__ == "__main__":
    main()
