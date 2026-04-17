import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")  # GUI 없이 그림만 저장
import matplotlib.pyplot as plt

import traceback
import time
from typing import List

from task.env.cbf_env import CBFEnv
from task.models.hocbf import DifferentiableCBFLayer
from task.utils.control_utils import get_nominal_control
from task.logger.sim_logger import SimLogger
from visualization import draw_frame, make_figure
import copy


def run_single_simulation(
    cfg: dict,
    episode_index: int,
    map_tag: str,
    steps: int,
    root_out_dir: str = "validation_results",
    frame_interval: int = 50,     # Interval of PNG
    gif_interval: int | None = None,  # Interval of GIF
    gif_fps: int = 30,            # FPS of GIF
) -> None:
    
    seed = cfg['env']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg = copy.deepcopy(cfg)
    map_filepath = f"maps/{map_tag}/map_{episode_index:03d}.png"
    env_cfg['env']['map']['map_filepath'] = map_filepath

    print(f"\n=== Simulation for {map_tag} (episode_index={episode_index}) ===")

    # --- Output dir for this map ---
    map_out_dir = os.path.join(root_out_dir, f"{map_tag}_{episode_index:03d}")
    os.makedirs(map_out_dir, exist_ok=True)
    print(f"[{map_tag} {episode_index}] Results dir: {map_out_dir}")

    # --- Device ---
    device = torch.device(cfg['env']['device'])

    # --- Environment  ---
    env = CBFEnv(cfg=env_cfg['env'], episode_index=episode_index)
    print("Environment created.")

    # --- Safety Layer (CBF) ---
    cfg['model']['safety']['a_max']      = env.cfg.max_acceleration
    cfg['model']['safety']['w_max']      = env.cfg.max_yaw_rate
    cfg['model']['safety']['d_max']      = env.neighbor_radius
    cfg['model']['safety']['d_obs']      = env.cfg.d_obs
    cfg['model']['safety']['d_safe']     = env.cfg.d_safe
    cfg['model']['safety']['max_agents'] = env.cfg.max_agents
    cfg['model']['safety']['max_obs']    = env.cfg.max_obs

    cbf_layer = DifferentiableCBFLayer(cfg=cfg['model']['safety'], device=device)

    num_agents = cfg['env']['num_agent']
    print("Safety CBF layer created.")

    # --- Visualization options ---
    show_traj = cfg.get("visualization", {}).get("show_trajectory", True)

    # --- Figure for offscreen rendering ---
    fig, ax1, ax2 = make_figure(env.map_info.gt.shape)

    # --- Data tracking ---
    logger = SimLogger(num_agents, env.cfg.d_obs, env.cfg.d_safe, env.neighbor_radius)

    # --- Simulation parameters (Stop sign) ---
    prev_positions  = None
    frozen_counter  = 0
    frozen_eps      = 1e-4
    frozen_thresh   = 20 

    # --- Reset env ---
    obs, state, info = env.reset(episode_index=episode_index)
    
    # --- Termination info ---
    termination_info = {
        "stopped": False,
        "stop_step": None,
        "stop_reason": None,
    }

    try:
        for step_num in range(steps):
            # --- Nominal control ---
            demo_nominal = get_nominal_control(
                p_target=info["nominal"]["p_targets"],
                follower=info["nominal"]["follower"],
                v_current=info["safety"]["v_current"],
                a_max=env.max_lin_acc,
                w_max=env.max_ang_vel,
                v_max=env.max_lin_vel
            )

            raw_actions = demo_nominal / np.array([env.max_lin_acc, env.max_ang_vel], dtype=np.float32)
            raw_actions_t = torch.tensor(raw_actions, device=device, dtype=torch.float32)

            # --- CBF Safety Filter ---
            actions, feasible = cbf_layer(raw_actions_t, info["safety"])

            # --- Env step ---
            next_obs, next_state, reward, terminated, truncated, next_info = env.step(actions)
            done = bool(np.any(terminated) or np.any(truncated))

            # --- Check stopping sign ---
            cur_pos = env.robot_locations.copy()
            if prev_positions is not None:
                pos_diff = np.linalg.norm(cur_pos - prev_positions, axis=1)
                if np.all(pos_diff < frozen_eps):
                    frozen_counter += 1
                else:
                    frozen_counter = 0
            prev_positions = cur_pos

            if frozen_counter >= frozen_thresh:
                print(
                    f"[{map_tag} {episode_index}] Agents frozen for "
                    f"{frozen_counter} consecutive steps → early stop."
                )
                done = True

            # --- Stopping logging ---
            if done and (not termination_info["stopped"]):
                termination_info["stopped"] = True
                termination_info["stop_step"] = step_num

                reached_goal = env.is_reached_goal.squeeze(-1)      # (N,)
                coll_obs     = env.is_collided_obstacle.squeeze(-1) # (N,)
                coll_drone   = env.is_collided_drone.squeeze(-1)    # (N,)

                if np.any(reached_goal):
                    termination_info["stop_reason"] = "goal"
                elif np.any(coll_obs):
                    termination_info["stop_reason"] = "obstacle_collision"
                elif np.any(coll_drone):
                    termination_info["stop_reason"] = "robot_collision"
                elif frozen_counter >= frozen_thresh:
                    termination_info["stop_reason"] = "frozen"
                elif np.any(truncated):
                    termination_info["stop_reason"] = "timeout"
                else:
                    termination_info["stop_reason"] = "unknown"

            # --- Logging ---
            logger.record(next_info, raw_actions, actions)

            # --- CLI Log ---
            locs = next_info["viz"]["robot_locations"]
            pos_str = " | ".join(
                [f"A{i}: ({locs[i,0]:.3f}, {locs[i,1]:.3f})"
                 for i in range(next_info["viz"]["num_agent"])]
            )
            print(
                f"[{map_tag} {episode_index}] "
                f"Step {step_num + 1}/{steps} | Done: {bool(done)} | {pos_str}"
            )

            # --- Save Frame (PNG / GIF) ---
            need_png = (frame_interval is not None) and (step_num % frame_interval == 0)
            need_gif = (gif_interval is not None) and (step_num % gif_interval == 0)

            if need_png or need_gif:
                viz_data = dict(next_info["viz"])
                viz_data["paths"]        = logger.get_path_history()
                viz_data["obs_local"]    = viz_data["obstacle_states"]
                viz_data["last_cmds"]    = [(actions[i][0].item(), actions[i][1].item())
                                            for i in range(num_agents)]
                viz_data["target_local"] = next_info["nominal"]["p_targets"]
                viz_data["follower"]     = next_info["nominal"]["follower"]

                ax1.cla()
                ax2.cla()
                draw_frame(ax1, ax2, viz_data, show_trajectory=show_traj)

                # PNG
                if need_png:
                    frame_path = os.path.join(
                        map_out_dir, f"frame_{step_num:04d}.png"
                    )
                    fig.savefig(frame_path, dpi=150, bbox_inches="tight")

                # GIF Frame
                if need_gif:
                    logger.append_gif_frame(fig)

            # --- Update ---
            obs = next_obs
            state = next_state
            info = next_info

            if done:
                print(f"[{map_tag} {episode_index}] Simulation ended (done=True).")
                break

    except Exception as e:
        print(f"[{map_tag} {episode_index}] Error during simulation loop: {e}")
        traceback.print_exc()

    finally:
        # --- Save final frame ---
        try:
            if 'fig' in locals() and fig is not None and 'info' in locals() and 'actions' in locals():
                viz_data = dict(info["viz"])
                viz_data["paths"]        = logger.get_path_history()
                viz_data["obs_local"]    = viz_data["obstacle_states"]
                viz_data["last_cmds"]    = [(actions[i][0].item(), actions[i][1].item())
                                            for i in range(num_agents)]
                viz_data["target_local"] = info["nominal"]["p_targets"]

                ax1.cla()
                ax2.cla()
                draw_frame(ax1, ax2, viz_data, show_trajectory=show_traj)

                final_frame_path = os.path.join(
                    map_out_dir, f"frame_{step_num:04d}_final.png"
                )
                fig.savefig(final_frame_path, dpi=150, bbox_inches="tight")
                print(f"[{map_tag} {episode_index}] Saved final frame: {final_frame_path}")

        except Exception as e:
            print(f"[{map_tag} {episode_index}] Failed to save final frame: {e}")

        plt.close(fig)

        # --- Termination Summary ---
        summary_path = os.path.join(map_out_dir, "termination_summary.txt")

        coverage_info = None
        try:
            coverage_info = compute_free_coverage(env)
        except Exception as e:
            print(f"[{map_tag}] Failed to compute coverage: {e}")
            coverage_info = None

        violation_info = None
        try:
            violation_info = compute_cbf_violation_rates(logger)
        except Exception as e:
            print(f"[{map_tag}] Failed to compute CBF violation rates: {e}")
            violation_info = None

        try:
            with open(summary_path, "w") as f:
                f.write(f"map_tag: {map_tag}\n")
                f.write(f"episode_index: {episode_index}\n")
                f.write(f"stopped: {termination_info['stopped']}\n")
                f.write(f"stop_step: {termination_info['stop_step']}\n")
                f.write(f"stop_reason: {termination_info['stop_reason']}\n")

                if coverage_info is None:
                    f.write("free_coverage: NA (compute failed)\n")
                else:
                    f.write(f"free_coverage: {coverage_info['coverage']:.6f}\n")
                    f.write(f"covered_free_cells: {coverage_info['covered_free']}\n")
                    f.write(f"total_gt_free_cells: {coverage_info['total_gt_free']}\n")
                    f.write(f"belief_free_cells: {coverage_info['belief_free']}\n")

                if violation_info is None:
                    f.write("cbf_violation_rates: NA (compute failed)\n")
                else:
                    f.write(f"cbf_total_steps: {violation_info['total_steps']}\n")
                    f.write(f"cbf_obs_violation_rate: {violation_info['obs_violation_rate']:.6f}\n")
                    f.write(f"cbf_avoid_violation_rate: {violation_info['avoid_violation_rate']:.6f}\n")
                    f.write(f"cbf_conn_violation_rate: {violation_info['conn_violation_rate']:.6f}\n")

        except Exception as e:
            print(f"[{map_tag}] Failed to write termination summary: {e}")

        # --- CSV ---
        try:
            logger.save_csv(map_out_dir, env.dt)
        except Exception as e:
            print(f"[{map_tag} {episode_index}] Failed to save CSV logs: {e}")

        # --- GIF ---
        try:
            logger.save_gif(map_out_dir, map_tag, episode_index, gif_fps)
        except Exception as e:
            print(f"[{map_tag} {episode_index}] Failed to save GIF: {e}")

        print(f"[{map_tag} {episode_index}] Finished.\n")


def compute_free_coverage(env, verbose: bool = True):
    maps = env.map_info
    gt  = np.asarray(maps.gt)
    bel = np.asarray(maps.belief)
    mm  = maps.map_mask

    free_like = [mm["free"], mm["start"], mm["goal"]]

    gt_free  = np.isin(gt,  free_like)
    bel_free = np.isin(bel, free_like)

    total_gt_free = int(gt_free.sum())                        
    covered_free  = int(np.logical_and(gt_free, bel_free).sum()) 
    belief_free   = int(bel_free.sum())                         

    coverage = (covered_free / total_gt_free) if total_gt_free > 0 else 0.0

    if verbose:
        print(f"free_coverage numerator(covered_free) = {covered_free}")
        print(f"free_coverage denominator(total_gt_free) = {total_gt_free}")
        print(f"belief_free_cells = {belief_free}")
        print(f"free_coverage = {coverage:.6f}")

    return {
        "coverage": float(coverage),
        "covered_free": covered_free,
        "total_gt_free": total_gt_free,
        "belief_free": belief_free,
    }


def compute_cbf_violation_rates(logger: SimLogger, verbose: bool = True) -> dict:
    """Compute per-condition CBF barrier violation rates over the episode.

    A timestep t is counted as violated if any agent has barrier value < 0 at t.

    Returns:
        dict with keys:
            obs_violation_rate   (float): obstacle avoidance violation rate
            avoid_violation_rate (float): inter-agent collision avoidance violation rate
            conn_violation_rate  (float): connectivity violation rate
            total_steps          (int):   number of recorded steps used
    """
    eps = 1e-6
    N = logger._num_agents
    histories = logger.cbf_history

    T = min(len(histories[j]) for j in range(N))
    if T == 0:
        return {
            "obs_violation_rate":   0.0,
            "avoid_violation_rate": 0.0,
            "conn_violation_rate":  0.0,
            "total_steps":          0,
        }

    obs_violated   = 0
    avoid_violated = 0
    conn_violated  = 0

    for t in range(T):
        # Ignore numerical errors
        if any(histories[j][t]["obs_avoid"]   < -eps for j in range(N)):
            obs_violated   += 1
        if any(histories[j][t]["agent_avoid"] < -eps for j in range(N)):
            avoid_violated += 1
        if any(histories[j][t]["agent_conn"]  < -eps for j in range(N)):
            conn_violated  += 1

    obs_rate   = obs_violated   / T
    avoid_rate = avoid_violated / T
    conn_rate  = conn_violated  / T

    if verbose:
        print(f"cbf_total_steps = {T}")
        print(f"cbf_obs_violation_rate   = {obs_rate:.6f}")
        print(f"cbf_avoid_violation_rate = {avoid_rate:.6f}")
        print(f"cbf_conn_violation_rate  = {conn_rate:.6f}")

    return {
        "obs_violation_rate":   float(obs_rate),
        "avoid_violation_rate": float(avoid_rate),
        "conn_violation_rate":  float(conn_rate),
        "total_steps":          T,
    }


def run_validation(
    cfg: dict,
    i_shape_indices: List[int],
    square_indices: List[int],
    steps: int = 5000,
    frame_interval: int = 100,
    root_out_dir: str = "results/default",
    gif_interval: int = 5,
    gif_fps: int = 30,
):
    os.makedirs(root_out_dir, exist_ok=True)
    # i_shape
    for idx in i_shape_indices:
        run_single_simulation(
            cfg=cfg,
            episode_index=idx,
            map_tag="i_shape",
            steps=steps,
            root_out_dir=root_out_dir,
            frame_interval=frame_interval,
            gif_interval=gif_interval,
            gif_fps=gif_fps,
        )

    # square
    for idx in square_indices:
        run_single_simulation(
            cfg=cfg,
            episode_index=idx,
            map_tag="square",
            steps=steps,
            root_out_dir=root_out_dir,
            frame_interval=frame_interval,
            gif_interval=gif_interval,
            gif_fps=gif_fps,
        )


if __name__ == '__main__':
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    i_shape_indices = [11] 
    square_indices  = []

    run_validation(
        cfg=config,
        i_shape_indices=i_shape_indices,
        square_indices=square_indices,
        steps=5000,
        frame_interval=50,
        root_out_dir="results/repro_test",
        gif_interval=5,
        gif_fps=30,
    )