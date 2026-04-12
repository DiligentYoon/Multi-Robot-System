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
from task.models.models import DifferentiableCBFLayer
from task.utils import get_nominal_control
from visualization import draw_frame
import copy
import imageio.v2 as imageio


def run_single_simulation(
    cfg: dict,
    episode_index: int,
    map_tag: str,              
    steps: int,
    layout: str,
    root_out_dir: str = "validation_results",
    frame_interval: int = 50,     # Interval of PNG 
    gif_interval: int | None = None,  # Interval of GIF
    gif_fps: int = 30,            # FPS of GIF
) -> None:
    
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

    # --- Figure for offscreen rendering ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Data tracking ---
    nominal_inputs_history = [[] for _ in range(num_agents)]  # [ [ [a_nom,w_nom], ... ], ...]
    safe_inputs_history    = [[] for _ in range(num_agents)]  # [ [ [a_safe,w_safe], ... ], ...]
    path_history           = [[] for _ in range(num_agents)]  # [ [ (x,y), ... ], ...]
    cbf_history            = [[] for _ in range(num_agents)]

    # GIF
    gif_frames = [] if gif_interval is not None else None

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
            for j in range(num_agents):
                # position
                path_history[j].append((env.robot_locations[j, 0],
                                        env.robot_locations[j, 1]))

                # control input
                nominal_inputs_history[j].append(raw_actions[j].copy())
                safe_inputs_history[j].append(actions[j].detach().cpu().numpy())

                # Barrier value
                if env.num_obstacles[j] > 0:
                    obs_state_j = env.obstacle_states[j, :env.num_obstacles[j]]
                    dist = np.linalg.norm(obs_state_j, axis=1)
                    min_ids = np.argmin(dist)
                    min_obs_dist_sq = (
                        obs_state_j[min_ids, 0]**2 + obs_state_j[min_ids, 1]**2
                    )
                else:
                    min_obs_dist_sq = 0.3**2  # dummy

                p_c = next_info["safety"]["p_c_agent"][j].reshape(-1)
                if len(p_c) > 0:
                    min_agent_dist_sq = p_c[0]**2 + p_c[1]**2
                else:
                    min_agent_dist_sq = 0.0

                agent_cbf_info = {
                    "obs_avoid": min_obs_dist_sq - env.cfg.d_obs**2,
                    "agent_conn": env.neighbor_radius**2 - min_agent_dist_sq,
                }
                cbf_history[j].append(agent_cbf_info)

            # --- 콘솔 로그 ---
            pos_str = " | ".join(
                [f"A{i}: ({env.robot_locations[i,0]:.3f}, {env.robot_locations[i,1]:.3f})"
                 for i in range(env.num_agent)]
            )
            print(
                f"[{map_tag} {episode_index}] "
                f"Step {step_num + 1}/{steps} | Done: {bool(done)} | {pos_str}"
            )

            # --- Save Frame (PNG / GIF) ---
            need_png = (frame_interval is not None) and (step_num % frame_interval == 0)
            need_gif = (gif_interval is not None) and (step_num % gif_interval == 0)

            if need_png or need_gif:
                # connectivity_pairs
                connectivity_pairs = []
                for i in range(env.num_agent):
                    pos1 = env.robot_locations[i]
                    if not env.root_mask[i]:
                        parent_id = env.connectivity_graph.get_parent(i)
                        if parent_id != -1:
                            pos2 = env.robot_locations[parent_id]
                        else:
                            pos2 = pos1
                    else:
                        pos2 = pos1
                    connectivity_pairs.append((pos1, pos2))

                viz_data = {
                    "paths": path_history,
                    "obs_local": env.obstacle_states,
                    "last_cmds": [(actions[i][0].item(), actions[i][1].item()) for i in range(num_agents)],
                    "connectivity_pairs": connectivity_pairs,
                    "target_local": info["nominal"]["p_targets"],
                    "connectivity_trajs": env.connectivity_traj,
                    "assigned_dests": getattr(env, "assigned_rc_viz", None),
                    "follower": info["nominal"]["follower"],
                }

                ax1.cla()
                ax2.cla()
                draw_frame(ax1, ax2, env, viz_data, layout=layout)

                # PNG
                if need_png:
                    frame_path = os.path.join(
                        map_out_dir, f"frame_{step_num:04d}.png"
                    )
                    fig.savefig(frame_path, dpi=150, bbox_inches="tight")

                # GIF Frame
                if need_gif and gif_frames is not None:
                    fig.canvas.draw()
                    w, h = fig.canvas.get_width_height()

                    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    img_rgba = buf.reshape(h, w, 4)

                    img_rgb = img_rgba[..., :3]
                    gif_frames.append(img_rgb.copy())

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
        # --- 마지막 프레임은 항상 한 번 저장 ---
        try:
            if 'fig' in locals() and fig is not None and 'env' in locals():
                if 'info' in locals() and 'actions' in locals():
                    connectivity_pairs = []
                    for i in range(env.num_agent):
                        pos1 = env.robot_locations[i]
                        if not env.root_mask[i]:
                            parent_id = env.connectivity_graph.get_parent(i)
                            if parent_id != -1:
                                pos2 = env.robot_locations[parent_id]
                            else:
                                pos2 = pos1
                        else:
                            pos2 = pos1
                        connectivity_pairs.append((pos1, pos2))

                    viz_data = {
                        "paths": path_history,
                        "obs_local": env.obstacle_states,
                        "last_cmds": [(actions[i][0].item(), actions[i][1].item())
                                      for i in range(num_agents)],
                        "connectivity_pairs": connectivity_pairs,
                        "target_local": info["nominal"]["p_targets"],
                        "connectivity_trajs": env.connectivity_traj,
                        "assigned_dests": getattr(env, "assigned_rc_viz", None),
                    }

                    ax1.cla()
                    ax2.cla()
                    draw_frame(ax1, ax2, env, viz_data, layout=layout)

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

        except Exception as e:
            print(f"[{map_tag}] Failed to write termination summary: {e}")

        # --- CSV ---
        try:
            dt = env.dt
            for i in range(num_agents):
                traj = np.asarray(path_history[i], dtype=float)           # (T,2)
                nom  = np.asarray(nominal_inputs_history[i], dtype=float) # (T,2)
                safe = np.asarray(safe_inputs_history[i], dtype=float)    # (T,2)

                cbf_arr = np.asarray(
                    [[d["obs_avoid"], d["agent_conn"]] for d in cbf_history[i]],
                    dtype=float
                ) 

                T = min(traj.shape[0], nom.shape[0], safe.shape[0], cbf_arr.shape[0])
                traj    = traj[:T]
                nom     = nom[:T]
                safe    = safe[:T]
                cbf_arr = cbf_arr[:T]

                steps_arr = np.arange(T)
                time_arr  = steps_arr * dt

                # step, time, x, y, a_nom, w_nom, a_safe, w_safe, obs_avoid, agent_conn
                data = np.column_stack([steps_arr, time_arr, traj, nom, safe, cbf_arr])

                header = "step,time,x,y,a_nom,w_nom,a_safe,w_safe,obs_avoid,agent_conn"

                csv_path = os.path.join(map_out_dir, f"agent_{i}_log.csv")
                np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
                print(f"[{map_tag} {episode_index}] Saved CSV: {csv_path}")
        except Exception as e:
            print(f"[{map_tag} {episode_index}] Failed to save CSV logs: {e}")

        # --- GIF ---
        try:
            if gif_frames is not None and len(gif_frames) > 0:
                gif_path = os.path.join(map_out_dir, f"{map_tag}_{episode_index:03d}.gif")
                imageio.mimsave(gif_path, gif_frames, fps=gif_fps)
                print(f"[{map_tag} {episode_index}] Saved GIF: {gif_path}")
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



def run_validation(
    cfg: dict,
    i_shape_indices: List[int],
    square_indices: List[int],
    steps: int = 2000,
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
            layout="square",
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
            layout="square",
            root_out_dir=root_out_dir,
            frame_interval=frame_interval,
            gif_interval=gif_interval,
            gif_fps=gif_fps,
        )


if __name__ == '__main__':
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['env']['seed'])
    np.random.seed(config['env']['seed'])

    i_shape_indices = [0, 9] 
    # square_indices  = [2, 13, 15, 18, 28]

    # i_shape_indices = [30, 31, 32, 35, 43]
    # square_indices  = [27, 29, 40, 42, 46]

    #i_shape_indices = [165, 166, 167, 168, 169, 170, 171, 172, 173, 174]
    #i_shape_indices = [165]
    square_indices  = []

    run_validation(
        cfg=config,
        i_shape_indices=i_shape_indices,
        square_indices=square_indices,
        steps=10000,
        frame_interval=50,
        root_out_dir="results/default",
        gif_interval=100,    
        gif_fps=30,     
    )
