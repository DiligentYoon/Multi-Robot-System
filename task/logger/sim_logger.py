import os
import numpy as np
import imageio.v2 as imageio


class SimLogger:
    """Accumulates per-step trajectory/control/CBF data and handles CSV and GIF export.

    Replaces the local history buffers and save logic that previously lived in
    main_driver.run_single_simulation().
    """

    def __init__(self, num_agents: int, d_obs: float, d_safe: float, neighbor_radius: float) -> None:
        self._num_agents = num_agents
        self._d_obs = d_obs
        self._d_safe = d_safe
        self._neighbor_radius = neighbor_radius

        self.path_history: list[list[tuple]] = [[] for _ in range(num_agents)]
        self.nominal_inputs_history: list[list] = [[] for _ in range(num_agents)]
        self.safe_inputs_history: list[list] = [[] for _ in range(num_agents)]
        self.cbf_history: list[list[dict]] = [[] for _ in range(num_agents)]
        self.gif_frames: list[np.ndarray] = []

    def record(self, info: dict, raw_actions: np.ndarray, safe_actions) -> None:
        """Record one simulation step.

        Args:
            info: next_info returned by env.step(). Must contain info["viz"] and
                  info["safety"].
            raw_actions: nominal actions, shape (N, 2), numpy array.
            safe_actions: CBF-filtered actions, shape (N, 2), torch.Tensor.
        """
        robot_locations = info["viz"]["robot_locations"]
        obstacle_states = info["viz"]["obstacle_states"]
        num_obstacles = info["viz"]["num_obstacles"]

        for j in range(self._num_agents):
            # Path
            self.path_history[j].append(
                (float(robot_locations[j, 0]), float(robot_locations[j, 1]))
            )

            # Control inputs
            self.nominal_inputs_history[j].append(raw_actions[j].copy())
            self.safe_inputs_history[j].append(
                safe_actions[j].detach().cpu().numpy()
            )

            # CBF values
            obs_state_j = obstacle_states[j, :num_obstacles[j]]
            if len(obs_state_j) > 0:
                dist = np.linalg.norm(obs_state_j, axis=1)
                min_id = np.argmin(dist)
                min_obs_dist_sq = (
                    obs_state_j[min_id, 0] ** 2 + obs_state_j[min_id, 1] ** 2
                )
            else:
                min_obs_dist_sq = 0.3 ** 2  # safe dummy

            p_agents_j = info["safety"]["p_agents"][j]
            if len(p_agents_j) > 0:
                p_ag = np.asarray(p_agents_j)  # (K, 2)
                min_agent_avoid = float((p_ag[:, 0] ** 2 + p_ag[:, 1] ** 2).min()) - self._d_safe ** 2
            else:
                min_agent_avoid = self._d_safe ** 2  # no neighbors → safe dummy

            p_c = info["safety"]["p_c_agent"][j].reshape(-1)
            min_agent_dist_sq = (
                p_c[0] ** 2 + p_c[1] ** 2 if len(p_c) > 0 else 0.0
            )

            self.cbf_history[j].append({
                "obs_avoid":   min_obs_dist_sq - self._d_obs ** 2,
                "agent_avoid": min_agent_avoid,
                "agent_conn":  self._neighbor_radius ** 2 - min_agent_dist_sq,
            })

    def append_gif_frame(self, fig) -> None:
        """Capture the current figure as an RGB array and store it."""
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_rgba = buf.reshape(h, w, 4)
        self.gif_frames.append(img_rgba[..., :3].copy())

    def get_path_history(self) -> list[list[tuple]]:
        return self.path_history

    def save_csv(self, out_dir: str, dt: float) -> None:
        """Save per-agent trajectory and control logs as CSV files."""
        for i in range(self._num_agents):
            traj = np.asarray(self.path_history[i], dtype=float)           # (T, 2)
            nom  = np.asarray(self.nominal_inputs_history[i], dtype=float) # (T, 2)
            safe = np.asarray(self.safe_inputs_history[i], dtype=float)    # (T, 2)
            cbf_arr = np.asarray(
                [[d["obs_avoid"], d["agent_avoid"] ,d["agent_conn"]] for d in self.cbf_history[i]],
                dtype=float,
            )  # (T, 2)

            T = min(traj.shape[0], nom.shape[0], safe.shape[0], cbf_arr.shape[0])
            traj    = traj[:T]
            nom     = nom[:T]
            safe    = safe[:T]
            cbf_arr = cbf_arr[:T]

            steps_arr = np.arange(T)
            time_arr  = steps_arr * dt

            # step, time, x, y, a_nom, w_nom, a_safe, w_safe, obs_avoid, agent_conn
            data = np.column_stack([steps_arr, time_arr, traj, nom, safe, cbf_arr])
            header = "step, time, x, y, a_nom, w_nom, a_safe, w_safe, obs_avoid, agent_avoid, agent_conn"

            csv_path = os.path.join(out_dir, f"agent_{i}_log.csv")
            np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
            print(f"[SimLogger] Saved CSV: {csv_path}")

    def save_gif(
        self,
        out_dir: str,
        map_tag: str,
        episode_index: int,
        fps: int,
    ) -> None:
        """Save accumulated GIF frames. Does nothing if no frames were captured."""
        if not self.gif_frames:
            return
        gif_path = os.path.join(out_dir, f"{map_tag}_{episode_index:03d}.gif")
        imageio.mimsave(gif_path, self.gif_frames, fps=fps)
        print(f"[SimLogger] Saved GIF: {gif_path}")
