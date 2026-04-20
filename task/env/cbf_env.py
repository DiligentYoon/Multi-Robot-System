import numpy as np
import math
import copy
import torch
from typing import Tuple, List

from .cbf_env_cfg import CBFEnvCfg
from task.base.env.env import Env
from task.graph.graph import ConnectivityGraph
from task.utils.graph_utils import *
from task.utils.sensor_utils import *
from task.utils.transform_utils import *
from task.planner.unknown_target_planner import TargetUnknownPlanner
from task.planner.frontier_planner import FrontierPlanner
from task.planner.agent_router import AgentRouter


class CBFEnv(Env):

    def __init__(self, episode_index: int | np.ndarray, cfg: dict):
        self.cfg = CBFEnvCfg(cfg)
        super().__init__(self.cfg)
        # Simulation Parameters
        self.dt = self.cfg.physics_dt
        self.decimation = self.cfg.centralized_decimation
        self.neighbor_radius = self.cfg.neighbor_sensing_distance
        self.max_episode_steps = self.cfg.max_episode_steps
        self.assign_mode = self.cfg.assign_mode

        # Reward related
        self.last_explored_area = 0.0
        self.robot_radius = self.cfg.d_obs

        # 핵심 Planning State
        self.connectivity_graph = ConnectivityGraph(self.num_agent)
        self.connectivity_traj = [[] for _ in range(self.num_agent)]
        self.root_mask = np.zeros(self.num_agent, dtype=np.int_)
        self.end_pos_world = np.zeros((self.num_agent, 2),  dtype=np.float32)
        self.robot_speeds = np.zeros(self.num_agent, dtype=np.float32)
        self.num_obstacles = np.zeros(self.num_agent, dtype=np.int_)
        self.num_neighbors = (self.num_agent-1) * np.ones(self.num_agent, dtype=np.int_)
        self.num_frontiers = np.zeros(self.num_agent, dtype=np.int_)
        self.local_frontiers = np.zeros((self.num_agent, self.cfg.num_rays, 2), dtype=np.float32)

        self.assigned_rc = np.zeros((self.num_agent, 2), dtype=int)

        # [agent_dim, max_dim, specific_dim]
        self.obstacle_states = np.zeros((self.num_agent, self.cfg.max_obs, 2), dtype=np.float32)
        self.neighbor_states = np.zeros((self.num_agent, self.cfg.max_agents-1, 4), dtype=np.float32)
        self.neighbor_ids = np.zeros((self.num_agent, self.cfg.max_agents-1), dtype=np.int_)

        # Done flags
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_reached_goal = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)

        # Additional Info
        self.infos["safety"] = {}
        self.infos["nominal"] = {}

        self.no_path_until_refresh = np.zeros(self.num_agent, dtype=bool)

        # === viz/debug safe defaults ===
        self.cluster_infos_viz = {}
        self.targets_rc_viz = []
        self.assigned_rc_viz = None      # np.ndarray[(N,2)] (row,col) 예정
        self.reconstructed_path_viz = []
        self.targets_prob_heat = None

        # Planner / Router
        if self.assign_mode == "target_unknwown":
            self.planner = TargetUnknownPlanner()
        elif self.assign_mode == "target_frontier":
            self.planner = FrontierPlanner()
        else:
            raise ValueError(f"Unknown assign_mode: {self.assign_mode}")
        self.router = AgentRouter()

    def reset(self, episode_index: int = None):
        # 나머지 플래그는 사용하기 전 계산 되므로 초기화 X
        self.actions = np.zeros((self.num_agent, self.cfg.num_act), dtype=np.float32)
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_reached_goal = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)
        self.robot_speeds = np.zeros(self.num_agent, dtype=np.float32)
        self.last_explored_area = 0
        self.tree_interval = 0
        super().reset(episode_index)

        #   - GT의 START 마스크가 있으면 그 주변 공개
        #   - 없으면 현재 에이전트 위치 주변을 공개
        self._reveal_start_zone(radius_m=0.2, use_gt_start=True)

        # Re-compute infos and observations after revealing start zone
        self.infos = self._update_infos()
        self.obs_buf = self._get_observations()
        self.state_buf = self._get_states()

        return self.obs_buf, self.state_buf, self.infos

    
    def _reveal_start_zone(self, radius_m: float = 0.8, use_gt_start: bool = True):
        """GT 기반으로 start 주변을 공개(known)하고 frontier를 재계산."""
        maps = self.map_info
        bel = maps.belief
        gt  = maps.gt
        mm  = maps.map_mask
        H, W = bel.shape
        res = maps.res_m

        r_px = max(1, int(round(radius_m / res)))

        # 1) 중심들: GT의 START가 있으면 그 좌표들, 없으면 에이전트 현재 위치를 셀로 변환
        centers_rc = []
        if use_gt_start and np.any(gt == mm["start"]):
            centers_rc = np.argwhere(gt == mm["start"])  # (row, col)
        else:
            # 에이전트 월드좌표 → 그리드셀
            cells = maps.world_to_grid_np(self.robot_locations)  # (N, 2) as (col, row)
            centers_rc = np.stack([cells[:, 1], cells[:, 0]], axis=1)  # (row, col)

        # 2) 각 중심에 대해 원형 반경 내의 셀을 공개
        for r0, c0 in centers_rc:
            rmin = max(0, r0 - r_px); rmax = min(H, r0 + r_px + 1)
            cmin = max(0, c0 - r_px); cmax = min(W, c0 + r_px + 1)

            rr = np.arange(rmin, rmax)[:, None]
            cc = np.arange(cmin, cmax)[None, :]
            disk_mask = (rr - r0)**2 + (cc - c0)**2 <= (r_px * r_px)

            sub_gt = gt[rmin:rmax, cmin:cmax]
            sub_bel = bel[rmin:rmax, cmin:cmax]

            # GT가 occupied면 occupied, 그 외(START 포함)는 free로 공개
            occ_mask = (sub_gt == mm["occupied"])
            free_mask = ~occ_mask  # START/GOAL/FREE 등은 free로 본다

            sub_bel[disk_mask & occ_mask] = mm["occupied"]
            sub_bel[disk_mask & free_mask] = mm["free"]

            bel[rmin:rmax, cmin:cmax] = sub_bel

        # 3) 공개 결과를 바탕으로 frontier 재계산
        belief_frontier = np.full_like(bel, mm["unknown"])
        belief_frontier[bel == mm["occupied"]] = mm["occupied"]
        belief_frontier[bel == mm["free"]]     = mm["free"]

        free_mask_all = (bel == mm["free"])
        unk = (bel == mm["unknown"])

        frontier_mask = np.zeros_like(bel, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.zeros_like(unk, dtype=bool)
                r_from = max(0, -dr); r_to = H - max(0, dr)
                c_from = max(0, -dc); c_to = W - max(0, dc)
                shifted[r_from:r_to, c_from:c_to] = unk[r_from+dr:r_to+dr, c_from+dc:c_to+dc]
                frontier_mask |= (free_mask_all & shifted)

        belief_frontier[frontier_mask] = mm["frontier"]

        maps.belief = bel
        maps.belief_frontier = belief_frontier


    def _set_init_state(self,
                        max_attempts: int = 1000
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            (world_x, world_y) for each agent (fixed positions).
        """

        # 1) cfg에 init_positions가 있으면 그걸 사용
        if hasattr(self.cfg, "init_positions") and (self.cfg.init_positions is not None):
            initial_positions = np.array(self.cfg.init_positions, dtype=np.float32)
        else:
            # 2) default value
            if self.num_agent == 3:
                initial_positions = np.array([
                    [0.2, 0.3], [0.2, 0.5], [0.2, 0.7],
                ], dtype=np.float32)
                
            elif self.num_agent == 5:
                initial_positions = np.array([
                    [0.2, 0.3], [0.2, 0.5], [0.2, 0.7],
                    [0.4, 0.4], [0.4, 0.6],
                ], dtype=np.float32)

            elif self.num_agent == 7:
                initial_positions = np.array([
                    [0.2, 0.3], [0.2, 0.5], [0.2, 0.7],
                    [0.4, 0.3], [0.4, 0.5], [0.4, 0.7],
                    [0.6, 0.5],
                ], dtype=np.float32)
            else:
                raise ValueError("Unvalid agent numbers")
            
        # Optional sanity warning if out of effective map area (before padding)
        for pos in initial_positions:
            if not (0.0 <= pos[0] <= (self.map_info.meters_w - 2*self.map_info.belief_origin_x) and
                    0.0 <= pos[1] <= (self.map_info.meters_h - 2*self.map_info.belief_origin_y)):
                print(f"Warning: Position {pos.tolist()} may be outside of the effective map area.")

        world_x = initial_positions[:, 0]
        world_y = initial_positions[:, 1]
        return world_x, world_y

    def _pre_apply_action(self, actions: np.ndarray | torch.Tensor) -> None:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        self.actions = actions.copy()
        # Acceleration & Angular Velocity 생성 [min_range, max_range]
        self.preprocessed_actions = actions.copy()
    
    def _apply_action(self, agent_id):
        # Acceleration을 바탕으로 속도 업데이트 (per-agent update)
        i = agent_id
        self.robot_speeds[i] += self.preprocessed_actions[i, 0] * self.dt
        self.robot_speeds[i] = np.clip(self.robot_speeds[i], 0.0, self.max_lin_vel)
        # Non-Holodemic Model 특성에 의해 Position 먼저 업데이트
        self.robot_locations[i, 0] += self.robot_speeds[i] * np.cos(self.robot_angles[i]) * self.dt
        self.robot_locations[i, 1] += self.robot_speeds[i] * np.sin(self.robot_angles[i]) * self.dt
        # Yaw rate를 바탕으로 각도 업데이트
        self.robot_yaw_rate[i] = np.clip(self.preprocessed_actions[i, 1], -self.max_ang_vel, self.max_ang_vel)
        self.robot_angles[i] = ((self.robot_angles[i] + self.robot_yaw_rate[i] * self.dt + np.pi) % (2 * np.pi)) - np.pi
        # World Frame 속도 세팅
        self.robot_velocities[i, 0] = self.robot_speeds[i] * np.cos(self.robot_angles[i])
        self.robot_velocities[i, 1] = self.robot_speeds[i] * np.sin(self.robot_angles[i])
    
    def _compute_intermediate_values(self):
        """
            업데이트된 state값들을 바탕으로, obs값에 들어가는 planning state 계산
        """
        drone_pos = np.hstack((self.robot_locations, self.robot_angles.reshape(-1, 1)))

        # --- Zero-padding Initialization ---
        self.neighbor_states.fill(0)
        self.obstacle_states.fill(0)  
        self.local_frontiers.fill(0)
        self.neighbor_ids.fill(0)
        frontier_cells = [[] for _ in range(self.num_agent)]
        # -----------------------------------

        for i in range(self.num_agent):
            # Pos
            drone_pos_i = drone_pos[i]
            # lx, ly
            rel_pos = world_to_local(w1=drone_pos_i[:2], w2=drone_pos[:, :2], yaw=drone_pos_i[2])
            # v_jx, v_jy
            rel_vel = world_to_local(w1=None, w2=self.robot_velocities, yaw=drone_pos_i[2])
            # sqrt(lx^2 + lx^2)
            distance = np.linalg.norm(rel_pos, axis=1)
            # 자기자신 제외한 모든 Agent Global Ids
            other_agent_ids = np.where(distance > 1e-5)[0]
            # Local Graph 반경에 포함된 이웃 Agent Global Ids
            neighbor_agent_ids = np.where(np.logical_and(distance <= self.neighbor_radius, distance > 1e-5))[0]

            # Neighbor State for Agent Collision Avoidance
            self.num_neighbors[i] = len(neighbor_agent_ids)
            if self.num_neighbors[i] > 0:
                self.neighbor_states[i, :self.num_neighbors[i], :2] = rel_pos[neighbor_agent_ids]
                self.neighbor_states[i, :self.num_neighbors[i], 2:] = rel_vel[neighbor_agent_ids]
                self.neighbor_ids[i, :self.num_neighbors[i]] = neighbor_agent_ids # Global Ids
            else:
                # 0개인 경우 (Connectivity Slack으로 인한 예외상황) : 가장 가까운 Agent가 neighbors
                closest_neighbor_id_local = np.argmin(distance[other_agent_ids])
                closest_neighbor_id_global = other_agent_ids[closest_neighbor_id_local]
                self.num_neighbors[i] = 1
                self.neighbor_states[i, 0, :2] = rel_pos[closest_neighbor_id_global]
                self.neighbor_states[i, 0, 2:] = rel_vel[closest_neighbor_id_global]
                self.neighbor_ids[i, 0] = closest_neighbor_id_global # Global Ids

            # Obstacles & Frontiers Sensing
            local_frontiers, frontiers_cell, local_obstacles = sense_and_update(map_info=self.map_info,
                                                                                fov=self.fov,
                                                                                num_rays=self.cfg.num_rays,
                                                                                sensor_range=self.sensor_range,
                                                                                agent_id=i,
                                                                                robot_locations=self.robot_locations,
                                                                                robot_angles=self.robot_angles)
            bel = self.map_info.belief
            bel[bel == self.map_info.map_mask["frontier"]] = self.map_info.map_mask["free"]
            for r, c in frontiers_cell:
                if bel[r, c] == self.map_info.map_mask["free"]: bel[r, c] = self.map_info.map_mask["frontier"]

            # Store obstacle information
            num_obs = min(len(local_obstacles), self.cfg.max_obs)
            if num_obs > 0:
                self.obstacle_states[i, :num_obs] = local_obstacles[:num_obs]
            else:
                self.obstacle_states[i, :] = 0
            self.num_obstacles[i] = num_obs
            # Store frontier information
            num_frontiers = min(len(local_frontiers), self.cfg.num_rays)
            if num_frontiers > 0:
                self.local_frontiers[i, :num_frontiers] = local_frontiers
            else:
                self.local_frontiers[i, :] = 0
            self.num_frontiers[i] = num_frontiers
            frontier_cells[i].append(frontiers_cell)
           
        # 마지막 루프에 대한 Frontier Marking 정리
        bel[bel == self.map_info.map_mask["frontier"]] = self.map_info.map_mask["free"]

        # Update된 Belief Map에 대한 Frontier 마킹
        reset_flag = np.all(self.map_info.belief_frontier == self.map_info.map_mask["unknown"])
        self.map_info.belief_frontier = global_frontier_marking(self.map_info, reset_flag, frontier_cells)

    
    def _get_observations(self) -> np.ndarray | list[dict]:
        """TODO: when RL implementation, resume this part."""
        return None

    def _get_states(self) -> np.ndarray | dict:
        """TODO: when RL implementation, resume this part."""
        return None


    def _get_dones(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            특정 종료조건 및 타임아웃 계산
            Return :
                1. terminated : 
                    1-1. 벽에 충돌
                    1-2. 드론끼리 충돌
                    1-3. 골 지점 도달
                2. truncated :
                    2-1. 타임아웃

        """
        # Planning State 업데이트
        self._compute_intermediate_values()

        # ============== Done 계산 로직 ===================

        # ---- Truncated 계산 -----
        timeout = self.num_step >= self.max_episode_steps - 1
        truncated = np.full((self.num_agent, 1), timeout, dtype=np.bool_)

        # ---- Terminated 계산 ----
        # 로봇 셀 좌표 변환
        cells = self.map_info.world_to_grid_np(self.robot_locations)
        rows, cols = cells[:, 1], cells[:, 0]

        # 목표 도달 유무 체크
        self.is_reached_goal = (self.map_info.gt[rows, cols] == self.map_info.map_mask["goal"]).reshape(-1, 1)

        # 맵 경계 체크
        H, W = self.map_info.H, self.map_info.W
        out_of_bounds = (rows < 0) | (rows >= H) | (cols < 0) | (cols >= W)

        # 유효한 셀에 대해서만 값 확인
        valid_indices = ~out_of_bounds
        valid_rows, valid_cols = rows[valid_indices], cols[valid_indices]

        # 장애물 충돌 (맵 밖 포함)
        hit_obstacle = np.zeros_like(out_of_bounds, dtype=np.bool_)
        hit_obstacle[valid_indices] = self.map_info.gt[valid_rows, valid_cols] == self.map_info.map_mask["occupied"]
        self.is_collided_obstacle = (hit_obstacle | out_of_bounds)[:, np.newaxis]

        # 드론 간 충돌 (점유 셀이 겹치면 충돌 판단)
        flat_indices = rows * W + cols
        unique_indices, counts = np.unique(flat_indices, return_counts=True)
        collided_indices = unique_indices[counts > 1]
        
        self.is_collided_drone.fill(False)
        for idx in collided_indices:
            colliding_agents = np.where(flat_indices == idx)[0]
            for agent_idx in colliding_agents:
                self.is_collided_drone[agent_idx] = True
        
        terminated = self.is_collided_obstacle | self.is_collided_drone | self.is_reached_goal

        # ========= 로그 추가 =========
        if np.any(terminated):
            term_mask = terminated.squeeze(-1)  # (N,)

            collided_obs_agents   = np.where(self.is_collided_obstacle.squeeze(-1))[0]
            collided_drone_agents = np.where(self.is_collided_drone.squeeze(-1))[0]
            reached_goal_agents   = np.where(self.is_reached_goal.squeeze(-1))[0]

            print("\n===== [DONE] Episode terminated at step {} =====".format(self.num_step))
            for i in range(self.num_agent):
                if not term_mask[i]:
                    continue
                status = []
                if i in collided_obs_agents:
                    status.append("hit_obstacle")
                if i in collided_drone_agents:
                    status.append("drone_collision")
                if i in reached_goal_agents:
                    status.append("reached_goal")
                pos = self.robot_locations[i]
                print(f"  Agent {i}: {', '.join(status)} at pos=({pos[0]:.3f}, {pos[1]:.3f})")
            print("========================================\n")


        return terminated, truncated, self.is_reached_goal
    

    def _get_rewards(self):
        """TODO: when RL implementation, resume this part."""
        return None
    

    def _update_infos(self):
        infos = {}

        # [A] Centralized Planning
        if self.tree_interval % self.cfg.centralized_decimation == 0:
            self.no_path_until_refresh[:] = False

            plan_result = self.planner.plan(
                map_info         = self.map_info,
                robot_locations  = self.robot_locations,
                robot_velocities = self.robot_velocities,
                num_agent        = self.num_agent,
                cfg              = self.cfg,
            )

            self.assigned_rc = plan_result["assigned_rc"]
            root_id = plan_result["root_id"]
            self.root_mask.fill(0)
            self.root_mask[root_id] = 1
            if self.cfg.graph_mode == "nn_tree":
                self.connectivity_graph.update_nearest_neighbor_tree(
                    self.robot_locations, root_id, self.neighbor_radius
                )
            else:
                self.connectivity_graph.update_and_compute_mst(self.robot_locations, root_id)

            # Visualization info
            self.targets_prob_heat = plan_result["viz"]["targets_prob_heat"]
            self.assigned_rc_viz   = plan_result["viz"]["assigned_rc_viz"]
            self.cluster_infos     = plan_result["viz"]["cluster_infos"]
            self.regions           = None
            self.valid_regions     = None

        # [B] Per-Agent Routing (A*)
        routing = self.router.route_all(
            map_info              = self.map_info,
            connectivity_graph    = self.connectivity_graph,
            robot_locations       = self.robot_locations,
            robot_angles          = self.robot_angles,
            robot_velocities      = self.robot_velocities,
            robot_speeds          = self.robot_speeds,
            obstacle_states       = self.obstacle_states,
            num_obstacles         = self.num_obstacles,
            neighbor_states       = self.neighbor_states,
            num_neighbors         = self.num_neighbors,
            assigned_rc           = self.assigned_rc,
            cfg                   = self.cfg,
            no_path_until_refresh = self.no_path_until_refresh,
        )

        # [C] CBFEnv properties
        self.end_pos_world         = routing["end_pos_world"]
        self.connectivity_traj     = routing["connectivity_traj"]
        self.no_path_until_refresh = routing["no_path_until_refresh"]

        # [D] infos
        infos["safety"] = {
            "v_current"      : list(self.robot_speeds),
            "p_obs"          : routing["p_obs_list"],
            "p_agents"       : routing["p_agents_list"],
            "v_agents_local" : routing["v_agents_local_list"],
            "p_c_agent"      : routing["p_c_agent_list"],
            "v_c_agent"      : routing["v_c_agent_list"],
        }
        infos["nominal"] = {
            "p_targets" : routing["target_pos_list"],
            "follower"  : routing["follower_list"],
        }

        infos["viz"] = self._get_viz_info()
        self.tree_interval += 1
        return infos

    def _get_viz_info(self) -> dict:
        """Collect all env-side data needed for visualization into a single dict.

        Computes connectivity_pairs here so main_driver does not need to access
        env internals directly.
        """
        connectivity_pairs = []
        for i in range(self.num_agent):
            pos1 = self.robot_locations[i]
            if not self.root_mask[i]:
                parent_id = self.connectivity_graph.get_parent(i)
                pos2 = self.robot_locations[parent_id] if parent_id != -1 else pos1
            else:
                pos2 = pos1
            connectivity_pairs.append((pos1.copy(), pos2.copy()))

        return {
            "map_info":          self.map_info,
            "num_agent":         self.num_agent,
            "robot_locations":   self.robot_locations.copy(),
            "robot_angles":      self.robot_angles.copy(),
            "cfg_d_max":         self.cfg.d_max,
            "cfg_d_safe":        self.cfg.d_safe,
            "cfg_fov":           self.cfg.fov,
            "cfg_sensor_range":  self.cfg.sensor_range,
            "obstacle_states":   self.obstacle_states.copy(),
            "num_obstacles":     self.num_obstacles.copy(),
            "targets_prob_heat": self.targets_prob_heat,
            "connectivity_pairs": connectivity_pairs,
            "connectivity_trajs": self.connectivity_traj,
            "assigned_dests":    self.assigned_rc_viz,
        }