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
        self.reward_weights = self.cfg.reward_weights
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
            # 2) 없으면 기존 하드코딩 초기값 사용 (기본값)
            initial_positions = np.array([
                [0.2, 0.3], [0.2, 0.5], [0.2, 0.7],
                [0.4, 0.3], [0.4, 0.5], [0.4, 0.7],
                [0.6, 0.3], [0.6, 0.5], [0.6, 0.7],
            ], dtype=np.float32)
            
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
        self.preprocessed_actions = actions.copy()
        # Acceleration & Angular Velocity 생성 [-1, 1] -> [min_range, max_range]
        self.preprocessed_actions[:, 0] *= self.max_lin_acc
        self.preprocessed_actions[:, 1] *= self.max_ang_vel
    
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
            local_frontiers, frontiers_cell, local_obstacles = self.sense_and_update(agent_id=i)

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

        # ========= 여기부터 로그 추가 =========
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
        follower_list  = []
        target_pos_list =[]
        end_pos_world_list = []
        p_obs_list = []
        p_agents_list = []
        p_c_agent_list = []
        v_c_agent_list = []
        v_agents_local_list = []
        connectivity_traj = [[] for _ in range(self.num_agent)]
        routing_log_lines: list[str] = []

        # Centralized Information Update
        if self.tree_interval % self.cfg.centralized_decimation == 0:
            
            if self.assign_mode == "target_unknwown":

                self.no_path_until_refresh[:] = False

                import cv2
                from scipy.ndimage import binary_fill_holes

                bel = self.map_info.belief
                bf  = self.map_info.belief_frontier
                mm  = self.map_info.map_mask
                H, W = self.map_info.H, self.map_info.W

                FREE_LABEL     = mm["free"]
                UNKNOWN_LABEL  = mm["unknown"]
                OCCUPIED_LABEL = mm["occupied"]
                FRONTIER_LABEL = mm["frontier"]

                free_mask     = (bel == FREE_LABEL)
                unknown_mask  = (bel == UNKNOWN_LABEL)
                occ_mask      = (bel == OCCUPIED_LABEL)
                frontier_mask = (bf  == FRONTIER_LABEL)
                
                # union_mask: frontier ∪ occupied  (H,W), 0/1 또는 False/True
                union_mask = (occ_mask | frontier_mask).astype(np.uint8)
                
                # 1) 가장 큰 contour 찾기 (outer boundary)
                contours, _ = cv2.findContours(
                        union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if len(contours) == 0:
                    raise ValueError("[target_unknwown] union_mask has no contour. Check occ/frontier.")

                largest = max(contours, key=cv2.contourArea)  # 가장 넓은 contour 하나 선택

                # 2) boundary mask (1-pixel contour)
                # boundary_mask = np.zeros_like(union_mask, dtype=np.uint8)
                # cv2.drawContours(boundary_mask, [largest], contourIdx=-1, color=1, thickness=1)
                #boundary_bool = boundary_mask.astype(bool)

                contours, _ = cv2.findContours(
                    union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                boundary_mask = np.zeros_like(union_mask, dtype=np.uint8)

                # 면적이 너무 작은 noise contour는 걸러도 됨 (예: area > 10 같은 조건)
                for cnt in contours:
                    if cv2.contourArea(cnt) < 5:   # 임계치는 맵 해상도에 맞게 튜닝
                        continue
                    cv2.drawContours(boundary_mask, [cnt], contourIdx=-1, color=1, thickness=1)

                boundary_bool = boundary_mask.astype(bool)

                inside_bool = binary_fill_holes(boundary_bool)

                # 3) radius만큼 contour를 확장해서 band 생성 (안쪽+바깥쪽 모두 포함)
                band_thickness = 0   # [cells] 튜닝 파라미터
                H, W = union_mask.shape

                interest_mask = np.zeros_like(boundary_bool, dtype=bool)
                idx_rc = np.argwhere(boundary_bool)  # boundary에 해당하는 (row, col)들

                for r, c in idx_rc:
                    r0 = max(0, r - band_thickness)
                    r1 = min(H, r + band_thickness + 1)
                    c0 = max(0, c - band_thickness)
                    c1 = min(W, c + band_thickness + 1)
                    interest_mask[r0:r1, c0:c1] = True


                # interest_mask == band_mask 로 사용
                band_mask = interest_mask.copy()

                if not np.any(band_mask):
                    raise ValueError("[target_unknwown] band_mask is empty. Check frontier/occupied configuration.")

                # contour 포인트를 (row, col)로 변환
                # largest: shape (N,1,2) with (x,y) = (col,row)
                cnt_xy = largest[:, 0, :]              # (N,2) = (x,y)
                cnt_rc = np.stack([cnt_xy[:, 1],       # row = y
                                cnt_xy[:, 0]],      # col = x
                                axis=1).astype(float)   # (N,2) = (r,c)

                n_x = np.full((H, W), np.nan, dtype=float)
                n_y = np.full((H, W), np.nan, dtype=float)
                band_indices = np.argwhere(band_mask)

                # ------------------------------------------------------------
                # 여기서부터: inside_bool을 이용해 "테두리 바깥 방향" 법선 벡터 n_x, n_y 계산
                # ------------------------------------------------------------
                # inside=1, outside=0
                solid = inside_bool.astype(np.float32)   # (H,W), 0 or 1

                # 너무 날카로운 edge를 조금 부드럽게 → 정상적인 gradient field
                solid_blur = cv2.GaussianBlur(solid, (5, 5), 0)

                # row, col 방향 gradient (중심차분)
                grad_r = np.zeros_like(solid_blur, dtype=float)  # d/d(row)
                grad_c = np.zeros_like(solid_blur, dtype=float)  # d/d(col)

                grad_r[1:-1, :] = 0.5 * (solid_blur[2:, :] - solid_blur[:-2, :])
                grad_c[:, 1:-1] = 0.5 * (solid_blur[:, 2:] - solid_blur[:, :-2])

                # band 바깥은 신경 안 쓰므로 NaN 처리
                grad_r[~band_mask] = np.nan
                grad_c[~band_mask] = np.nan

                # magnitude
                mag = np.sqrt(grad_r**2 + grad_c**2)

                # outward normal = -grad / ||grad||   (grad는 inside(1) 쪽 → 안쪽, 우리는 바깥 방향이 필요)
                n_x = np.full((H, W), np.nan, dtype=float)
                n_y = np.full((H, W), np.nan, dtype=float)

                valid = band_mask & (mag > 1e-6)
                n_x[valid] = -grad_r[valid] / mag[valid]
                n_y[valid] = -grad_c[valid] / mag[valid]

                # ------------------------------------------------------------
                # 7) 모든 band cell에서 outward normal 방향으로 offset 나간 지점을 타깃 후보로 생성
                #    (cos_align, 점수 기반 필터링은 사용하지 않음)
                # ------------------------------------------------------------
                band_indices = np.argwhere(band_mask)

                num_agents   = self.num_agent
                offset_cells = 10   # grid 상에서 몇 cell 나갈지 (튜닝 파라미터)
                targets_raw = []

                for r, c in band_indices:
                    nx = n_x[r, c]
                    ny = n_y[r, c]
                    
                    if occ_mask[r, c]:
                        continue

                    if np.isnan(nx) or np.isnan(ny):
                        continue

                    # outward normal 방향으로 offset_cells 만큼 나간 지점
                    r_t = int(round(r + offset_cells * nx))
                    c_t = int(round(c + offset_cells * ny))

                    # 맵 밖이면 스킵
                    if not (0 <= r_t < H and 0 <= c_t < W):
                        continue
                    
                    # contour 내부로 들어간 셀은 제외
                    if inside_bool[r_t, c_t]:
                        continue

                    if not unknown_mask[r_t, c_t]:
                        continue

                    r0 = max(0, r_t - 3)
                    r1 = min(H, r_t + 3)  
                    c0 = max(0, c_t - 3)
                    c1 = min(W, c_t + 3)

                    if np.any(occ_mask[r0:r1, c0:c1]):
                        continue

                    targets_raw.append([r_t, c_t])

                if len(targets_raw) == 0:
                    raise ValueError("[target_unknwown] No valid targets generated along outward normals (band → offset).")

                # ---------------------------------------
                # 1) targets_raw -> binary cand_mask 생성
                # ---------------------------------------
                targets_raw = np.asarray(targets_raw, dtype=int)

                if targets_raw.shape[0] == 0:
                    raise ValueError("[target_unknown] targets_raw is empty before clustering.")

                targets_rc_all = np.unique(targets_raw, axis=0)
                H, W = occ_mask.shape

                cand_mask = np.zeros((H, W), dtype=np.uint8)
                cand_mask[targets_rc_all[:, 0], targets_rc_all[:, 1]] = 1

                # ---------------------------------------
                # 2) cand_mask inflate (장애물은 항상 제외)
                # ---------------------------------------
                inflate_radius = 2  # 필요 시 cfg로 빼도 됨
                if inflate_radius > 0:
                    k_size = 2 * inflate_radius + 1
                    kernel = np.ones((k_size, k_size), np.uint8)
                    cand_mask = cv2.dilate(cand_mask, kernel, iterations=1)

                # 장애물 셀은 무조건 후보에서 제거
                cand_mask[occ_mask] = 0

                if not np.any(cand_mask):
                    print("[target_unknown] cand_mask empty after dilation & obstacle removal; fallback to raw targets.")
                    cand_mask[targets_rc_all[:, 0], targets_rc_all[:, 1]] = 1  # raw targets만 사용

                # ---------------------------------------
                # 3) 8-연결 컴포넌트로 클러스터링
                # ---------------------------------------
                num_labels, labels = cv2.connectedComponents(cand_mask, connectivity=8)

                if num_labels <= 1:
                    # 클러스터가 없거나 전체가 한 덩어리
                    largest_cluster_mask = cand_mask.astype(bool)
                    areas = np.array([np.count_nonzero(largest_cluster_mask)], dtype=int)
                    best_label = 1
                else:
                    # -----------------------------
                    # 1) 각 라벨별 area, center 계산
                    # -----------------------------
                    num_comp = num_labels - 1  # 라벨 1..num_labels-1
                    areas   = np.zeros(num_comp, dtype=int)
                    centers = np.zeros((num_comp, 2), dtype=float)  # (row, col)

                    for idx, lbl in enumerate(range(1, num_labels)):
                        mask_lbl = (labels == lbl)
                        area_lbl = np.count_nonzero(mask_lbl)
                        areas[idx] = area_lbl

                        if area_lbl > 0:
                            rc = np.argwhere(mask_lbl)   # shape (area_lbl, 2) = (row,col)
                            centers[idx] = rc.mean(axis=0)
                        else:
                            centers[idx] = np.array([np.nan, np.nan])

                    # -----------------------------
                    # 2) area 기준 임계값 적용
                    # -----------------------------
                    area_max = int(areas.max()) if areas.size > 0 else 0
                    # 셀 개수 기준 임계값 (config 에서 가져오고, 없으면 기본값 사용)
                    area_thresh = 200

                    if area_max < area_thresh:
                        # 전체적으로 다 작으면: 그냥 가장 넓은 클러스터 선택 (기존 방식)
                        best_idx = int(np.argmax(areas))   # 0..num_comp-1
                    else:
                        # -----------------------------
                        # 3) 넓이가 충분히 큰 클러스터들 중에서
                        #    로봇 군집 중심과 가장 가까운 클러스터 선택
                        # -----------------------------
                        # 로봇 군집 중심 (world -> grid(row,col))
                        cluster_center_world = self.robot_locations.mean(axis=0)  # (x,y)
                        r_c, c_c = self.map_info.world_to_grid(
                            cluster_center_world[0], cluster_center_world[1]
                        )
                        robot_center_rc = np.array([r_c, c_c], dtype=float)

                        # area_thresh 이상인 라벨들만 후보
                        candidate_idx = np.where(areas >= area_thresh)[0]

                        if candidate_idx.size == 0:
                            # 방어적: 혹시라도 없으면 다시 기존 방식
                            best_idx = int(np.argmax(areas))
                        else:
                            # 각 후보의 중심까지 grid 상 거리
                            cand_centers = centers[candidate_idx]          # (K,2)
                            dists = np.linalg.norm(
                                cand_centers - robot_center_rc[None, :], axis=1
                            )                                              # (K,)
                            best_idx = int(candidate_idx[np.argmin(dists)])  # 0..num_comp-1

                    # best_idx 는 0..num_comp-1, 실제 라벨 번호는 1..num_labels-1
                    best_label = 1 + best_idx
                    largest_cluster_mask = (labels == best_label)

                print(
                        f"[target_unknown] {num_comp} clusters, "
                        f"area_max={area_max}, area_thresh={area_thresh}, "
                        f"chosen_label={best_label}, chosen_area={areas[best_idx]}"
                    )

                # ---------------------------------------
                # 4) largest_cluster_mask의 "모든 셀"을 샘플링 후보로 사용
                # ---------------------------------------
                largest_cluster_rc = np.argwhere(largest_cluster_mask)  # (M,2) = (row,col)

                if largest_cluster_rc.shape[0] == 0:
                    print("[target_unknown] largest cluster has no cells; fallback to targets_rc_all.")
                    largest_cluster_rc = targets_rc_all.copy()

                targets_base = largest_cluster_rc   # 샘플링 기준 배열은 이것 하나만 사용
                M = targets_base.shape[0]

                # ---------------------------------------
                # 5) targets_base에서 에이전트 수만큼 샘플링 (spread 제한 포함)
                # ---------------------------------------
                num_agents = self.num_agent
                rng = np.random.default_rng()
                max_pairwise_cells = 80.0  # 필요 시 cfg에서 가져오기
                max_trials = 100

                sampled_idx = None

                for _ in range(max_trials):
                    if M >= num_agents:
                        candidate_idx = rng.choice(M, size=num_agents, replace=False)
                    else:
                        candidate_idx = rng.choice(M, size=num_agents, replace=True)

                    cand = targets_base[candidate_idx]   # (N,2), 여기서 N=num_agents

                    diff = cand[None, :, :] - cand[:, None, :]
                    dist_pair = np.linalg.norm(diff, axis=2)
                    spread = float(dist_pair.max())

                    if spread > max_pairwise_cells:
                        continue
                    else:
                        sampled_idx = candidate_idx
                        break

                if sampled_idx is None:
                    print(f"[target_unknown] could not sample {num_agents} targets within "
                        f"max_pairwise_cells={max_pairwise_cells} from largest cluster. "
                        f"Falling back to core-based sampling.")

                    K_max = min(num_agents, M)
                    core_idx = None
                    K_core = 0

                    for K in range(K_max, 0, -1):
                        if M < K:
                            continue
                        cand_idx = rng.choice(M, size=K, replace=False)
                        cand_pts = targets_base[cand_idx]

                        diff = cand_pts[None, :, :] - cand_pts[:, None, :]
                        dist_pair = np.linalg.norm(diff, axis=2)
                        spread_K = float(dist_pair.max())

                        if spread_K <= max_pairwise_cells:
                            core_idx = cand_idx
                            K_core = K
                            break

                    if core_idx is None:
                        base_idx = int(rng.integers(0, M))
                        core_idx = np.array([base_idx], dtype=int)
                        K_core = 1
                        print(f"[target_unknown] fallback: using 1 core target index={base_idx} (largest cluster).")

                    if K_core >= num_agents:
                        sampled_idx = rng.choice(core_idx, size=num_agents, replace=False)
                    else:
                        sampled_idx = rng.choice(core_idx, size=num_agents, replace=True)

                # 여기서 sampled_idx 는 반드시 0 <= idx < M 범위
                targets_rc = targets_base[sampled_idx]

                # ------------------------------------------------------------
                # 8) Heat map: band=1.0, 타깃 후보=0.5
                # ------------------------------------------------------------
                inflate_radius = 2   # band를 시각화용으로 조금 부풀리기
                k_size = 2 * inflate_radius + 1
                kernel = np.ones((k_size, k_size), np.uint8)

                band_inflated_u8 = cv2.dilate(band_mask.astype(np.uint8), kernel, iterations=1)
                band_inflated = band_inflated_u8.astype(bool)

                heat = np.full((H, W), np.nan, dtype=float)

                # 1) band 전체를 1.0으로 칠하기
                heat[band_inflated] = 1.0

                heat[largest_cluster_mask] = 0.5

                self.targets_prob_heat = heat
                 # ------------------------------------------------------------
                # 9) world 좌표 변환 (root 선택 및 시각화용)
                # ------------------------------------------------------------
                target_world = []
                for (r_c, c_c) in targets_rc:
                    x, y = self.map_info.grid_to_world(r_c, c_c)
                    target_world.append([x, y])
                target_world = np.asarray(target_world, dtype=float)

                # ------------------------------------------------------------
                # 10) root 선택 + D-MST 업데이트 (region 모드와 동일 전략)
                # ------------------------------------------------------------
                if target_world.shape[0] > 0:
                    radius_root = 0.5
                    counts     = np.zeros(self.num_agent, dtype=int)
                    mean_dists = np.zeros(self.num_agent, dtype=float)

                    for i in range(self.num_agent):
                        pos_i = self.robot_locations[i]
                        dists = np.linalg.norm(target_world - pos_i, axis=1)

                        counts[i]     = np.sum(dists <= radius_root)
                        mean_dists[i] = np.mean(dists)

                    if np.any(counts > 0):
                        root_id = int(np.argmax(counts))
                    else:
                        root_id = int(np.argmin(mean_dists))
                else:
                    root_id = 0  # fallback

                #Proposed
                self.root_mask.fill(0)
                self.root_mask[root_id] = 1
                self.connectivity_graph.update_and_compute_mst(self.robot_locations, root_id)
                

                # ------------------------------------------------------------
                # 11) Hungarian matching으로 에이전트 ↔ 타깃 할당
                #      targets_rc: (row, col) 셀 좌표
                # ------------------------------------------------------------
                assigned_rc = assign_targets_hungarian(
                    self.map_info, self.robot_locations, targets_rc, self.num_agent
                )

                self.regions       = None
                self.valid_regions = None
                self.cluster_infos = {}
                self.assigned_rc   = np.asarray(assigned_rc, dtype=int)
                # Hungarian에서 (row, col)로 나왔다고 가정 → (col, row)로 맞춤
                self.assigned_rc   = self.assigned_rc[:, ::-1].copy()
                self.assigned_rc_viz = self.assigned_rc.copy()

                print('Team decision')

            elif self.assign_mode == "target_frontier":
                # ------------------------------------------------
                # Frontier 기반 타깃 선정: 모든 Frontier 점수화 -> 상위 N개 선택
                # ------------------------------------------------
                maps = self.map_info
                bel  = maps.belief
                bf   = maps.belief_frontier
                mm   = maps.map_mask
                H, W = maps.H, maps.W

                FRONTIER_LABEL = mm["frontier"]
                UNKNOWN_LABEL  = mm["unknown"]
                OCCUPIED_LABEL = mm["occupied"]

                frontier_mask   = (bf == FRONTIER_LABEL)
                unknown_mask    = (bel == UNKNOWN_LABEL)
                occ_mask        = (bel == OCCUPIED_LABEL)

                import cv2

                # 1) 장애물 주변 Frontier 필터링 (Safety Margin)
                clearance_m     = 0.5 * self.cfg.d_obs
                clearance_cells = max(1, int(round(clearance_m / maps.res_m)))
                k_size  = 2 * clearance_cells + 1
                kernel  = np.ones((k_size, k_size), np.uint8)

                inflated_occ = cv2.dilate(occ_mask.astype(np.uint8), kernel, iterations=1)
                frontier_mask_safe = frontier_mask & (~inflated_occ.astype(bool))
                frontier_rc_all = np.argwhere(frontier_mask_safe) # (M, 2)

                num_agents = self.num_agent
                M = frontier_rc_all.shape[0]

                if M == 0:
                    raise ValueError("[target_frontier] No safe frontier cells found.")

                # 2) 에이전트별 위치를 그리드 좌표로 변환
                agent_rc = np.array([maps.world_to_grid(self.robot_locations[i, 0], self.robot_locations[i, 1]) 
                                    for i in range(num_agents)], dtype=float)  # (num_agents, 2)

                # 3) 모든 Frontier에 대해 에이전트별 Heuristic Score 계산
                window_r = 5
                max_dim  = float(max(H, W))
                scores = np.zeros((num_agents, M), dtype=float)  # (num_agents, M_frontiers)

                # 가중치 설정
                w_u = 1.0  # 주변 미탐사 영역 비중
                w_d = 0.3  # 각 에이전트까지의 거리 페널티
                w_o = 0.5  # 주변 장애물 비중 페널티

                for idx, (r, c) in enumerate(frontier_rc_all):
                    r0 = max(0, r - window_r); r1 = min(H, r + window_r + 1)
                    c0 = max(0, c - window_r); c1 = min(W, c + window_r + 1)

                    patch_unknown = unknown_mask[r0:r1, c0:c1]
                    patch_occ     = occ_mask[r0:r1, c0:c1]

                    u_score = patch_unknown.mean() if patch_unknown.size > 0 else 0.0
                    o_score = patch_occ.mean() if patch_occ.size > 0 else 0.0
                    
                    # 각 에이전트별로 거리 점수 계산
                    for agent_i in range(num_agents):
                        d_score = np.linalg.norm(np.array([r, c]) - agent_rc[agent_i]) / max_dim
                        scores[agent_i, idx] = (w_u * u_score) - (w_d * d_score) - (w_o * o_score)

                # 4) Hungarian matching으로 에이전트-frontier 최적 할당
                from scipy.optimize import linear_sum_assignment
                
                if M >= num_agents:
                    # Case A: Frontier가 충분함 -> Hungarian matching으로 최적 할당
                    cost_matrix = -scores  # 최대값 찾기 위해 부호 반전
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    idx_selected = col_ind  # 선택된 frontier 인덱스
                else:
                    # Case B: Frontier가 부족함 -> 각 에이전트가 자신의 최고점 frontier 선택
                    # (중복 허용, 충돌은 하위 HOCBF가 처리)
                    idx_selected = np.argmax(scores, axis=1)  # (num_agents,)

                targets_rc = frontier_rc_all[idx_selected]  # 최종 (num_agents, 2) 결정

                # 5) 후속 처리 (MST 업데이트 및 Hungarian Matching)
                target_world = []
                for (rr, cc) in targets_rc:
                    x, y = self.map_info.grid_to_world(rr, cc)
                    target_world.append([x, y])
                target_world = np.asarray(target_world, dtype=float)

                # Root 선정 (타깃들이 뭉쳐있는 곳에 가까운 에이전트)
                mean_dists = np.mean(np.linalg.norm(target_world[None, :, :] - self.robot_locations[:, None, :], axis=2), axis=1)
                root_id = int(np.argmin(mean_dists))

                self.root_mask.fill(0)
                self.root_mask[root_id] = 1
                self.connectivity_graph.update_and_compute_mst(self.robot_locations, root_id)

                # 에이전트와 타깃 매칭 (scoring 기반)
                self.assigned_rc = np.asarray(targets_rc, dtype=int)  # (row, col)
                self.assigned_rc = self.assigned_rc[:, ::-1].copy()    # (col, row)로 변환
                self.assigned_rc_viz = self.assigned_rc.copy()

                self.regions        = None
                self.valid_regions  = None
                self.cluster_infos  = {}

                print(f"[target_frontier] {M} safe frontiers available")
                print(f"[target_frontier] Assigned {num_agents} targets via scoring-based Hungarian matching")
                print(f"[target_frontier] Root agent: {root_id}, Team decision complete")
                
            else:
                raise ValueError(f"Unknown assign_mode: {self.assign_mode}")

        for i in range(self.num_agent):
            pos_i = self.robot_locations[i]
            yaw_i = self.robot_angles[i]

            # Obstacle Info
            num_obs = self.num_obstacles[i]
            p_obs_list.append(self.obstacle_states[i, :num_obs])
            # Neighbor Agent Info
            num_neighbors = self.num_neighbors[i]
            p_agents_list.append(self.neighbor_states[i, :num_neighbors, :2])
            v_agents_local_list.append(self.neighbor_states[i, :num_neighbors, 2:])

            # ==== Target Agent Info ====
            # Root Node : Parent Node 존재 X
            # Leaf Node : CHild Node 존재 X
            # Reciprocal Connectivity Relationship:
            #   1. Parent Node : Child Node와 HOCBF 제약
            #   2. Child Node : Parent Node까지 A* Optimal Path
            #   3. Root Node : Child Node에 대해서 Only HOCBF제약
            #   4. Leaf Node : Parents Node에 대해서 Only A* Optimal Path

            #Proposed
            parent_id = self.connectivity_graph.get_parent(i)
            child_id = self.connectivity_graph.get_child(i)


            if parent_id == -1:
                # Root Node : Only HOCBF
                pos_i_c = self.robot_locations[child_id]
                vel_i_c = self.robot_velocities[child_id]
                if pos_i_c.shape[0] > 1:
                    # MST는 여러개의 자식노드가 있을 수 있음 -> Closest로 지정
                    min_ids = np.argmin(np.linalg.norm(pos_i-pos_i_c, axis=1))
                    pos_i_cbf = pos_i_c[min_ids]
                    vel_i_cbf = vel_i_c[min_ids]
                else:
                    pos_i_cbf = pos_i_c.reshape(-1)
                    vel_i_cbf = vel_i_c.reshape(-1)
            elif child_id is None:
                # Leaf Node : Only A*
                pos_i_op = self.robot_locations[parent_id]
            else:
                # Other Node
                pos_i_op = self.robot_locations[parent_id]

                pos_i_c = self.robot_locations[child_id]
                vel_i_c = self.robot_velocities[child_id]
                if pos_i_c.shape[0] > 1:
                    # MST는 여러개의 자식노드가 있을 수 있음 -> Closest로 지정
                    min_ids = np.argmin(np.linalg.norm(pos_i-pos_i_c, axis=1))
                    pos_i_cbf = pos_i_c[min_ids]
                    vel_i_cbf = vel_i_c[min_ids]
                else:
                    pos_i_cbf = pos_i_c.reshape(-1)
                    vel_i_cbf = vel_i_c.reshape(-1)

            # Control Barrier Function Info for Backward Connectivity
            if child_id is None:
                # Leaf Node : CBF 적용 X, A*를 위한 Position만 할당
                p_p = world_to_local(w1=pos_i, w2=pos_i_op, yaw=yaw_i) # Parent for Connectivity A*
                p_c_agent_list.append(np.array([]))
                v_c_agent_list.append(np.array([]))
            else:
                # Child Node가 있는 Node들 {Root Node, Other Node}
                if parent_id == -1:
                    p_p = np.array([0, 0]) # Root Node don't have the parent node
                else:
                    p_p = world_to_local(w1=pos_i, w2=pos_i_op, yaw=yaw_i)  # Parent for Connectivity A*
                p_c = world_to_local(w1=pos_i, w2=pos_i_cbf, yaw=yaw_i)     # Child for Connectivity HOCBF 
                v_c = world_to_local(w1=None, w2=vel_i_cbf, yaw=yaw_i)
                p_c_agent_list.append(p_c)
                v_c_agent_list.append(v_c)

            # Target Position Info & Forward Connectivity Info
            min_dist = np.linalg.norm(p_p)

            if (parent_id == -1):
                # 루트는 무조건 TARGET 쪽으로
                follower = False
                start_cell = self.map_info.world_to_grid_np(pos_i)
                end_cell = self.assigned_rc[i]
                end_world = self.map_info.grid_to_world_np(end_cell)
            else:
                min_dist = np.linalg.norm(p_p)
                overlap = (np.linalg.norm(self.assigned_rc[i] - self.assigned_rc[parent_id]) < 10)

                if (min_dist < self.cfg.d_max-0.1) and (not overlap):
                    follower = False
                    start_cell = self.map_info.world_to_grid_np(pos_i)
                    end_cell = self.assigned_rc[i]
                    end_world = self.map_info.grid_to_world_np(end_cell)
                else:
                    follower = True
                    start_cell = self.map_info.world_to_grid_np(pos_i)
                    end_cell = self.map_info.world_to_grid_np(pos_i_op)
                    end_world = pos_i_op


            # 이후 A* 탐색 및 리스트 추가 로직은 동일하게 유지
            # ----- 라우팅 요약 로그 생성 -----
            if not follower:
                # 그냥 타깃으로 가는 경우: 이유는 안 붙임
                if parent_id == -1:
                    log_line = f"Agent {i}: --> TARGET (root)"
                else:
                    log_line = f"Agent {i}: --> TARGET"
            else:
                # follower = True 인 경우에만 "왜 follower가 되었는지" 이유를 상세히 남김
                reasons = []

                # 1) 부모 관계 (여기선 항상 parent_id != -1)
                reasons.append(f"parent={parent_id}")

                # 2) 거리 조건
                if min_dist >= self.cfg.d_max - 0.1:
                    reasons.append(f"dist_to_parent={min_dist:.2f} ≥ d_max-0.1")

                # 3) 타깃 중첩
                if overlap:
                    reasons.append("target cell overlaps with parent")

                if not reasons:
                    reasons.append("connectivity at risk")

                log_reason = "; ".join(reasons)
                log_line = f"Agent {i}: --> Agent {parent_id} | reason: {log_reason}"
                #log_line = f"Agent {i}: --> TARGET (Greedy/No-Role)"
            routing_log_lines.append(log_line)

            if self.no_path_until_refresh[i]:
                target_pos = np.array([0.0, 0.0])
                target_pos_list.append(target_pos)
                end_pos_world_list.append(end_world.reshape(-1))
                follower_list.append(follower)
                # 이 에이전트에 대해서는 더 계산할 것 없이 다음 에이전트로
                continue

            # A* Graph Search for Optimal Path
            look_ahead_distance = 0.1
            path_cells = astar_search(self.map_info,
                                       start_pos=np.flip(start_cell), # (row, col)
                                       end_pos=np.flip(end_cell), # (row, col)
                                       agent_id=i)

            if path_cells is not None and len(path_cells) > 0:
                optimal_traj = self.map_info.grid_to_world_np(np.flip(np.array(path_cells), axis=1))
                connectivity_traj[i].append(optimal_traj)
                optimal_traj_local = world_to_local(w1=pos_i, w2=optimal_traj, yaw=yaw_i)
                distance_traj = np.linalg.norm(optimal_traj_local, axis=1)
                ids = np.argwhere(distance_traj >= look_ahead_distance)
                if ids.size > 0:
                    target_pos = optimal_traj_local[ids[0][0]]
                else:
                    target_pos = optimal_traj_local[-1]
            else:
                # 경로가 안 나왔을 때: world 좌표로 디버그 메시지만 출력
                self.no_path_until_refresh[i] = True
                start_world = pos_i              # 이미 world 좌표 (x, y)
                # end_world는 위에서 start_cell/end_cell 설정할 때 같이 계산됨
                print(
                    f"[A*] No valid path for agent {i}: "
                )
                # fallback: 제자리 명령
                target_pos = np.array([0.0, 0.0])
            
            target_pos_list.append(target_pos)
            end_pos_world_list.append(end_world.reshape(-1))
            follower_list.append(follower)
        
        self.end_pos_world = np.array(end_pos_world_list)
        self.connectivity_traj = connectivity_traj

        if routing_log_lines:
            print("\n[Routing summary]")
            for line in routing_log_lines:
                print("  " + line)
            print()

        self.end_pos_world = np.array(end_pos_world_list)
        self.connectivity_traj = connectivity_traj

        infos["safety"] = {
            "v_current": list(self.robot_speeds),
            "p_obs": p_obs_list,
            "p_agents": p_agents_list,
            "v_agents_local": v_agents_local_list,
            "p_c_agent": p_c_agent_list,
            "v_c_agent": v_c_agent_list
        }

        infos["nominal"] = {
            "p_targets": target_pos_list,
            "follower": follower_list
        }

        self.tree_interval += 1

        return infos
    
    # ============= Auxilary Methods ==============
    def sense_and_update(self, 
                     agent_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Raycast in FOV, update belief FREE/OCCUPIED 
                Inputs:
                    - agent_id: Agent Numbering
                Return:
                    - frontier_local:  [(lx, ly), ...]  
                    - frontier_rc:     [(row, col), ...] 
                    - obs_local:       [(lx, ly), ...] 
        """
        drone_pose = np.hstack((self.robot_locations[agent_id], self.robot_angles[agent_id]))
        maps = self.map_info
        H, W = maps.H, maps.W
        half = math.radians(self.fov / 2.0)
        angles = np.linspace(-half, half, self.cfg.num_rays)

        # ======== 추가: goal 도 unknown-like 이웃으로 취급 ========
        mm = maps.map_mask
        UNKNOWN_LABEL = mm["unknown"]
        GOAL_LABEL    = mm["goal"]
        # belief 기준 unknown 이거나, GT 기준 goal 이면 "unknown 비슷"
        unknown_like = (maps.belief == UNKNOWN_LABEL) | (maps.gt == GOAL_LABEL)
        # ====================================================

        frontier_local: List[Tuple[float, float]] = []
        frontier_rc: List[Tuple[int, int]] = []
        obs_local: List[Tuple[float, float]] = []

        for a in angles:
            ang = drone_pose[2] + a
            step = maps.res_m
            L = int(self.sensor_range / step)

            last_rc = None
            hit_recorded = False          # per-ray: obs 최대 1개
            frontier_candidate_rc = None  # per-ray: frontier 후보(마지막 FREE∧UNKNOWN-like-인접)

            for i in range(1, L + 1):
                x = drone_pose[0] + i * step * math.cos(ang)
                y = drone_pose[1] + i * step * math.sin(ang)
                if x < 0 or y < 0 or x > maps.meters_w or y > maps.meters_h:
                    break

                r, c = maps.world_to_grid(x, y)
                if last_rc == (r, c):
                    continue
                last_rc = (r, c)

                if maps.gt[r, c] == maps.map_mask["occupied"]:
                    # 첫 OCC 히트만 기록
                    maps.belief[r, c] = maps.map_mask["occupied"]
                    if not hit_recorded:
                        dx = x - drone_pose[0]; dy = y - drone_pose[1]
                        cth = math.cos(-drone_pose[2]); sth = math.sin(-drone_pose[2])
                        lx = cth*dx - sth*dy; ly = sth*dx + cth*dy
                        obs_local.append((lx, ly))
                        hit_recorded = True
                    break  # 이 ray 종료 (더 이상 진행 X)

                else:
                    # 관측된 FREE 갱신 (start는 보존)
                    if maps.belief[r, c] != maps.map_mask["start"]:
                        maps.belief[r, c] = maps.map_mask["free"]

                    # 이 셀의 8-이웃 중 UNKNOWN-like 이 있으면 'frontier 후보'
                    found_unknown_like = False
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr = r + dr; cc = c + dc
                            if 0 <= rr < H and 0 <= cc < W and unknown_like[rr, cc]:
                                found_unknown_like = True
                                break
                        if found_unknown_like:
                            break

                    # frontier 후보는 ray를 따라 '마지막으로' 갱신하여, 경계에 가장 가까운 FREE를 선택
                    if found_unknown_like:
                        frontier_candidate_rc = (r, c)

            # ray가 끝난 뒤, 후보가 있으면 frontier를 1개만 최종 채택
            if frontier_candidate_rc is not None:
                r, c = frontier_candidate_rc
                wx, wy = maps.grid_to_world(r, c)
                dx = wx - drone_pose[0]; dy = wy - drone_pose[1]
                cth = math.cos(-drone_pose[2]); sth = math.sin(-drone_pose[2])
                lx = cth*dx - sth*dy; ly = sth*dx + cth*dy
                frontier_local.append((lx, ly))
                frontier_rc.append((r, c))

        return np.array(frontier_local), np.array(frontier_rc), np.array(obs_local)