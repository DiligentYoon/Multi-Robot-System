import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .base_planner import AbstractPlanner


class FrontierPlanner(AbstractPlanner):
    def plan(self, map_info, robot_locations, robot_velocities,
             num_agent, cfg) -> dict:

        maps = map_info
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

        # 1) 장애물 주변 Frontier 필터링 (Safety Margin)
        clearance_m     = 0.5 * cfg.d_obs
        clearance_cells = max(1, int(round(clearance_m / maps.res_m)))
        k_size  = 2 * clearance_cells + 1
        kernel  = np.ones((k_size, k_size), np.uint8)

        inflated_occ = cv2.dilate(occ_mask.astype(np.uint8), kernel, iterations=1)
        frontier_mask_safe = frontier_mask & (~inflated_occ.astype(bool))
        frontier_rc_all = np.argwhere(frontier_mask_safe) # (M, 2)

        M = frontier_rc_all.shape[0]

        if M == 0:
            raise ValueError("[target_frontier] No safe frontier cells found.")

        # 2) 에이전트별 위치를 그리드 좌표로 변환
        agent_rc = np.array([maps.world_to_grid(robot_locations[i, 0], robot_locations[i, 1])
                             for i in range(num_agent)], dtype=float)  # (num_agents, 2)

        # 3) 모든 Frontier에 대해 에이전트별 Heuristic Score 계산
        window_r = 5
        max_dim  = float(max(H, W))
        scores = np.zeros((num_agent, M), dtype=float)  # (num_agents, M_frontiers)

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
            for agent_i in range(num_agent):
                d_score = np.linalg.norm(np.array([r, c]) - agent_rc[agent_i]) / max_dim
                scores[agent_i, idx] = (w_u * u_score) - (w_d * d_score) - (w_o * o_score)

        # 4) Hungarian matching으로 에이전트-frontier 최적 할당
        if M >= num_agent:
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
            x, y = map_info.grid_to_world(rr, cc)
            target_world.append([x, y])
        target_world = np.asarray(target_world, dtype=float)

        # Root 선정 (타깃들이 뭉쳐있는 곳에 가까운 에이전트)
        mean_dists = np.mean(np.linalg.norm(target_world[None, :, :] - robot_locations[:, None, :], axis=2), axis=1)
        root_id = int(np.argmin(mean_dists))

        # 에이전트와 타깃 매칭 (scoring 기반)
        assigned_rc = np.asarray(targets_rc, dtype=int)  # (row, col)
        assigned_rc = assigned_rc[:, ::-1].copy()         # (col, row)로 변환

        print(f"[target_frontier] {M} safe frontiers available")
        print(f"[target_frontier] Assigned {num_agent} targets via scoring-based Hungarian matching")
        print(f"[target_frontier] Root agent: {root_id}, Team decision complete")

        return {
            "assigned_rc": assigned_rc,
            "root_id"    : root_id,
            "viz": {
                "targets_prob_heat": None,
                "assigned_rc_viz"  : assigned_rc.copy(),
                "cluster_infos"    : {},
            }
        }
