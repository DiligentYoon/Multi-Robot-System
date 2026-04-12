# task/models/avoidance_hocbf.py

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class AvoidanceHOCBFLayer(nn.Module):
    """
    Static obstacle + inter-agent collision avoidance 전용 2계 HOCBF QP 레이어.

    - 입력 u_nominal: [-1, 1] 범위의 정규화된 (a_nom, w_nom)
    - 출력 u_safe_normalized: [-1, 1] 범위의 필터링된 (a_safe, w_safe)

    cfg에 필요한 키:
        max_obs, max_agents,
        d_obs, d_safe,
        a_max, w_max,
        gamma_c1, gamma_c2,
        gamma_a1, gamma_a2,
        w_slack
    """

    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__()
        self.cfg = cfg

        # -------------------------
        # Dimension / limits
        # -------------------------
        self.max_obs = cfg["max_obs"]
        self.max_agents = cfg["max_agents"]
        self.max_neighbors = self.max_agents - 1

        self.num_input = 2                 # [a, w]
        self.num_slack = 2                 # [delta_obs, delta_agent]
        self.num_vars = self.num_input + self.num_slack

        # HOCBF constraints: obs + agent-collision
        self.num_hocbf_constraints = self.max_obs + self.max_agents
        # Box on slack + HOCBF
        self.num_constraints = self.num_slack + self.num_hocbf_constraints

        # Physical limits
        self.a_max = cfg["a_max"]
        self.w_max = cfg["w_max"]

        self.d_obs = cfg["d_obs"]
        self.d_safe = cfg["d_safe"]

        # HOCBF gains for obstacles
        self.damping = cfg["gamma_c1"] + cfg["gamma_c2"]
        self.stiffness = cfg["gamma_c1"] * cfg["gamma_c2"]

        # HOCBF gains for inter-agent
        self.damping_agent = cfg["gamma_a1"] + cfg["gamma_a2"]
        self.stiffness_agent = cfg["gamma_a1"] * cfg["gamma_a2"]

        # Slack penalty
        self.w_slack = cfg["w_slack"]

        self.dtype = torch.double
        self.device = device

        # Action scaling: [-1,1] → physical → [-1,1]
        self.action_scale = torch.tensor(
            [self.a_max, self.w_max],
            device=self.device,
            dtype=self.dtype,
        )

        # -------------------------
        # Cvxpy QP 정의
        # -------------------------
        u = cp.Variable(self.num_input, name="u")  # [a, w]
        delta_obs = cp.Variable(1, name="delta_obs")
        delta_agent = cp.Variable(1, name="delta_agent")

        x_vars = cp.hstack([u, delta_obs, delta_agent])

        # Parameters
        u_ref = cp.Parameter(self.num_input, name="u_ref")
        G = cp.Parameter((self.num_constraints, self.num_vars), name="G")
        h = cp.Parameter(self.num_constraints, name="h")

        objective = cp.Minimize(
            cp.sum_squares(u - u_ref)
            + self.w_slack * cp.sum_squares(delta_obs)
            + self.w_slack * cp.sum_squares(delta_agent)
        )

        constraints = [G @ x_vars <= h]
        constraints += [
            u[0] >= -self.a_max,
            u[0] <= self.a_max,
            u[1] >= -self.w_max,
            u[1] <= self.w_max,
        ]

        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(
            problem,
            parameters=[u_ref, G, h],
            variables=[u, delta_obs, delta_agent],
        )

    def forward(
        self,
        u_nominal: torch.Tensor,
        state: Dict[str, List[np.ndarray]] | List[Dict[str, List[np.ndarray]]],
    ) -> tuple[torch.Tensor, bool]:
        """
        :param u_nominal: [-1, 1] 정규화된 공칭 입력 (B, 2)
        :param state: {
                'v_current': [v_i, ...],
                'p_obs': [np.ndarray(#obs_i, 2), ...],
                'p_agents': [np.ndarray(#nbrs_i, 2), ...],
                'v_agents_local': [np.ndarray(#nbrs_i, 2), ...],
                ... (p_c_agent, v_c_agent 등은 무시)
            }
        :return: (u_safe_normalized, feasible)
        """
        # training-time batch (List[Dict]) 처리
        if isinstance(state, list):
            if u_nominal.dim() > 2:
                u_nominal = u_nominal.reshape(-1, u_nominal.shape[-1])

            state_dict = {
                "v_current": [v for d in state for v in d["v_current"]],
                "p_obs": [p for d in state for p in d["p_obs"]],
                "p_agents": [p for d in state for p in d["p_agents"]],
                "v_agents_local": [v for d in state for v in d["v_agents_local"]],
            }
            state = state_dict

        device = self.device
        dtype = self.dtype
        B = u_nominal.shape[0]

        # -------------------------
        # 상태 → padded tensor
        # -------------------------
        v_current = torch.tensor(
            np.array(state["v_current"]), device=device, dtype=dtype
        ).unsqueeze(1)

        p_obs_padded = torch.zeros(B, self.max_obs, 2, device=device, dtype=dtype)
        p_agents_padded = torch.zeros(
            B, self.max_neighbors, 2, device=device, dtype=dtype
        )
        v_agents_local_padded = torch.zeros(
            B, self.max_neighbors, 2, device=device, dtype=dtype
        )

        obs_mask = torch.zeros(B, self.max_obs, device=device, dtype=dtype)
        agents_mask = torch.zeros(B, self.max_neighbors, device=device, dtype=dtype)

        for i in range(B):
            # Obstacles
            obs_list = state["p_obs"][i]
            if len(obs_list) > 0:
                num_obs = min(len(obs_list), self.max_obs)
                p_obs_padded[i, :num_obs] = torch.tensor(
                    obs_list[:num_obs], device=device, dtype=dtype
                )
                obs_mask[i, :num_obs] = 1.0

            # Agents
            p_agents_list = state["p_agents"][i]
            v_agents_list = state["v_agents_local"][i]
            if len(p_agents_list) > 0:
                num_neighbors = min(len(p_agents_list), self.max_neighbors)
                p_agents_padded[i, :num_neighbors] = torch.tensor(
                    p_agents_list[:num_neighbors], device=device, dtype=dtype
                )
                v_agents_local_padded[i, :num_neighbors] = torch.tensor(
                    v_agents_list[:num_neighbors], device=device, dtype=dtype
                )
                agents_mask[i, :num_neighbors] = 1.0

        n = self.num_vars
        m = self.num_constraints

        G = torch.zeros(B, m, n, device=device, dtype=dtype)
        h = torch.full((B, m), 1e3, device=device, dtype=dtype)

        current_idx = 0

        # -------------------------
        # Box constraints on slack: -delta <= 0
        # -------------------------
        G[:, 0, 2] = -1.0  # -delta_obs <= 0
        h[:, 0] = 0.0

        G[:, 1, 3] = -1.0  # -delta_agent <= 0
        h[:, 1] = 0.0

        current_idx = self.num_slack

        # -------------------------
        # Static obstacle HOCBF
        # -------------------------
        lx_obs, ly_obs = p_obs_padded[..., 0], p_obs_padded[..., 1]

        h_obs = lx_obs**2 + ly_obs**2 - self.d_obs**2
        h_dot_obs = -2.0 * lx_obs * v_current
        h_rhs_obs = 2.0 * v_current**2 + self.damping * h_dot_obs + self.stiffness * h_obs

        # Form: 2 lx a + 2 ly v w - delta_obs <= RHS
        G_obs_a = 2.0 * lx_obs * obs_mask
        G_obs_w = 2.0 * ly_obs * v_current * obs_mask
        G_obs_delta = -1.0 * obs_mask

        G[:, current_idx : current_idx + self.max_obs, 0] = G_obs_a
        G[:, current_idx : current_idx + self.max_obs, 1] = G_obs_w
        G[:, current_idx : current_idx + self.max_obs, 2] = G_obs_delta

        h[:, current_idx : current_idx + self.max_obs] = h_rhs_obs
        h[:, current_idx : current_idx + self.max_obs][obs_mask == 0] = 1e3
        current_idx += self.max_obs

        # -------------------------
        # Inter-agent HOCBF
        # -------------------------
        lx_ag, ly_ag = p_agents_padded[..., 0], p_agents_padded[..., 1]
        v_jx, v_jy = v_agents_local_padded[..., 0], v_agents_local_padded[..., 1]

        h_avoid = lx_ag**2 + ly_ag**2 - self.d_safe**2
        h_dot_avoid = -2.0 * lx_ag * v_current + 2.0 * (lx_ag * v_jx + ly_ag * v_jy)
        h_dot_dot_avoid_const = (
            2.0 * v_current**2
            - 2.0 * v_current * v_jx
            + 2.0 * (-v_current * v_jx + v_jx**2 + v_jy**2)
        )

        h_rhs_avoid = (
            h_dot_dot_avoid_const
            + self.damping_agent * h_dot_avoid
            + self.stiffness_agent * h_avoid
        )

        G_avoid_a = 2.0 * lx_ag * agents_mask
        G_avoid_w = (
            2.0 * ly_ag * v_current
            - 2.0 * ly_ag * v_jx
            + 2.0 * lx_ag * v_jy
        ) * agents_mask
        G_avoid_delta = -1.0 * agents_mask

        G[
            :,
            current_idx : current_idx + self.max_neighbors,
            0,
        ] = G_avoid_a
        G[
            :,
            current_idx : current_idx + self.max_neighbors,
            1,
        ] = G_avoid_w
        G[
            :,
            current_idx : current_idx + self.max_neighbors,
            3,
        ] = G_avoid_delta

        h[:, current_idx : current_idx + self.max_neighbors] = h_rhs_avoid
        h[:, current_idx : current_idx + self.max_neighbors][agents_mask == 0] = 1e3
        current_idx += self.max_neighbors

        # -------------------------
        # QP solve
        # -------------------------
        if device == torch.device("cpu"):
            u_ref_ = (
                u_nominal.to(device=device, dtype=dtype)
                * self.action_scale.to(device=device, dtype=dtype)
            ).cpu()
            G_ = G.cpu()
            h_ = h.cpu()
        else:
            u_ref_ = (
                u_nominal.to(device=device, dtype=dtype)
                * self.action_scale.to(device=device, dtype=dtype)
            )
            G_ = G
            h_ = h

        solution = self.layer(u_ref_, G_, h_, solver_args={"solve_method": "ECOS"})[0]
        feasible = True  # 현재는 solve 실패 여부를 별도로 체크하지 않음

        # Scale back to normalized [-1, 1]
        u_safe = solution.to(device=u_nominal.device, dtype=u_nominal.dtype)
        u_safe_normalized = u_safe / self.action_scale.to(u_nominal.device)

        return u_safe_normalized, feasible
