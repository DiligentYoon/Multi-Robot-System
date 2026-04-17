import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp

from cvxpylayers.torch import CvxpyLayer
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from typing import Dict, List

import warnings
warnings.filterwarnings("ignore", message="Converting G to a CSC matrix; may take a while.")

import sys
if sys.platform == "win32":
    try:
        import contextlib
        import diffcp.cone_program as dc

        def _dummy_threadpool_limits(*args, **kwargs):
            return contextlib.nullcontext()

        dc.threadpool_limits = _dummy_threadpool_limits
    except Exception:
        pass

class DifferentiableCBFLayer(nn.Module):
    def __init__(self, cfg: Dict, device: torch.device):
        super().__init__()
        # 설정값(hyperparameters) 저장
        self.cfg = cfg
        self.max_obs = cfg['max_obs']
        self.max_agents = cfg['max_agents']
        self.max_neighbors = self.max_agents - 1
        
        self.num_input = 2
        self.num_slack = 3

        self.num_vars = self.num_input + self.num_slack
        self.num_hocbf_constraints = self.max_obs + self.max_agents + 1 # obs + agent-collision + connectivity
        # self.num_hocbf_constraints = self.max_obs + 1 # obs + agent-collision + connectivity
        self.num_box_constraints = 4
        self.num_constraints = self.num_hocbf_constraints + self.num_slack

        # Constraints Info
        self.a_max = cfg['a_max']
        self.w_max = cfg['w_max']

        self.d_max =  cfg['d_max']
        self.d_obs = cfg['d_obs']
        self.d_safe = cfg['d_safe']

        self.damping = cfg['gamma_c1'] + cfg['gamma_c2']
        self.stiffness = cfg['gamma_c1'] * cfg['gamma_c2']

        self.damping_agent = cfg['gamma_a1'] + cfg['gamma_a2']
        self.stiffness_agent = cfg['gamma_a1'] * cfg['gamma_a2']

        self.damping_conn = cfg['gamma_conn1'] + cfg['gamma_conn2']
        self.stiffness_conn = cfg['gamma_conn1'] * cfg['gamma_conn2']

        # Objective Info
        self.w_slack = cfg['w_slack']

        self.dtype = torch.double 
        # self.device = torch.device('cpu')
        self.device = device

        # Action Scaler
        self.action_scale = torch.tensor([self.a_max, self.w_max], device=self.device)

        # --- 1. 최적화 변수 정의 ---
        u = cp.Variable(self.num_input, name='u')
        delta_obs = cp.Variable(1, name='delta_obs')
        delta_agent = cp.Variable(1, name='delta_agent')
        delta_conn = cp.Variable(1, name='delta_conn')
        x_vars_for_constraints = cp.hstack([u, delta_obs, delta_agent, delta_conn])

        # --- 2. 파라미터 정의 ---
        # 목적 함수용 파라미터
        u_ref = cp.Parameter(self.num_input, name='u_ref') 
        # 제약 조건용 파라미터
        G = cp.Parameter((self.num_constraints, self.num_vars), name='G')
        h = cp.Parameter(self.num_constraints, name='h')

        # --- 3. 목적 함수 정의 ---
        objective = cp.Minimize(
            cp.sum_squares(u - u_ref) + 
            self.cfg['w_slack'] * cp.sum_squares(delta_obs) + 
            self.cfg['w_slack'] * cp.sum_squares(delta_agent) + 
            self.cfg['w_slack'] * cp.sum_squares(delta_conn))

        # --- 4. 제약 조건 정의 (Gx <= h 행렬 형태 유지) ---
        constraints = [ G @ x_vars_for_constraints <= h ]
        constraints += [
            u[0] >= -self.cfg['a_max'], u[0] <= self.cfg['a_max'],
            u[1] >= -self.cfg['w_max'], u[1] <= self.cfg['w_max']]
        
        # --- 5. CvxpyLayer 생성 ---
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(
            problem,
            parameters=[u_ref, G, h],
            variables=[u, delta_obs, delta_agent, delta_conn])

    def forward(self, 
                u_nominal: torch.Tensor, 
                state: Dict[str, List[np.ndarray]] | List[Dict[str, List[np.ndarray]]]) -> tuple[torch.Tensor, bool]:
        """
        :param u_nominal: 제안된 공칭 제어 입력 (B & N, 2)
        :param state: 현재 상태 정보를 담은 딕셔너리
        :return: 안전 필터를 거친 최종 제어 입력 (B & N, 2)
        """
        # Handle training-time batch (List of Dictionaries)
        if isinstance(state, list):
            # Reshape u_nominal from (B_train, N_agents, 2) to (B_train * N_agents, 2)
            if u_nominal.dim() > 2:
                u_nominal = u_nominal.reshape(-1, u_nominal.shape[-1])

            # Flatten the list of dictionaries into a single large dictionary
            state_dict = {
                'v_current': [v for d in state for v in d['v_current']],
                'p_obs': [p for d in state for p in d['p_obs']],
                'p_agents': [p for d in state for p in d['p_agents']],
                'p_c_agent': [p for d in state for p in d['p_c_agent']],
                'v_agents_local': [v for d in state for v in d['v_agents_local']],
                'v_c_agent': [v for d in state for v in d['v_c_agent']]
            }
            state = state_dict # Overwrite state with the new flattened dictionary

        device = self.device
        dtype = self.dtype
        B = u_nominal.shape[0]

        # Process state dictionary and create padded tensors/masks
        v_current = torch.tensor(np.array(state['v_current']), device=device, dtype=dtype).unsqueeze(1)
        # Initialize padded tensors
        p_obs_padded = torch.zeros(B, self.max_obs, 2, device=device, dtype=dtype)
        p_agents_padded = torch.zeros(B, self.max_neighbors, 2, device=device, dtype=dtype)
        p_closest_agent = torch.zeros(B, 1, 2, device=device, dtype=dtype)
        v_agents_local_padded = torch.zeros(B, self.max_neighbors, 2, device=device, dtype=dtype)
        v_closest_agent = torch.zeros(B, 1, 2, device=device, dtype=dtype)
        
        # Initialize masks
        obs_mask = torch.zeros(B, self.max_obs, device=device, dtype=dtype)
        agents_mask = torch.zeros(B, self.max_neighbors, device=device, dtype=dtype)
        closest_mask = torch.zeros(B, 1, device=device, dtype=dtype)

        for i in range(B):
            # Obstacles
            obs_list = state['p_obs'][i]
            if len(obs_list) > 0:
                num_obs = min(len(obs_list), self.max_obs)
                p_obs_padded[i, :num_obs] = torch.tensor(obs_list[:num_obs], device=device, dtype=dtype)
                obs_mask[i, :num_obs] = 1.0
            # Agents
            p_agents_list = state['p_agents'][i]
            v_agents_list = state['v_agents_local'][i]
            if len(p_agents_list) > 0:
                num_neighbors = min(len(p_agents_list), self.max_neighbors)
                p_agents_padded[i, :num_neighbors] = torch.tensor(p_agents_list[:num_neighbors], device=device, dtype=dtype)
                v_agents_local_padded[i, :num_neighbors] = torch.tensor(v_agents_list[:num_neighbors], device=device, dtype=dtype)
                agents_mask[i, :num_neighbors] = 1.0
            # Connectivity Agent
            p_closest_agent_list = state['p_c_agent'][i]
            v_closest_agent_list = state['v_c_agent'][i]
            if len(p_closest_agent_list) > 0:
                p_closest_agent[i] = torch.tensor(p_closest_agent_list, device=device, dtype=dtype)
                v_closest_agent[i] = torch.tensor(v_closest_agent_list, device=device, dtype=dtype)
                closest_mask[i] = 1.0

        # n = self.num_input
        n = self.num_vars
        m = self.num_constraints
        # --- 제약 조건 행렬 G, h 계산 ---
        # 제약 조건 담을 때, 감지된 obs와 agent수 만큼만 유효하도록 마스킹 해야 함.
        G = torch.zeros(B, m, n, device=device, dtype=dtype)
        h = torch.full((B, m), 1e3, device=device, dtype=dtype)

        current_idx = 0
        # Box Constraints
        G[:, 0, 2] = -1.0; h[:, 0] = 0.0  # -delta <= 0
        G[:, 1, 3] = -1.0; h[:, 1] = 0.0
        G[:, 2, 4] = -1.0; h[:, 2] = 0.0
        current_idx = self.num_slack
    
        # --- 정적 장애물 제약 ---
        lx_obs, ly_obs = p_obs_padded[..., 0], p_obs_padded[..., 1]

        # Collision Avoidance (max_obs)
        h_obs = lx_obs**2 + ly_obs**2 - self.d_obs**2
        h_dot_obs = -2 * lx_obs * v_current
        h_rhs_obs = (2.0 * v_current**2 + self.damping * h_dot_obs + self.stiffness * h_obs)
        
        # Form : 2l_x * a + 2l_y * v * w <= 2v^2 + k_1 * \dot{h} + k_2 * h
        G_obs_a = (2 * lx_obs * obs_mask)
        G_obs_w = (2 * ly_obs * v_current * obs_mask)
        G_obs_delta = -1.0 * obs_mask
        G[:, current_idx:current_idx+self.max_obs, 0] = G_obs_a
        G[:, current_idx:current_idx+self.max_obs, 1] = G_obs_w
        G[:, current_idx:current_idx+self.max_obs, 2] = G_obs_delta                     
        h[:, current_idx:current_idx+self.max_obs] = h_rhs_obs
        h[:, current_idx:current_idx+self.max_obs][obs_mask == 0] = 1e3
        current_idx += self.max_obs

        # # --- 동적 에이전트 제약 ---
        lx_ag, ly_ag = p_agents_padded[..., 0], p_agents_padded[..., 1]
        v_jx, v_jy = v_agents_local_padded[..., 0], v_agents_local_padded[..., 1]
        
        # Collision Avoidance (max_agents)
        h_avoid = lx_ag**2 + ly_ag**2 - self.d_safe**2
        h_dot_avoid = -2*lx_ag*v_current + 2*(lx_ag*v_jx + ly_ag*v_jy)
        h_dot_dot_avoid_const = 2*v_current**2 - 2*v_current*v_jx + 2*(-v_current*v_jx + v_jx**2 + v_jy**2)
        h_rhs_avoid = h_dot_dot_avoid_const + self.damping_agent * h_dot_avoid + self.stiffness_agent * h_avoid

        G_avoid_a = 2.0 * lx_ag * agents_mask
        G_avoid_w = (2.0 * ly_ag * v_current - 2.0 * ly_ag * v_jx + 2.0 * lx_ag * v_jy) * agents_mask
        G_avoid_delta = -1.0 * agents_mask
        
        G[:, current_idx:current_idx+self.max_neighbors, 0] = G_avoid_a
        G[:, current_idx:current_idx+self.max_neighbors, 1] = G_avoid_w
        G[:, current_idx:current_idx+self.max_neighbors, 3] = G_avoid_delta
        h[:, current_idx:current_idx+self.max_neighbors] = h_rhs_avoid
        h[:, current_idx:current_idx+self.max_neighbors][agents_mask == 0] = 1e3
        current_idx += self.max_neighbors

        # Connectivity (1)
        clx_ag, cly_ag = p_closest_agent[..., 0], p_closest_agent[..., 1]
        cv_jx, cv_jy = v_closest_agent[..., 0], v_closest_agent[..., 1]
        h_conn = self.d_max**2 - (clx_ag**2 + cly_ag**2)
        h_dot_conn = -1.0 * (-2*clx_ag*v_current + 2*(clx_ag*cv_jx + cly_ag*cv_jy))
        h_dot_dot_conn_const = -1.0 * (2*v_current**2 - 2*v_current*cv_jx + 2*(-v_current*cv_jx + cv_jx**2 + cv_jy**2))
        h_rhs_conn = h_dot_dot_conn_const + self.damping_conn * h_dot_conn + self.stiffness_conn * h_conn

        G_conn_a = -1.0 * (2 * clx_ag) * closest_mask
        G_conn_w = -1.0 * (2 * cly_ag * v_current - 2 * cly_ag * cv_jx + 2 * clx_ag * cv_jy) * closest_mask
        G_conn_delta = torch.full((B, 1), -1.0, device=device, dtype=dtype) * closest_mask

        G[:, current_idx, 0] = G_conn_a.squeeze(-1)
        G[:, current_idx, 1] = G_conn_w.squeeze(-1)
        G[:, current_idx, 4] = G_conn_delta.squeeze(-1)
        h[:, current_idx] = h_rhs_conn.squeeze(-1)
        h[:, current_idx][(closest_mask == 0).squeeze()] = 1e3
        # h[:, current_idx] = 1e3

        if device == torch.device('cpu'):
            u_ref_ = (u_nominal.to(device=device, dtype=dtype)).cpu().contiguous()
            G_ = G.cpu().contiguous()
            h_ = h.cpu().contiguous()
        else:
            u_ref_ = u_nominal.to(device=device, dtype=dtype)
            G_ = G
            h_ = h
        solution = self.layer(u_ref_, G_, h_, solver_args={'solve_method': 'ECOS'})[0]
        feasible = True
        
        # viol = (G[:, :, :2] @ solution.unsqueeze(-1)).squeeze(-1) - h
        # print("max ineq viol per batch:", viol.max(dim=1).values.detach().cpu().numpy())

        # 솔루션에서 안전한 제어 입력 u_safe만 추출
        u_safe = solution.to(device=u_nominal.device, dtype=u_nominal.dtype)
        
        return u_safe, feasible
