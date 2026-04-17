import os
import numpy as np
from PIL import Image
from abc import abstractmethod
from typing import Tuple, Optional

from .env_cfg import EnvCfg
from task.utils.sensor_utils import *

class MapInfo:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.map_mask = self.cfg["map_representation"]
        self.map_filepath = cfg.get("map_filepath")

        # --- 1. Create original gt map ---
        gt_original = None
        if self.map_filepath:
            try:
                script_dir = os.path.dirname(__file__)
                project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
                full_map_path = os.path.join(project_root, self.map_filepath)
                
                with Image.open(full_map_path) as img:
                    img_array = np.array(img.convert('RGB'))

                H_gt, W_gt, _ = img_array.shape
                gt_original = np.full((H_gt, W_gt), self.map_mask["free"], dtype=np.int8)

                BLACK = np.array([0, 0, 0])
                WHITE = np.array([255, 255, 255])
                GREEN = np.array([50, 200, 80])
                PURPLE = np.array([180, 50, 200])

                gt_original[np.all(img_array == BLACK, axis=-1)] = self.map_mask["occupied"]
                gt_original[np.all(img_array == WHITE, axis=-1)] = self.map_mask["free"]
                gt_original[np.all(img_array == GREEN, axis=-1)] = self.map_mask["start"]
                gt_original[np.all(img_array == PURPLE, axis=-1)] = self.map_mask["goal"]

                self.res_m = cfg.get("resolution", 0.01)
                meters_h_gt = H_gt * self.res_m
                meters_w_gt = W_gt * self.res_m

            except (FileNotFoundError, Exception) as e:
                print(f"Error loading map image {self.map_filepath}: {e}. Falling back to procedural map.")
                self.map_filepath = None
        
        if not self.map_filepath:
            meters_h_gt = cfg.get("height", 1.0)
            meters_w_gt = cfg.get("width", 5.0)
            self.res_m = cfg.get("resolution", 0.01)
            
            H_gt = int(round(meters_h_gt / self.res_m))
            W_gt = int(round(meters_w_gt / self.res_m))
            gt_original = np.full((H_gt, W_gt), self.map_mask["free"], dtype=np.int8)
            # Procedural additions will be done on the padded map later

        # --- 2. Pad the gt map to create margin ---
        margin_m = 0.5 # Increased margin from 0.2 to 0.5
        margin_cells = int(round(margin_m / self.res_m))

        H_padded = gt_original.shape[0] + 2 * margin_cells
        W_padded = gt_original.shape[1] + 2 * margin_cells

        self.gt = np.full((H_padded, W_padded), self.map_mask["occupied"], dtype=np.int8)
        self.gt[margin_cells : margin_cells + gt_original.shape[0], 
                margin_cells : margin_cells + gt_original.shape[1]] = gt_original

        # --- 3. Update dimensions and origins ---
        self.H = H_padded
        self.W = W_padded
        self.meters_h = H_padded * self.res_m
        self.meters_w = W_padded * self.res_m
        
        self.belief_origin_x = -margin_m
        self.belief_origin_y = -margin_m

        # --- 4. Add procedural elements if needed ---
        if not self.map_filepath:
            self.add_border_walls()
            self.add_start_and_goal_zones()

        # --- 5. Create belief maps ---
        self.belief = np.full((self.H, self.W), self.map_mask["unknown"], dtype=np.int8)
        self.belief_frontier = np.full((self.H, self.W), self.map_mask["unknown"], dtype=np.int8)


    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        H, W = self.H, self.W
        x_shifted = x - self.belief_origin_x
        y_shifted = y - self.belief_origin_y
        col = int(np.clip(x_shifted / self.res_m, 0, W - 1))
        row_from_bottom = int(np.clip(y_shifted / self.res_m, 0, H - 1))
        row = (H - 1) - row_from_bottom
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        H, W = self.H, self.W
        y_from_bottom = (H - 1 - row) * self.res_m
        x_shifted = col * self.res_m
        y_shifted = y_from_bottom
        x = x_shifted + self.belief_origin_x
        y = y_shifted + self.belief_origin_y
        return x, y
    
    def world_to_grid_np(self, world: np.ndarray) -> np.ndarray:
        H, W = self.H, self.W
        if world.ndim == 1:
            world = world.reshape(1, -1)
        x_shifted = world[:, 0] - self.belief_origin_x
        y_shifted = world[:, 1] - self.belief_origin_y
        col = np.clip(x_shifted / self.res_m, 0, W - 1).astype(np.long)
        row = (H - 1) - (np.clip(y_shifted / self.res_m, 0, H - 1)).astype(np.long)

        grid_position = np.stack((col, row), axis=-1)

        return grid_position

    def grid_to_world_np(self, grid: np.ndarray) -> np.ndarray:
        H, W = self.H, self.W
        if grid.ndim == 1:
            grid = grid.reshape(1, -1)
        col = grid[:, 0]
        row = grid[:, 1]
        x_shifted = col * self.res_m
        y_shifted = (H - 1 - row) * self.res_m    
        x = x_shifted + self.belief_origin_x
        y = y_shifted + self.belief_origin_y

        world = np.stack((x, y), axis=-1)

        return world

    def reset_gt_and_belief(self):
        if not self.map_filepath:
            # Only reset and add procedural elements if not loading from file
            self.gt.fill(self.map_mask["free"])
            self.add_border_walls()
            self.add_start_and_goal_zones()
        # Belief map is always reset
        self.belief.fill(self.map_mask["unknown"])
        self.belief_frontier.fill(self.map_mask["unknown"])

    def place_start_goal(self, start_xy=(0.1, 0.5), goal_xy=(4.9, 0.5)):
        # This method should only be called for procedural maps or if start/goal are not in image
        if self.map_filepath:
            print("Warning: place_start_goal called on image-loaded map. This might overwrite image-defined features.")
        rs, cs = self.world_to_grid(*start_xy)
        rg, cg = self.world_to_grid(*goal_xy)
        self.gt[rs, cs] = self.map_mask["start"]
        self.gt[rg, cg] = self.map_mask["goal"]
        self.belief[rs, cs] = self.map_mask["start"]
        self.belief_frontier[rs, cs] = self.map_mask["start"]

    def add_rect_obstacle(self, xmin: float, ymin: float, xmax: float, ymax: float):
        if self.map_filepath:
            print("Warning: add_rect_obstacle called on image-loaded map. This might overwrite image-defined obstacles.")
        r1, c1 = self.world_to_grid(max(0.0, xmin), max(0.0, ymin))
        r2, c2 = self.world_to_grid(min(self.meters_w, xmax), min(self.meters_h, ymax))
        r_lo, r_hi = sorted((r1, r2))
        c_lo, c_hi = sorted((c1, c2))
        self.gt[r_lo:r_hi+1, c_lo:c_hi+1] = self.map_mask["occupied"]

    def add_random_rect_obstacles(self, n: int = 10, 
                                  min_w_m: float = 0.05, min_h_m: float = 0.05,
                                  max_w_m: float = 0.15, max_h_m: float = 0.15,
                                  min_x_m: float = 0.25,
                                  seed: Optional[int] = None):
        """Place N rectangular obstacles of random size with min/max dimensions."""
        if self.map_filepath:
            print("Warning: add_random_rect_obstacles called on image-loaded map. This might overwrite image-defined obstacles.")
            return # Do not add random obstacles if map is loaded from file
        rng = np.random.default_rng(seed)
        for _ in range(max(0, int(n))):
            w = rng.uniform(min_w_m, max_w_m)
            h = rng.uniform(min_h_m, max_h_m)
            x = rng.uniform(min_x_m, max(min_x_m, self.meters_w - w))
            y = rng.uniform(0.0, max(min_x_m, self.meters_h - h))
            self.add_rect_obstacle(x, y, x+w, y+h)

    def add_border_walls(self, thickness_m: float = 0.05):
        if self.map_filepath:
            print("Warning: add_border_walls called on image-loaded map. This might overwrite image-defined borders.")
        t = max(1, int(round(thickness_m / self.res_m)))
        self.gt[:t, :] = self.map_mask["occupied"]
        self.gt[-t:, :] = self.map_mask["occupied"]
        self.gt[:, :t] = self.map_mask["occupied"]
        self.gt[:, -t:] = self.map_mask["occupied"]

    def add_start_and_goal_zones(self, 
                                 wall_thickness_m: float = 0.05,
                                 thickness_m: float = 0.25):
        if self.map_filepath:
            print("Warning: add_start_and_goal_zones called on image-loaded map. This might overwrite image-defined zones.")
        H = self.H
        W = self.W
        wall_t = max(1, int(round(wall_thickness_m / self.res_m)))
        t = max(1, int(round(thickness_m / self.res_m)))
        self.gt[wall_t-1:H-wall_t, wall_t:wall_t+t+1] = self.map_mask["start"]
        self.gt[wall_t-1:H-wall_t, W-1-wall_t-t:W-wall_t] = self.map_mask["goal"]


class Env():
    def __init__(self, cfg: EnvCfg) ->None:
        self.cfg = cfg

        self.device = self.cfg.device
        self.seed = self.cfg.seed
        self.dt = self.cfg.physics_dt

        self.fov = self.cfg.fov
        self.sensor_range = self.cfg.sensor_range
        self.num_agent = self.cfg.num_agent
        self.max_lin_vel = self.cfg.max_velocity
        self.max_ang_vel = self.cfg.max_yaw_rate
        self.max_lin_acc = self.cfg.max_acceleration

        self.map_info = MapInfo(cfg=cfg.map)
        self.total_cells = self.map_info.H * self.map_info.W
        
        # Location은 2D, Velocity는 스칼라 커맨드
        self.robot_locations = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.robot_velocities = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.robot_global_velocities = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.robot_angles = np.zeros((self.num_agent, 1), dtype=np.float32)
        self.robot_yaw_rate = np.zeros((self.num_agent, 1), dtype=np.float32)

        self.num_step = 0
        self.reached_goal = np.zeros((self.cfg.num_agent, 1), dtype=np.bool_)

        self.infos = {}


    def reset(self, episode_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if episode_seed is not None:
            self.seed = episode_seed
        # Load ground truth map and initial cell
        self.reached_goal = np.zeros((self.cfg.num_agent, 1), dtype=np.bool_)
        self.num_step = 0
        self.map_info.reset_gt_and_belief()
        if not self.map_info.map_filepath: # Only add random obstacles if not loading from file
            self.map_info.add_random_rect_obstacles(seed=self.seed)

        # Randomly place N_AGENTS in start zone 
        world_x, world_y = self._set_init_state()
        self.robot_locations = np.stack([world_x, world_y], axis=1)

        # Initialize headings
        self.robot_angles = np.zeros(self.num_agent, dtype=np.float32)
        # Perform initial sensing update for each agent
        for i in range(self.num_agent):
            cell = self.map_info.world_to_grid_np(self.robot_locations[i])

            self.map_info.belief = sensor_work_heading(cell,
                                                       round(self.sensor_range / self.map_info.res_m),
                                                       self.map_info.belief,
                                                       self.map_info.gt,
                                                       np.rad2deg(self.robot_angles[i]),
                                                       360,
                                                       self.map_info.map_mask)
        
        self._compute_intermediate_values()
        self.infos = self._update_infos()
        self.obs_buf = self._get_observations()
        self.state_buf = self._get_states()

        return self.obs_buf, self.state_buf, self.infos

    def _set_init_state(self) -> Tuple[np.ndarray, np.ndarray]:
        map_info = self.map_info
        H = map_info.H
        start_cells = np.column_stack((np.nonzero(map_info.gt == map_info.map_mask["start"])[0], 
                                       np.nonzero(map_info.gt == map_info.map_mask["start"])[1]))
        idx = np.random.choice(len(start_cells), self.num_agent, replace=False)
        chosen = start_cells[idx]
        rows, cols = chosen[:, 0], chosen[:, 1]

        world_x, world_y = map_info.grid_to_world(rows, cols)

        return world_x, world_y

    def update_robot_belief(self, robot_cell, heading) -> None:
        self.map_info.belief = sensor_work_heading(robot_cell, 
                                                   round(self.sensor_range / self.map_info.res_m), 
                                                   self.map_info.belief,
                                                   self.map_info.gt, 
                                                   heading, 
                                                   self.fov, 
                                                   self.map_info.map_mask)


    def step(self, actions) -> Tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     dict[str, np.ndarray]]:
        """
            actions
                [n, 0] : linear acceleration command of n'th agent
                [n, 1] : angular velocity command of n'th agent

            Return :
                obs_buf -> [n, obs_dim]         : t+1 observation
                state_buf -> [n, state_dim]     : t+1 state
                action_buf -> [n, act_dim]      : t action
                reward_buf -> [n, 1]            : t+1 reward
                termination_buf -> [n, 1]       : t+1 terminated
                truncation_buf  -> [n, 1]       : t+1 truncated
                info -> dict[str, [n, dim]]     : additional metric 

        """
        
        # RL Action 전처리 단계
        self._pre_apply_action(actions)

        for i in range(self.cfg.decimation):
            for j in range(self.num_agent):
                # 이미 도달한 에이전트는 상태 업데이트 X
                if self.reached_goal[j]:
                    #continuestep
                    pass
                # ============== Step Numerical Simulation ================

                # action을 적용하여 robot state (위치 및 각도) 업데이트
                self._apply_action(j)

                # ========================================================

                # Belief 업데이트
                cell = self.map_info.world_to_grid_np(self.robot_locations[j])
                self.update_robot_belief(cell, np.rad2deg(self.robot_angles[j]))

        # Done 신호 생성
        self.num_step += 1
        self.termination_buf, self.truncation_buf, self.reached_goal = self._get_dones()

        # 추가정보 Infos 업데이트
        self.infos = self._update_infos()

        # 보상 계산
        self.reward_buf = self._get_rewards()
        
        # Next Observation 세팅
        self.obs_buf = self._get_observations()
        self.state_buf = self._get_states()


        return self.obs_buf, self.state_buf, self.reward_buf, self.termination_buf, self.truncation_buf, self.infos


    # =============== Env-Specific Abstract Methods =================
    
    @abstractmethod
    def _pre_apply_action(self, actions):
        raise NotImplementedError(f"Please implement the '_pre_apply_action' method for {self.__class__.__name__}.") 

    @abstractmethod
    def _apply_action(self, agent_id: int) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.") 
    

    @abstractmethod
    def _get_observations(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")
    

    @abstractmethod
    def _get_states(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_states' method for {self.__class__.__name__}.")
    

    @abstractmethod
    def _get_dones(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")


    @abstractmethod
    def _compute_intermediate_values(self) -> None:
        raise NotImplementedError(f"Please implement the '_compute_intermediate_values' method for {self.__class__.__name__}.")


    @abstractmethod
    def _get_rewards(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")
    
    @abstractmethod
    def _update_infos(self):
        raise NotImplementedError(f"Please implement the '_update_infos' method for {self.__class__.__name__}.")