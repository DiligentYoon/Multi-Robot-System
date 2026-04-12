
from ..base.env.env_cfg import EnvCfg

class CBFEnvCfg(EnvCfg):
    num_obs: int
    num_state: int
    decimation: int
    max_episode_steps: int
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Space Information
        self.num_virtual_rays = 37
        self.num_obs = 10
        self.num_state = 10
        self.num_act = 2

        # Episode Information
        self.decimation = 1
        self.max_episode_steps = 1000000
        self.centralized_decimation = 30

        # Controller Cfg
        self.neighbor_sensing_distance = 0.75
        self.d_max = 0.75 # Connectivity distance
        self.d_safe = 0.05 #Inter-agent safety distance
        self.d_obs = 0.05 # Obstacle safety radius
        self.max_obs = self.num_rays
        self.max_agents = self.num_agent
        
        # Graph Info
        self.valid_threshold = 0.15


        self.assign_mode = "target_unknwown" # or target_unknwown or target_known

        # Reward Info
        self.reward_weights = {
            "exploration": 10.0,
            "obs_penalty": 1.0,
            "coll_penalty": 1.0,
            "conn_penalty": 1.0,
            "action_penalty": 1.0,
            "collision": -10.0,
            "goal": 10.0
        }


