
from ..base.env.env_cfg import EnvCfg

class CBFEnvCfg(EnvCfg):
    decimation: int
    max_episode_steps: int
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Space Information
        self.num_act = 2

        # Episode Information
        self.decimation = 1
        self.max_episode_steps = 1000000
        self.centralized_decimation = 30

        # Controller Cfg
        self.neighbor_sensing_distance = 0.5
        self.d_max = 0.5 # Connectivity distance
        self.d_safe = 0.05 # Inter-agent safety distance
        self.d_obs = 0.05 # Obstacle safety radius
        self.max_obs = self.num_rays
        self.max_agents = self.num_agent
        
        # Graph Info
        self.valid_threshold = 0.15


        self.assign_mode = "target_unknwown" # or target_unknwown or target_known
        if not hasattr(self, "graph_mode"):
            self.graph_mode = "mst"  # "mst" | "nn_tree"
