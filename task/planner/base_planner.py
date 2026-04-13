from abc import ABC, abstractmethod
import numpy as np


class AbstractPlanner(ABC):
    @abstractmethod
    def plan(self, map_info, robot_locations, robot_velocities,
             num_agent, cfg) -> dict:
        """
        Centralized target planning for all agents.

        Returns:
            dict with keys:
                "assigned_rc" : np.ndarray (N, 2)  - assigned target cells in (col, row) order
                "root_id"     : int                - selected root agent ID
                "viz"         : dict with keys:
                    "targets_prob_heat" : np.ndarray | None  - heatmap for visualization
                    "assigned_rc_viz"   : np.ndarray | None  - copy of assigned_rc for viz
                    "cluster_infos"     : dict               - cluster debug info
        """
