import numpy as np

def get_nominal_control(p_target: list[np.ndarray] | np.ndarray,
                        follower: list[bool] | np.ndarray,
                        v_current: list[float] | np.ndarray,
                        a_max: float,
                        w_max: float,
                        v_max: float,
                        k_v: float = 1.0,
                        k_w: float = 1.5) -> np.ndarray:

        if isinstance(p_target, list):
            p_target = np.vstack(p_target) # (n, 2)
        if isinstance(v_current, list):
            v_current = np.array(v_current).reshape(-1, 1) # (n, 1)
        if isinstance(follower, list):
            follower = np.array(follower)
            k_v_arr = np.where(follower, k_v*1, k_v)
            k_w_arr = np.where(follower, k_w*1, k_w)

        lx, ly = p_target[:, 0], p_target[:, 1]
            
        dist_to_target = np.sqrt(lx**2 + ly**2)
        angle_to_target = np.arctan2(ly, lx)

        # Target velocity based on distance
        v_target = np.clip(k_v_arr * dist_to_target, 0.0, v_max).reshape(-1, 1)
        
        # P-control for acceleration
        a_ref = k_v * (v_target - v_current)
        a_ref = np.clip(a_ref, -a_max, a_max)

        # P-control for angular velocity
        w_ref = np.clip(k_w_arr * angle_to_target, -w_max, w_max).reshape(-1, 1)
        
        return np.hstack([a_ref, w_ref])