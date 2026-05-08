import numpy as np

from scipy.optimize import linear_sum_assignment

def world_to_local(w1: np.ndarray = None, w2: np.ndarray = None, yaw: float = None) -> np.ndarray:
    """Transforms a point from world coordinates to the robot's local frame."""
    if w1 is None:
        delta = w2
    else:
        delta = w2 - w1
    rot_mat = np.array([[np.cos(-yaw), -np.sin(-yaw)], 
                        [np.sin(-yaw),  np.cos(-yaw)]])
    
    local_pos = np.matmul(rot_mat, delta.transpose())
    return local_pos.transpose()

def local_to_world(w1:np.ndarray, l1: np.ndarray, yaw: float) -> np.ndarray:
    """Transforms a point from local coordinates to the world frame
    
        wl : World Frame Reference Point
        l1 : Local Frame Target Point
    """
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], 
                        [np.sin(yaw),  np.cos(yaw)]])
    
    world_pos = w1 + np.matmul(rot_mat, l1.transpose()).transpose()

    return world_pos

def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line algorithm을 사용하여 (x0, y0)에서 (x1, y1)까지의 모든 셀 좌표를 반환"""
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    while True:
        yield (y, x)  # (row, col) 순서로 반환

        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

def pair_cost(maps, robot_pos, target_rc):
    ax, ay = robot_pos
    tr, tc = int(target_rc[0]), int(target_rc[1])
    tx, ty = maps.grid_to_world(tr, tc)

    return float(np.hypot(tx - ax, ty - ay))

def assign_targets_hungarian(maps, robot_pos, targets_rc, num_agent):
    """
    targets_rc: list[(gr,gc)] 
    return: assigned_rc[i] = (gr,gc)
    """
    N = num_agent
    M = len(targets_rc)
    C = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            C[i, j] = pair_cost(maps, robot_pos[i], targets_rc[j])

    row, col = linear_sum_assignment(C)  # row: 0..N-1
    assigned = [targets_rc[col[i]] for i in range(N)]

    return assigned