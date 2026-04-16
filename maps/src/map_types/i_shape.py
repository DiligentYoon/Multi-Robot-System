import numpy as np
from ..utils import (
    FREE, OCCUPIED, START, GOAL,
    BORDER_THICK, OB_MIN, OB_MAX, INFLATION_RADIUSM, START_GOAL_SIZE,
    meters_to_cells, effective_sq_size, fill_rect,
    add_border_wall, sprinkle_obstacles_dense,
    build_inflated_occupancy, grid_path_exists,
)


def create_i_map(width_m, height_m, res, seed,
                 inflation_radius_m=INFLATION_RADIUSM, max_attempts=8):
    """Generate a 1x5 m I-shaped map with random rectangular obstacles.

    Args:
        width_m:  map width in meters (e.g. 5.0).
        height_m: map height in meters (e.g. 1.0).
        res:      cell size in meters/cell.
        seed:     random seed for reproducibility.
        inflation_radius_m: robot radius used for BFS path validation.
        max_attempts: number of random attempts before returning last result.

    Returns:
        gt (np.ndarray uint8): label map with FREE/OCCUPIED/START/GOAL cells.
    """
    rng_master = np.random.default_rng(seed)
    H = meters_to_cells(height_m, res)
    W = meters_to_cells(width_m, res)

    last_gt = None

    for _ in range(max_attempts):
        rng = np.random.default_rng(rng_master.integers(0, 10**9))

        gt = np.full((H, W), FREE, dtype=np.uint8)
        add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK)

        b  = BORDER_THICK
        sq = effective_sq_size(width_m, height_m, b, desired=START_GOAL_SIZE)

        start_rect = (b, b, b + sq, b + sq)
        goal_rect  = (width_m - b - sq, height_m - b - sq, width_m - b, height_m - b)

        sprinkle_obstacles_dense(
            gt, res, width_m, height_m, rng,
            n_min=5, n_max=10,
            w_min=OB_MIN, w_max=OB_MAX,
            h_min=OB_MIN, h_max=OB_MAX,
            forbid_rects=[start_rect, goal_rect],
            forbid_mask=None,
            max_trials=5000,
        )

        fill_rect(gt, *start_rect, START, res)
        fill_rect(gt, *goal_rect,  GOAL,  res)

        inflated_occ = build_inflated_occupancy(gt, res, inflation_radius_m)

        sx = (start_rect[0] + start_rect[2]) / 2.0
        sy = (start_rect[1] + start_rect[3]) / 2.0
        gx = (goal_rect[0]  + goal_rect[2])  / 2.0
        gy = (goal_rect[1]  + goal_rect[3])  / 2.0

        if grid_path_exists((sx, sy), (gx, gy), inflated_occ, res):
            return gt

        last_gt = gt

    return last_gt
