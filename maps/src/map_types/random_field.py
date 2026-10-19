import numpy as np
from ..utils import (
    FREE, OCCUPIED, START, GOAL,
    BORDER_THICK, INFLATION_RADIUSM,
    meters_to_cells, fill_rect,
    add_border_wall, carve_segment_rect, sprinkle_obstacles_dense,
    dilate_mask_bool, build_inflated_occupancy, grid_path_exists,
)


def create_random_field_5x5_map(width_m, height_m, res,
                                 seed,
                                 inflation_radius_m=INFLATION_RADIUSM,
                                 max_attempts=20,
                                 n_min=20, n_max=20):
    """Generate a 5x5 m open field with random obstacles and a guaranteed corridor.

    Layout:
        - START: bottom-left 1x1 m zone.
        - GOAL:  top-right  1x1 m zone.
        - A fixed winding corridor (0.6 m wide) connects START to GOAL.
        - Large random obstacles fill the rest of the FREE area.
        - BFS on inflated occupancy validates connectivity.

    Args:
        width_m:  map width in meters (e.g. 5.0).
        height_m: map height in meters (e.g. 5.0).
        res:      cell size in meters/cell.
        seed:     random seed.
        inflation_radius_m: robot radius for BFS validation.
        max_attempts: attempts before returning last result.
        n_min, n_max: obstacle count range.

    Returns:
        gt (np.ndarray uint8): label map.
    """
    rng_master = np.random.default_rng(seed)
    H = meters_to_cells(height_m, res)
    W = meters_to_cells(width_m,  res)

    last_gt = None

    for _ in range(max_attempts):
        rng = np.random.default_rng(rng_master.integers(0, 10**9))

        gt = np.full((H, W), OCCUPIED, dtype=np.uint8)
        add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK)

        # START: bottom-left 1x1 m, GOAL: top-right 1x1 m
        sg_size = 1.0

        sx0, sy0, sx1, sy1 = BORDER_THICK, BORDER_THICK, BORDER_THICK + sg_size, BORDER_THICK + sg_size
        gx1, gy1 = width_m - BORDER_THICK, height_m - BORDER_THICK
        gx0, gy0 = gx1 - sg_size, gy1 - sg_size

        start_rect = (sx0, sy0, sx1, sy1)
        goal_rect  = (gx0, gy0, gx1, gy1)

        sx_c = (sx0 + sx1) / 2.0
        sy_c = (sy0 + sy1) / 2.0
        gx_c = (gx0 + gx1) / 2.0
        gy_c = (gy0 + gy1) / 2.0

        # Fixed winding corridor (0.6 m wide)
        corridor_width = 0.6
        nodes = [
            (sx_c, sy_c),
            (sx_c, sy_c + 1.0),
            (1.5,  sy_c + 1.0),
            (1.5,  3.0),
            (3.0,  3.0),
            (3.0,  gy_c),
            (gx_c, gy_c),
        ]
        prev_x, prev_y = nodes[0]
        for (x1, y1) in nodes[1:]:
            carve_segment_rect(gt, prev_x, prev_y, x1, y1, corridor_width, res)
            prev_x, prev_y = x1, y1

        fill_rect(gt, *start_rect, FREE, res)
        fill_rect(gt, *goal_rect,  FREE, res)

        corridor_mask = (gt == FREE)

        # Dilate corridor mask to create a clearance zone around it
        path_clear_rad = meters_to_cells(0.2, res)
        corridor_forbid = dilate_mask_bool(corridor_mask, path_clear_rad)

        # Open up the interior for obstacle placement
        t = max(1, meters_to_cells(BORDER_THICK, res))
        gt[t:H-t, t:W-t] = FREE

        sprinkle_obstacles_dense(
            gt, res, width_m, height_m, rng,
            n_min=n_min, n_max=n_max,
            w_min=0.3, w_max=0.8,
            h_min=0.3, h_max=0.8,
            forbid_rects=[start_rect, goal_rect],
            max_trials=50000,
        )

        fill_rect(gt, *start_rect, START, res)
        fill_rect(gt, *goal_rect,  GOAL,  res)

        inflated_occ = build_inflated_occupancy(gt, res, inflation_radius_m)

        if grid_path_exists((sx_c, sy_c), (gx_c, gy_c), inflated_occ, res):
            return gt

        last_gt = gt

    return last_gt
