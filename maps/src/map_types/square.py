import numpy as np
from ..utils import (
    FREE, OCCUPIED, START, GOAL,
    BORDER_THICK, INFLATION_RADIUSM, START_GOAL_SIZE,
    meters_to_cells, fill_rect,
    add_border_wall, carve_segment_rect, sprinkle_obstacles_dense,
    build_inflated_occupancy, grid_path_exists,
)


def create_square_map(width_m, height_m, res, seed,
                      inflation_radius_m=INFLATION_RADIUSM, max_attempts=8):
    """Generate a 5x5 m square map with a Manhattan corridor and random obstacles.

    Layout:
        - Background: fully OCCUPIED.
        - START: bottom-left corner (fixed).
        - GOAL: right wall, random y position.
        - Three random inner waypoints are connected by a 1 m wide Manhattan corridor.
        - Small obstacles are placed only inside the corridor.

    Args:
        width_m:  map width in meters (e.g. 5.0).
        height_m: map height in meters (e.g. 5.0).
        res:      cell size in meters/cell.
        seed:     random seed.
        inflation_radius_m: robot radius for BFS validation.
        max_attempts: random attempts before returning last result.

    Returns:
        gt (np.ndarray uint8): label map.
    """
    rng_master = np.random.default_rng(seed)
    H = meters_to_cells(height_m, res)
    W = meters_to_cells(width_m, res)

    corridor_width = 1.0

    last_gt = None

    for _ in range(max_attempts):
        rng = np.random.default_rng(rng_master.integers(0, 10**9))

        gt = np.full((H, W), OCCUPIED, dtype=np.uint8)
        add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK)

        sg_size = min(START_GOAL_SIZE, 0.8,
                      width_m  - 2 * (BORDER_THICK + 0.1),
                      height_m - 2 * (BORDER_THICK + 0.1))
        sg_size = max(0.3, sg_size)
        sg_half = sg_size / 2.0

        # START: bottom-left
        sx = BORDER_THICK + sg_half
        sy = BORDER_THICK + sg_half

        # GOAL: right wall, random y
        y_low  = BORDER_THICK + sg_half
        y_high = height_m - BORDER_THICK - sg_half
        if y_high <= y_low:
            y_low  = BORDER_THICK + 0.5
            y_high = height_m - BORDER_THICK - 0.5
        gx = width_m - BORDER_THICK - sg_half
        gy = float(rng.uniform(y_low, y_high))

        start_rect = (sx - sg_half, sy - sg_half, sx + sg_half, sy + sg_half)
        goal_rect  = (gx - sg_half, gy - sg_half, gx + sg_half, gy + sg_half)

        # Three inner waypoints sorted by x
        x_inner_min = sx + 0.5
        x_inner_max = gx - 0.5
        inner_nodes = [
            (float(rng.uniform(x_inner_min, x_inner_max)),
             float(rng.uniform(y_low, y_high)))
            for _ in range(3)
        ]
        inner_nodes.sort(key=lambda p: p[0])

        nodes = [(sx, sy)] + inner_nodes + [(gx, gy)]

        # Carve 1 m wide Manhattan corridor between consecutive nodes
        for (x0, y0), (x1, y1) in zip(nodes[:-1], nodes[1:]):
            if rng.random() < 0.5:
                mid = (x1, y0)  # horizontal first
            else:
                mid = (x0, y1)  # vertical first
            mx, my = mid

            if abs(x0 - mx) > 1e-6 or abs(y0 - my) > 1e-6:
                carve_segment_rect(gt, x0, y0, mx, my, corridor_width, res)
            if abs(mx - x1) > 1e-6 or abs(my - y1) > 1e-6:
                carve_segment_rect(gt, mx, my, x1, y1, corridor_width, res)

        fill_rect(gt, *start_rect, FREE, res)
        fill_rect(gt, *goal_rect,  FREE, res)

        corridor_mask = (gt == FREE)

        inner_wmax = min(0.4, 0.5 * corridor_width)
        inner_hmax = min(0.4, 0.5 * corridor_width)
        if inner_wmax > 0.05 and inner_hmax > 0.05:
            sprinkle_obstacles_dense(
                gt, res, width_m, height_m, rng,
                n_min=5, n_max=10,
                w_min=0.05, w_max=inner_wmax,
                h_min=0.05, h_max=inner_hmax,
                forbid_rects=[start_rect, goal_rect],
                forbid_mask=~corridor_mask,
                max_trials=5000,
            )

        fill_rect(gt, *start_rect, START, res)
        fill_rect(gt, *goal_rect,  GOAL,  res)

        inflated_occ = build_inflated_occupancy(gt, res, inflation_radius_m)

        if grid_path_exists((sx, sy), (gx, gy), inflated_occ, res):
            return gt

        last_gt = gt

    return last_gt
