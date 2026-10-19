import numpy as np
from ..utils import (
    FREE, OCCUPIED, START, GOAL,
    BORDER_THICK,
    meters_to_cells, fill_rect,
    add_border_wall, carve_segment_rect,
)


def create_maze_5x5_map(width_m, height_m, res, corridor_width=0.5):
    """Generate a 5x5 m maze map with a hardcoded main path and branch corridors.

    Layout:
        - Background: fully OCCUPIED.
        - A main Manhattan path connects bottom-left to top-right.
        - Several dead-end branches add maze complexity.
        - START (0.8x0.8 m) at bottom-left, GOAL (0.8x0.8 m) at top-right.

    Args:
        width_m:  map width in meters (e.g. 5.0).
        height_m: map height in meters (e.g. 5.0).
        res:      cell size in meters/cell.
        corridor_width: width of all carved corridors in meters.

    Returns:
        gt (np.ndarray uint8): label map.
    """
    H = meters_to_cells(height_m, res)
    W = meters_to_cells(width_m,  res)
    gt = np.full((H, W), OCCUPIED, dtype=np.uint8)
    add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK)

    # Main path
    nodes_main = [
        (0.5, 0.5),
        (0.5, 1.5),
        (1.5, 1.5),
        (1.5, 3.5),
        (0.8, 3.5),
        (0.8, 4.5),
        (2.2, 4.5),
        (2.2, 2.5),
        (3.4, 2.5),
        (3.4, 0.8),
        (4.5, 0.8),
        (4.5, 4.5),
    ]
    for (x0, y0), (x1, y1) in zip(nodes_main[:-1], nodes_main[1:]):
        carve_segment_rect(gt, x0, y0, x1, y1, corridor_width, res)

    # Branch / dead-end corridors
    side_paths = [
        [(1.0, 0.5), (2.0, 0.5)],
        [(1.5, 1.5), (1.5, 0.5)],
        [(2.2, 2.5), (2.2, 3.5)],
        [(2.2, 3.5), (1.2, 3.5)],
        [(2.2, 4.5), (3.0, 4.5)],
        [(3.0, 4.5), (3.0, 3.8)],
        [(3.4, 2.5), (4.5, 2.5)],
        [(4.0, 2.5), (4.0, 1.5)],
    ]
    for path in side_paths:
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            carve_segment_rect(gt, x0, y0, x1, y1, corridor_width, res)

    # START / GOAL zones (0.8 x 0.8 m)
    sx, sy = 0.5, 0.5
    gx, gy = 4.5, 4.5
    sg_w, sg_h = 0.8, 0.8

    start_rect = (sx - sg_w/2, sy - sg_h/2, sx + sg_w/2, sy + sg_h/2)
    goal_rect  = (gx - sg_w/2, gy - sg_h/2, gx + sg_w/2, gy + sg_h/2)

    fill_rect(gt, *start_rect, START, res)
    fill_rect(gt, *goal_rect,  GOAL,  res)

    return gt
