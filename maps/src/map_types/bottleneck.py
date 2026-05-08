import numpy as np
from ..utils import (
    FREE, OCCUPIED, START, GOAL,
    BORDER_THICK,
    meters_to_cells, fill_rect,
    add_border_wall, carve_segment_rect,
)


def create_bottleneck_1x5_map(width_m, height_m, res,
                               wide_width=0.8,
                               narrow_width=0.3,
                               bottleneck_x_range=(1.0, 2.0)):
    """Generate a 1x5 m corridor map with a bottleneck (narrow section).

    Args:
        width_m:  map width in meters (e.g. 5.0).
        height_m: map height in meters (e.g. 1.0).
        res:      cell size in meters/cell.
        wide_width:  corridor width outside the bottleneck.
        narrow_width: corridor width inside the bottleneck.
        bottleneck_x_range: (x_start, x_end) of the narrow section in meters.

    Returns:
        gt (np.ndarray uint8): label map.
    """
    H = meters_to_cells(height_m, res)
    W = meters_to_cells(width_m,  res)
    gt = np.full((H, W), OCCUPIED, dtype=np.uint8)
    add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK)

    # Carve full-width corridor along the centerline
    yc = height_m * 0.5
    carve_segment_rect(gt,
                       BORDER_THICK + 0.1, yc,
                       width_m - BORDER_THICK - 0.1, yc,
                       wide_width, res)

    # Re-fill the bottleneck section above/below the narrow band
    xB0, xB1 = bottleneck_x_range
    y_low  = yc - narrow_width / 2
    y_high = yc + narrow_width / 2

    wide_top    = yc + wide_width / 2
    wide_bottom = yc - wide_width / 2

    if wide_top > y_high + 1e-3:
        fill_rect(gt, xB0, y_high, xB1, wide_top,    OCCUPIED, res)
    if wide_bottom < y_low - 1e-3:
        fill_rect(gt, xB0, wide_bottom, xB1, y_low,  OCCUPIED, res)

    # START / GOAL: centered vertically, hugging left/right walls
    margin = 0.01
    sg_w = min(1.0, width_m  - 2 * (BORDER_THICK + margin))
    sg_h = min(1.0, height_m - 2 * (BORDER_THICK + margin))

    sx = BORDER_THICK + margin + sg_w / 2.0
    gx = width_m - (BORDER_THICK + margin + sg_w / 2.0)
    sy = gy = yc

    start_rect = (sx - sg_w/2, sy - sg_h/2, sx + sg_w/2, sy + sg_h/2)
    goal_rect  = (gx - sg_w/2, gy - sg_h/2, gx + sg_w/2, gy + sg_h/2)

    fill_rect(gt, *start_rect, START, res)
    fill_rect(gt, *goal_rect,  GOAL,  res)

    return gt
