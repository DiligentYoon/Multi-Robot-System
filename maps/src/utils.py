import os
import numpy as np
from PIL import Image
from collections import deque
from scipy.ndimage import binary_dilation

# ============== GT labels & colors ==============
FREE     = 0
UNKNOWN  = 1
OCCUPIED = 2
GOAL     = 3
START    = 4

COLORS = {
    FREE:     [255, 255, 255],  # white
    UNKNOWN:  [230, 230, 230],  # light gray
    OCCUPIED: [0,   0,   0],    # black
    GOAL:     [180, 50,  200],  # vivid purple
    START:    [50,  200, 80],   # vivid green
}

# Sizes (meters)
START_GOAL_SIZE   = 1.0
BORDER_THICK      = 0.05        # outer wall thickness
OB_MIN, OB_MAX    = 0.10, 0.30  # obstacle side range
INFLATION_RADIUSM = 0.15        # robot radius for inflated path check


# ============== Coordinate utilities ==============

def meters_to_cells(m, res):
    return int(np.round(m / res))


def effective_sq_size(width_m, height_m, border_thick, desired=1.0, eps=1e-6):
    """Compute START/GOAL square side length that fits inside the border walls."""
    inner_w = width_m  - 2 * border_thick
    inner_h = height_m - 2 * border_thick
    return max(0.0, min(desired, inner_w - eps, inner_h - eps))


def rect_to_rc_bounds(x0, y0, x1, y1, res, H, W):
    """Convert (bottom-left, top-right) [m] to inclusive (row, col) bounds.

    World frame: origin at bottom-left, x right, y up.
    Image frame: row 0 at top, so y-axis is flipped.
    """
    x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0

    c0 = meters_to_cells(x0, res)
    c1 = meters_to_cells(x1, res) - 1
    r1 = (H - 1) - meters_to_cells(y0, res)   # lower y → higher row
    r0 = (H - 1) - meters_to_cells(y1, res) + 1  # upper y → lower row

    r0 = int(np.clip(r0, 0, H - 1))
    r1 = int(np.clip(r1, 0, H - 1))
    c0 = int(np.clip(c0, 0, W - 1))
    c1 = int(np.clip(c1, 0, W - 1))

    if r0 > r1: r0, r1 = r1, r0
    if c0 > c1: c0, c1 = c1, c0
    return r0, r1, c0, c1


def cell_from_xy(x, y, res, H, W):
    """Convert world (x, y) [m] to (row, col) with y-axis flip."""
    col = meters_to_cells(x, res)
    row = (H - 1) - meters_to_cells(y, res)
    row = int(np.clip(row, 0, H - 1))
    col = int(np.clip(col, 0, W - 1))
    return row, col


# ============== Map drawing utilities ==============

def fill_rect(gt, x0, y0, x1, y1, value, res):
    """Fill rectangle defined by (bottom-left x0,y0) and (top-right x1,y1) with value."""
    H, W = gt.shape
    r0, r1, c0, c1 = rect_to_rc_bounds(x0, y0, x1, y1, res, H, W)
    gt[r0:r1+1, c0:c1+1] = value


def fill_rect_tlbr(gt, x_left, y_top, x_right, y_bottom, value, res):
    """Fill rectangle defined by top-left (x_left, y_top) and bottom-right (x_right, y_bottom).

    World frame: (0,0) is bottom-left, x right, y up.
    """
    x0 = float(min(x_left,  x_right))
    x1 = float(max(x_left,  x_right))
    y0 = float(min(y_bottom, y_top))
    y1 = float(max(y_bottom, y_top))
    fill_rect(gt, x0, y0, x1, y1, value, res)


def add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK):
    """Fill the outer border of the map with OCCUPIED cells."""
    t = max(1, meters_to_cells(thickness, res))
    H, W = gt.shape
    gt[:t, :]   = OCCUPIED  # top
    gt[-t:, :]  = OCCUPIED  # bottom
    gt[:, :t]   = OCCUPIED  # left
    gt[:, -t:]  = OCCUPIED  # right


def carve_segment_rect(gt, x0, y0, x1, y1, width, res):
    """Carve a rectangular FREE corridor of given width centered on segment (x0,y0)-(x1,y1).

    Only axis-aligned segments (horizontal or vertical) are handled precisely.
    Diagonal segments fall back to bounding-box carve.
    """
    if abs(x0 - x1) < 1e-6:        # vertical
        xc = x0
        xL, xR = xc - width / 2, xc + width / 2
        yL, yU = sorted((y0, y1))
        fill_rect(gt, xL, yL, xR, yU, FREE, res)
    elif abs(y0 - y1) < 1e-6:      # horizontal
        yc = y0
        yL, yU = yc - width / 2, yc + width / 2
        xL, xR = sorted((x0, x1))
        fill_rect(gt, xL, yL, xR, yU, FREE, res)
    else:                           # diagonal — bounding box fallback
        xL, xR = sorted((x0, x1))
        yL, yU = sorted((y0, y1))
        fill_rect(gt, xL, yL, xR, yU, FREE, res)


def save_map_as_image(map_array, path, filename):
    """Save a label map as an RGB PNG image."""
    os.makedirs(path, exist_ok=True)
    h, w = map_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for v, col in COLORS.items():
        rgb[map_array == v] = col
    Image.fromarray(rgb).save(os.path.join(path, filename))


# ============== Obstacle sprinkling ==============

def sprinkle_obstacles_dense(
    gt, res, width_m, height_m, rng,
    n_min, n_max,
    w_min=0.06, w_max=0.15,
    h_min=0.06, h_max=0.15,
    forbid_rects=None,
    forbid_mask=None,
    max_trials=50000,
):
    """Place random rectangular obstacles only in FREE cells.

    Args:
        forbid_rects: list of (x0, y0, x1, y1) no-placement zones (e.g. START/GOAL).
        forbid_mask:  boolean array; True cells are forbidden for placement.
    """
    H, W = gt.shape
    forbid_rects = [] if forbid_rects is None else list(forbid_rects)
    if forbid_mask is None:
        forbid_mask = np.zeros((H, W), dtype=bool)

    n_target = int(rng.integers(n_min, n_max + 1))
    placed, trials = 0, 0

    while placed < n_target and trials < max_trials:
        trials += 1

        w = float(rng.uniform(w_min, w_max))
        h = float(rng.uniform(h_min, h_max))
        if w <= 0 or h <= 0:
            continue

        x = float(rng.uniform(BORDER_THICK, max(BORDER_THICK, width_m  - BORDER_THICK - w)))
        y = float(rng.uniform(BORDER_THICK, max(BORDER_THICK, height_m - BORDER_THICK - h)))

        bad = False
        for (fx0, fy0, fx1, fy1) in forbid_rects:
            if not (x + w <= fx0 or fx1 <= x or y + h <= fy0 or fy1 <= y):
                bad = True
                break
        if bad:
            continue

        r0, r1, c0, c1 = rect_to_rc_bounds(x, y, x + w, y + h, res, H, W)
        if r1 < r0 or c1 < c0:
            continue

        if np.any(forbid_mask[r0:r1+1, c0:c1+1]):
            continue

        if not (gt[r0:r1+1, c0:c1+1] == FREE).all():
            continue

        gt[r0:r1+1, c0:c1+1] = OCCUPIED
        placed += 1


# ============== Occupancy inflation ==============

def dilate_mask_bool(mask, rad):
    """Dilate a boolean mask by Manhattan radius rad using scipy binary_dilation."""
    if rad <= 0:
        return mask.copy()
    # Build a diamond-shaped (L1-ball) structuring element
    size = 2 * rad + 1
    struct = np.zeros((size, size), dtype=bool)
    for dy in range(-rad, rad + 1):
        dx_max = rad - abs(dy)
        struct[dy + rad, rad - dx_max : rad + dx_max + 1] = True
    return binary_dilation(mask, structure=struct)


def build_inflated_occupancy(gt, res, inflation_radius_m):
    """Dilate OCCUPIED cells by the robot radius to create an inflated occupancy mask.

    Returns:
        Boolean array — True means the cell is occupied or within inflation radius.
    """
    occ_mask = (gt == OCCUPIED)
    rad_cells = meters_to_cells(inflation_radius_m, res)
    return dilate_mask_bool(occ_mask, rad_cells)


# ============== Path validation ==============

def grid_path_exists(start_xy, goal_xy, inflated_occ, res):
    """Check whether a collision-free path exists via 4-connected BFS.

    Args:
        start_xy: (x, y) [m] of start position.
        goal_xy:  (x, y) [m] of goal position.
        inflated_occ: boolean mask — True means blocked.

    Returns:
        True if a path exists, False otherwise.
    """
    H, W = inflated_occ.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    sr, sc = cell_from_xy(sx, sy, res, H, W)
    gr, gc = cell_from_xy(gx, gy, res, H, W)

    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return False

    free_mask = ~inflated_occ
    if not (free_mask[sr, sc] and free_mask[gr, gc]):
        return False

    q = deque([(sr, sc)])
    seen = np.zeros((H, W), dtype=bool)
    seen[sr, sc] = True

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not seen[rr, cc] and free_mask[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return False
