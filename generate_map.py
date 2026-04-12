import os
import sys
import yaml
import numpy as np
from PIL import Image
from collections import deque

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
BORDER_THICK      = 0.05      # 외곽 벽 두께
OB_MIN, OB_MAX    = 0.10, 0.30  # 장애물 한 변 범위
INFLATION_RADIUSM = 0.15     # inflated 경로용 "로봇 반경" (≈ 최소 통로 0.25 m 요구)

# ============== Utils ==============
def meters_to_cells(m, res):
    return int(np.round(m / res))

def effective_sq_size(width_m, height_m, border_thick, desired=1.0, eps=1e-6):
    """외곽 벽 안에 완전히 들어가도록 START/GOAL 정사각형의 한 변 길이 계산."""
    inner_w = width_m  - 2*border_thick
    inner_h = height_m - 2*border_thick
    return max(0.0, min(desired, inner_w - eps, inner_h - eps))

def rect_to_rc_bounds(x0, y0, x1, y1, res, H, W):
    """좌하단(x0,y0), 우상단(x1,y1) [m] → (row,col) inclusive 범위 (넘파이 y축 반전)."""
    x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0

    c0 = meters_to_cells(x0, res)
    c1 = meters_to_cells(x1, res) - 1
    r1 = (H - 1) - meters_to_cells(y0, res)            # lower y
    r0 = (H - 1) - meters_to_cells(y1, res) + 1        # upper y

    r0 = np.clip(r0, 0, H - 1); r1 = np.clip(r1, 0, H - 1)
    c0 = np.clip(c0, 0, W - 1); c1 = np.clip(c1, 0, W - 1)

    if r0 > r1: r0, r1 = r1, r0
    if c0 > c1: c0, c1 = c1, c0
    return int(r0), int(r1), int(c0), int(c1)

def fill_rect(gt, x0, y0, x1, y1, value, res):
    H, W = gt.shape
    r0, r1, c0, c1 = rect_to_rc_bounds(x0, y0, x1, y1, res, H, W)
    gt[r0:r1+1, c0:c1+1] = value

def save_map_as_image(map_array, path, filename):
    os.makedirs(path, exist_ok=True)
    h, w = map_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for v, col in COLORS.items():
        rgb[map_array == v] = col
    Image.fromarray(rgb).save(os.path.join(path, filename))

def cell_from_xy(x, y, res, H, W):
    col = meters_to_cells(x, res)
    row = (H - 1) - meters_to_cells(y, res)
    row = np.clip(row, 0, H - 1); col = np.clip(col, 0, W - 1)
    return int(row), int(col)

def add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK):
    """외곽에 thickness(미터) 벽 추가."""
    t = max(1, meters_to_cells(thickness, res))
    H, W = gt.shape
    gt[:t, :]    = OCCUPIED  # top
    gt[-t:, :]   = OCCUPIED  # bottom
    gt[:, :t]    = OCCUPIED  # left
    gt[:, -t:]   = OCCUPIED  # right

def carve_segment_rect(gt, x0, y0, x1, y1, width, res):
    """선분 (x0,y0)-(x1,y1)를 중심으로 폭 width짜리 직사각형 통로를 FREE로 carve."""
    if abs(x0 - x1) < 1e-6:  # vertical
        xc = x0; xL, xR = xc - width/2, xc + width/2
        yL, yU = sorted((y0, y1))
        fill_rect(gt, xL, yL, xR, yU, FREE, res)
    elif abs(y0 - y1) < 1e-6:  # horizontal
        yc = y0; yL, yU = yc - width/2, yc + width/2
        xL, xR = sorted((x0, x1))
        fill_rect(gt, xL, yL, xR, yU, FREE, res)
    else:
        # 대각선이면 단순 bounding box carve (여기서는 안 쓰도록 설계)
        xL, xR = sorted((x0, x1)); yL, yU = sorted((y0, y1))
        fill_rect(gt, xL, yL, xR, yU, FREE, res)

def sprinkle_obstacles_dense(
    gt, res, width_m, height_m, rng,
    n_min, n_max,
    w_min=0.06, w_max=0.15,
    h_min=0.06, h_max=0.15,
    forbid_rects=None,
    forbid_mask=None,
    max_trials=50000
):
    """
    FREE 영역에만 작은 직사각형 장애물을 뿌린다.
    - forbid_rects: START/GOAL 같은 사각형 금지 영역
    - forbid_mask: True인 셀에는 장애물 배치 금지 (예: 통로 밖/안 제어)
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

        x = float(rng.uniform(BORDER_THICK,
                              max(BORDER_THICK, width_m  - BORDER_THICK - w)))
        y = float(rng.uniform(BORDER_THICK,
                              max(BORDER_THICK, height_m - BORDER_THICK - h)))

        # START/GOAL 사각형과 겹치면 안 됨
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

        # forbid_mask가 True인 셀을 건드리면 안 됨
        if np.any(forbid_mask[r0:r1+1, c0:c1+1]):
            continue

        # 후보 영역이 전부 FREE인 곳에만 배치
        if not (gt[r0:r1+1, c0:c1+1] == FREE).all():
            continue

        gt[r0:r1+1, c0:c1+1] = OCCUPIED
        placed += 1

def dilate_mask_bool(mask, rad):
    """scipy 없이 간단 dilation: Manhattan 반경 rad로 확장."""
    if rad <= 0:
        return mask
    H, W = mask.shape
    out = mask.copy()
    for dy in range(-rad, rad+1):
        dx_max = rad - abs(dy)
        for dx in range(-dx_max, dx_max+1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(mask, dtype=bool)
            r_src0 = max(0, -dy); r_src1 = min(H, H - dy)
            c_src0 = max(0, -dx); c_src1 = min(W, W - dx)
            r_dst0 = r_src0 + dy; r_dst1 = r_src1 + dy
            c_dst0 = c_src0 + dx; c_dst1 = c_src1 + dx
            shifted[r_dst0:r_dst1, c_dst0:c_dst1] = mask[r_src0:r_src1, c_src0:c_src1]
            out |= shifted
    return out

def build_inflated_occupancy(gt, res, inflation_radius_m):
    """
    gt에서 OCCUPIED를 dilation 해서 inflated occupancy mask (boolean) 생성.
    START/GOAL/FREE/UNKNOWN은 free로 취급.
    """
    occ_mask = (gt == OCCUPIED)
    rad_cells = meters_to_cells(inflation_radius_m, res)
    inflated = dilate_mask_bool(occ_mask, rad_cells)
    return inflated  # boolean mask

def grid_path_exists(start_xy, goal_xy, inflated_occ, res):
    """
    inflated_occ (True=충돌) 위에서 4-connected BFS로
    start → goal 경로 존재 여부만 판단.
    """
    H, W = inflated_occ.shape
    sx, sy = start_xy
    gx, gy = goal_xy

    sr, sc = cell_from_xy(sx, sy, res, H, W)
    gr, gc = cell_from_xy(gx, gy, res, H, W)

    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return False

    free_mask = ~inflated_occ  # False=충돌, True=통과 가능
    if not (free_mask[sr, sc] and free_mask[gr, gc]):
        return False

    q = deque()
    q.append((sr, sc))
    seen = np.zeros((H, W), dtype=bool)
    seen[sr, sc] = True

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):  # 4-connected
            rr, cc = r + dr, c + dc
            if (0 <= rr < H and 0 <= cc < W and
                not seen[rr, cc] and free_mask[rr, cc]):
                seen[rr, cc] = True
                q.append((rr, cc))
    return False

# ============== I-shape (1×5 m): 랜덤 장애물 + BFS 검증 ==============
def create_i_map(width_m, height_m, res, seed,
                 inflation_radius_m=INFLATION_RADIUSM, max_attempts=8):
    """
    1×5 I-shape:
    - 내부는 FREE
    - START/GOAL 사각형을 좌/우에 배치
    - 전체에 직사각형 장애물을 랜덤하게 뿌린 뒤,
    - inflated occupancy 위에서 start→goal BFS 성공하면 그 맵 확정
    """
    rng_master = np.random.default_rng(seed)
    H = meters_to_cells(height_m, res)
    W = meters_to_cells(width_m, res)

    last_gt = None

    for _ in range(max_attempts):
        rng = np.random.default_rng(rng_master.integers(0, 10**9))

        # 배경 FREE + 외곽 벽
        gt = np.full((H, W), FREE, dtype=np.uint8)
        add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK)

        b  = BORDER_THICK
        sq = effective_sq_size(width_m, height_m, b, desired=START_GOAL_SIZE)

        start_rect = (b, b, b + sq, b + sq)
        goal_rect  = (width_m - b - sq, height_m - b - sq, width_m - b, height_m - b)

        # 장애물 랜덤 뿌리기
        sprinkle_obstacles_dense(
            gt, res, width_m, height_m, rng,
            n_min=5, n_max=10,
            w_min=OB_MIN, w_max=OB_MAX,
            h_min=OB_MIN, h_max=OB_MAX,
            forbid_rects=[start_rect, goal_rect],
            forbid_mask=None,
            max_trials=5000
        )

        # START/GOAL 라벨
        fill_rect(gt, *start_rect, START, res)
        fill_rect(gt, *goal_rect,  GOAL,  res)

        # inflated occupancy
        inflated_occ = build_inflated_occupancy(gt, res, inflation_radius_m)

        sx = (start_rect[0] + start_rect[2]) / 2.0
        sy = (start_rect[1] + start_rect[3]) / 2.0
        gx = (goal_rect[0] + goal_rect[2]) / 2.0
        gy = (goal_rect[1] + goal_rect[3]) / 2.0

        # BFS로 경로 존재 확인
        if grid_path_exists((sx, sy), (gx, gy), inflated_occ, res):
            return gt

        last_gt = gt

    # 여러 번 실패하면 마지막 것이라도 반환
    return last_gt

# ============== Square (5×5 m): start=왼쪽 하단, goal=오른쪽(상~하단) + 3 랜덤 점 + 1m 맨해튼 통로 + 통로 안 장애물 + BFS 검증 ==============
def create_square_map(width_m, height_m, res, seed,
                      inflation_radius_m=INFLATION_RADIUSM, max_attempts=8):
    """
    Square 5×5:
    - start: 왼쪽 하단 내부 (벽 안쪽)
    - goal : 오른쪽 벽 안쪽 (y는 상~하단 범위에서 랜덤)
    - 중앙 영역에서 랜덤 3점 (waypoints) 샘플
    - [start, w1, w2, w3, goal] 을 폭 1m 맨해튼 경로로 잇고 carve
    - 통로 안에만 작은 장애물 소량 배치
    - inflated occupancy 위에서 start→goal BFS 성공하면 확정
    """
    rng_master = np.random.default_rng(seed)
    H = meters_to_cells(height_m, res)
    W = meters_to_cells(width_m, res)

    corridor_width = 1.0
    half_cw = corridor_width / 2.0

    last_gt = None

    for _ in range(max_attempts):
        rng = np.random.default_rng(rng_master.integers(0, 10**9))

        gt = np.full((H, W), OCCUPIED, dtype=np.uint8)
        add_border_wall(gt, res, width_m, height_m, thickness=BORDER_THICK)

        # START/GOAL 사각형 크기 (너무 크지 않게)
        sg_size = min(
            START_GOAL_SIZE,
            0.8,
            width_m - 2 * (BORDER_THICK + 0.1),
            height_m - 2 * (BORDER_THICK + 0.1),
        )
        sg_size = max(0.3, sg_size)
        sg_half = sg_size / 2.0

        # --- start: 왼쪽 하단 ---
        sx = BORDER_THICK + sg_half
        sy = BORDER_THICK + sg_half

        # --- goal: 오른쪽 벽 안쪽, y는 상~하단에서 랜덤 ---
        y_low  = BORDER_THICK + sg_half
        y_high = height_m - BORDER_THICK - sg_half
        if y_high <= y_low:
            # 극단적인 설정 방지용 fallback (This is a guess.)
            y_low  = BORDER_THICK + 0.5
            y_high = height_m - BORDER_THICK - 0.5
        gx = width_m - BORDER_THICK - sg_half
        gy = float(rng.uniform(y_low, y_high))

        start_rect = (sx - sg_half, sy - sg_half,
                      sx + sg_half, sy + sg_half)
        goal_rect  = (gx - sg_half, gy - sg_half,
                      gx + sg_half, gy + sg_half)

        # 내부 3개 waypoint: start와 goal 사이 x 영역에서 샘플
        x_inner_min = sx + 0.5
        x_inner_max = gx - 0.5
        # y는 goal y와 같은 대략적인 세로 범위에서 랜덤
        inner_y_low  = y_low
        inner_y_high = y_high

        inner_nodes = []
        for _in in range(3):
            x = float(rng.uniform(x_inner_min, x_inner_max))
            y = float(rng.uniform(inner_y_low, inner_y_high))
            inner_nodes.append((x, y))

        # x 기준 정렬 (좌→우)
        inner_nodes.sort(key=lambda p: p[0])

        # 전체 노드 시퀀스: [start, w1, w2, w3, goal]
        nodes = [(sx, sy)] + inner_nodes + [(gx, gy)]

        # 폭 1m 맨해튼 경로 carve
        for (x0, y0), (x1, y1) in zip(nodes[:-1], nodes[1:]):
            # 상하좌우로만: 두 leg로 나눔
            if rng.random() < 0.5:
                mid = (x1, y0)  # 가로 먼저, 그다음 세로
            else:
                mid = (x0, y1)  # 세로 먼저, 그다음 가로
            mx, my = mid

            # 1차 leg
            if abs(x0 - mx) > 1e-6 or abs(y0 - my) > 1e-6:
                carve_segment_rect(gt, x0, y0, mx, my, corridor_width, res)
            # 2차 leg
            if abs(mx - x1) > 1e-6 or abs(my - y1) > 1e-6:
                carve_segment_rect(gt, mx, my, x1, y1, corridor_width, res)

        # START/GOAL 주변은 확실히 FREE로
        fill_rect(gt, *start_rect, FREE, res)
        fill_rect(gt, *goal_rect,  FREE, res)

        # 통로 mask
        corridor_mask = (gt == FREE)

        # 통로 안에만 작은 장애물 소량
        inner_wmax = min(0.4, 0.5 * corridor_width)
        inner_hmax = min(0.4, 0.5 * corridor_width)
        if inner_wmax > 0.05 and inner_hmax > 0.05:
            sprinkle_obstacles_dense(
                gt, res, width_m, height_m, rng,
                n_min=5, n_max=10,
                w_min=0.05, w_max=inner_wmax,
                h_min=0.05, h_max=inner_hmax,
                forbid_rects=[start_rect, goal_rect],
                forbid_mask=~corridor_mask,  # 통로 밖 금지 → 통로 안만 허용
                max_trials=5000
            )

        # START/GOAL 라벨링
        fill_rect(gt, *start_rect, START, res)
        fill_rect(gt, *goal_rect,  GOAL,  res)

        # inflated occupancy
        inflated_occ = build_inflated_occupancy(gt, res, inflation_radius_m)

        # BFS로 경로 존재 확인
        if grid_path_exists((sx, sy), (gx, gy), inflated_occ, res):
            return gt

        last_gt = gt

    return last_gt

# ============== Main ==============
if __name__ == "__main__":
    NUM_PER_TYPE = 50
    CONFIG_PATH = "config/cbf_test.yaml"
    I_DIR = "maps/i_shape"
    SQ_DIR = "maps/square"

    with open(CONFIG_PATH, "r") as f:
        env_map_cfg = yaml.safe_load(f)["env"]["map"]
    res = float(env_map_cfg.get("resolution", 0.05))

    # sizes
    i_height, i_width = 1.0, 5.0
    s_height, s_width = 5.0, 5.0

    print("--- Generating randomized I-shape maps (1m x 5m, random obstacles + BFS validation) ---")
    os.makedirs(I_DIR, exist_ok=True)
    for i in range(NUM_PER_TYPE):
        gt = create_i_map(i_width, i_height, res, seed=10_001 + i)
        save_map_as_image(gt, I_DIR, f"map_{i:03d}.png")
        if (i + 1) % 10 == 0:
            print(f"I-shape: {i+1}/{NUM_PER_TYPE}")

    print("\n--- Generating randomized Square maps (5m x 5m, bottom-left start / right-side goal + 3 waypoints + 1m corridor + obstacles + BFS validation) ---")
    os.makedirs(SQ_DIR, exist_ok=True)
    for i in range(NUM_PER_TYPE):
        gt = create_square_map(s_width, s_height, res, seed=20_001 + i)
        save_map_as_image(gt, SQ_DIR, f"map_{i:03d}.png")
        if (i + 1) % 10 == 0:
            print(f"Square: {i+1}/{NUM_PER_TYPE}")

    print("\nMap generation complete.")
