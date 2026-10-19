import numpy as np
import math
import copy

def normalize_angle(angle):
    """Normalize an angle to be within [0, 360) degrees."""
    return angle % 360

def collision_check(x0, y0, x1, y1, ground_truth, robot_belief, map_mask):
    """
    Ray-cast from (x0,y0) to (x1,y1) in cell coordinates.
    Update robot_belief cell-by-cell:
    """
    # 1) 정수 셀 인덱스로 변환
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))

    # 2) Bresenham 준비
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    # 3) 레이 캐스팅 루프
    while True:
        # 3.1) 맵 범위 체크
        if not (0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]):
            break

        # 3.2) 셀 클래스 읽기
        gt = ground_truth[y, x]

        if gt == map_mask["occupied"]:
            # 충돌 지점만 OCCUPIED로 업데이트하고 종료
            robot_belief[y, x] = map_mask["occupied"]
            break
        elif gt == map_mask["goal"]:
            robot_belief[y, x] = map_mask["goal"]
        else:
            # FREE 또는 기타(UNKNOWN) 영역은 FREE로 업데이트
            robot_belief[y, x] = map_mask["free"]

        # 3.3) 종료 조건: 끝점 도달
        if x == x1 and y == y1:
            break

        # 3.4) Bresenham step
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 <  dx:
            err += dx
            y   += sy

    return robot_belief

def calculate_fov_boundaries(center_angle, fov):
    """Calculate the start and end angles of the field of vision (FOV).
    
    Args:
        center_angle (float): The central angle of the FOV in degrees.
        fov (float): The total field of vision in degrees.
        
    Returns:
        (float, float): The start and end angles of the FOV.
    """
    half_fov = fov / 2
    
    start_angle = center_angle - half_fov
    end_angle = center_angle + half_fov
    
    start_angle = normalize_angle(start_angle)
    end_angle = normalize_angle(end_angle)
    
    return start_angle, end_angle

def fov_sweep(start_angle, end_angle, increment):
    """Generate the correct sequence of angles to sweep the FOV from start to end with a specified increment.
    
    Args:
        start_angle (float): The starting angle of the FOV in degrees.
        end_angle (float): The ending angle of the FOV in degrees.
        increment (float): The angle increment in degrees.
        
    Returns:
        list: The sequence of angles representing the FOV sweep.
    """
    angles = []
    
    if start_angle < end_angle:
        angles = list(np.arange(start_angle, end_angle + increment, increment))
    else:
        angles = list(np.arange(start_angle, 360, increment)) + list(np.arange(0, end_angle + increment, increment))
    
    angles = [angle % 360 for angle in angles]
    
    angles_in_radians = np.radians(angles)

    return angles_in_radians

def sensor_work_heading(robot_position, 
                        sensor_range, 
                        robot_belief, 
                        ground_truth, 
                        heading, 
                        fov,
                        map_mask):

    sensor_angle_inc = 2.0
    if robot_position.shape[0] == 1:
        robot_position = robot_position.reshape(-1)
    x0 = robot_position[0]
    y0 = robot_position[1]
    start_angle, end_angle = calculate_fov_boundaries(heading, fov)
    sweep_angles = fov_sweep(start_angle, end_angle, sensor_angle_inc)

    x1_values = []
    y1_values = []
    
    for angle in sweep_angles:
        x1 = x0 + np.cos(angle) * sensor_range    
        y1 = y0 + np.sin(-angle) * sensor_range
        x1_values.append(x1)
        y1_values.append(y1)    
        
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief, map_mask)

    return robot_belief

def sense_and_update(map_info,
                     fov: int,
                     num_rays: int,
                     sensor_range: int,
                     agent_id: int, 
                     robot_locations: np.ndarray, 
                     robot_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Raycast in FOV, update belief FREE/OCCUPIED 
            Inputs:
                - agent_id: Agent Numbering
            Return:
                - frontier_local:  [(lx, ly), ...]  
                - frontier_rc:     [(row, col), ...] 
                - obs_local:       [(lx, ly), ...] 
    """
    drone_pose = np.hstack((robot_locations[agent_id], robot_angles[agent_id]))
    maps = map_info
    H, W = maps.H, maps.W
    half = math.radians(fov / 2.0)
    angles = np.linspace(-half, half, num_rays)

    # ======== 추가: goal 도 unknown-like 이웃으로 취급 ========
    mm = maps.map_mask
    UNKNOWN_LABEL = mm["unknown"]
    GOAL_LABEL    = mm["goal"]
    # belief 기준 unknown 이거나, GT 기준 goal 이면 "unknown 비슷"
    unknown_like = (maps.belief == UNKNOWN_LABEL) | (maps.gt == GOAL_LABEL)
    # ====================================================

    frontier_local: list[tuple[float, float]] = []
    frontier_rc: list[tuple[int, int]] = []
    obs_local: list[tuple[float, float]] = []

    for a in angles:
        ang = drone_pose[2] + a
        step = maps.res_m
        L = int(sensor_range / step)

        last_rc = None
        hit_recorded = False          # per-ray: obs 최대 1개
        frontier_candidate_rc = None  # per-ray: frontier 후보(마지막 FREE∧UNKNOWN-like-인접)

        for i in range(1, L + 1):
            x = drone_pose[0] + i * step * math.cos(ang)
            y = drone_pose[1] + i * step * math.sin(ang)
            if x < 0 or y < 0 or x > maps.meters_w or y > maps.meters_h:
                break

            r, c = maps.world_to_grid(x, y)
            if last_rc == (r, c):
                continue
            last_rc = (r, c)

            if maps.gt[r, c] == maps.map_mask["occupied"]:
                # 첫 OCC 히트만 기록
                maps.belief[r, c] = maps.map_mask["occupied"]
                if not hit_recorded:
                    dx = x - drone_pose[0]; dy = y - drone_pose[1]
                    cth = math.cos(-drone_pose[2]); sth = math.sin(-drone_pose[2])
                    lx = cth*dx - sth*dy; ly = sth*dx + cth*dy
                    obs_local.append((lx, ly))
                    hit_recorded = True
                break  # 이 ray 종료 (더 이상 진행 X)

            else:
                # 관측된 FREE 갱신 (start는 보존)
                if maps.belief[r, c] != maps.map_mask["start"]:
                    maps.belief[r, c] = maps.map_mask["free"]

                # 이 셀의 8-이웃 중 UNKNOWN-like 이 있으면 'frontier 후보'
                found_unknown_like = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = r + dr; cc = c + dc
                        if 0 <= rr < H and 0 <= cc < W and unknown_like[rr, cc]:
                            found_unknown_like = True
                            break
                    if found_unknown_like:
                        break

                # frontier 후보는 ray를 따라 '마지막으로' 갱신하여, 경계에 가장 가까운 FREE를 선택
                if found_unknown_like:
                    frontier_candidate_rc = (r, c)

        # ray가 끝난 뒤, 후보가 있으면 frontier를 1개만 최종 채택
        if frontier_candidate_rc is not None:
            r, c = frontier_candidate_rc
            wx, wy = maps.grid_to_world(r, c)
            dx = wx - drone_pose[0]; dy = wy - drone_pose[1]
            cth = math.cos(-drone_pose[2]); sth = math.sin(-drone_pose[2])
            lx = cth*dx - sth*dy; ly = sth*dx + cth*dy
            frontier_local.append((lx, ly))
            frontier_rc.append((r, c))

    return np.array(frontier_local), np.array(frontier_rc), np.array(obs_local)

def global_frontier_marking(map_info, reset_flag, frontier_cells: list[np.ndarray] = None):
    map = map_info
    belief = map.belief
    gt     = map.gt
    H, W = belief.shape

    UNKNOWN  = map.map_mask["unknown"]
    FREE     = map.map_mask["free"]
    FRONTIER = map.map_mask["frontier"]
    GOAL     = map.map_mask["goal"]

    frontier_belief = copy.deepcopy(belief)

    free_mask = (belief == FREE)

    # 1) unknown-like 정의: unknown 또는 goal
    unknown_like = (belief == UNKNOWN) | (gt == GOAL)

    # 2) 8-neighbor unknown-like 검사
    pad = np.pad(unknown_like, 1, constant_values=False)
    unk_n8 = (
        pad[0:H,0:W]   + pad[0:H,1:W+1]   + pad[0:H,2:W+2] +
        pad[1:H+1,0:W] +                    pad[1:H+1,2:W+2] +
        pad[2:H+2,0:W] + pad[2:H+2,1:W+1] + pad[2:H+2,2:W+2]
    ).astype(np.uint8)

    frontier_mask = free_mask & (unk_n8 > 0)
    rs, cs = np.where(frontier_mask)
    if rs.size:
        frontier_belief[rs, cs] = FRONTIER
    else:
        raise ValueError("Belief Map Initialization is failed")

    return frontier_belief