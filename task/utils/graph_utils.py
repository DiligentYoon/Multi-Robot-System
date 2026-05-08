import numpy as np
import heapq
from scipy.ndimage import binary_dilation

from ..base.env.env import MapInfo
from collections import deque

class AStarNode:
    """A* 알고리즘에 사용될 노드 객체 (변경 없음)"""
    def __init__(self, position, parent=None):
        self.position = position  # (row, col) 튜플
        self.parent = parent
        self.g = 0  # 시작 노드로부터의 비용
        self.h = 0  # 목표 노드까지의 추정 비용 (Heuristic)
        self.f = 0  # 총 비용 (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

def astar_search(map_info: MapInfo, 
                 start_pos: np.ndarray | tuple, 
                 end_pos: np.ndarray | tuple,
                 agent_id: int,
                 inflation_radius_cells: int = 7,
                 w_free: float = 1.0,
                 w_unknown: float = 1.0,
                 start_free_radius_cells: int = 5,
                 end_free_radius_cells: int = 5) -> np.ndarray | None:
    """
    A* 알고리즘을 사용하여 최적 경로를 효율적으로 찾습니다.
    start/goal 주변 반경 내에서는 inflated_map 대신 원래 belief 기준으로만
    장애물을 체크합니다.
    """
    # 1. 입력 위치를 튜플로 일관성 있게 변환
    if isinstance(start_pos, np.ndarray):
        start_pos = tuple(start_pos.flatten())
    if isinstance(end_pos, np.ndarray):
        end_pos = tuple(end_pos.flatten())

    H, W = map_info.H, map_info.W
        
    # --- belief / map_mask / inflated ---
    belief   = map_info.belief
    map_mask = map_info.map_mask
    inflated_map = inflate_obstacles(map_info, inflation_radius_cells=inflation_radius_cells)

    # ---- 반경^2 미리 계산 (셀 단위) ----
    r2_start = start_free_radius_cells * start_free_radius_cells
    r2_end   = end_free_radius_cells   * end_free_radius_cells

    # --------- Start Pos 예외처리 ----------
    # 여기서는 "진짜 장애물"만 막기 위해 belief 기준 먼저 확인
    if belief[start_pos] == map_mask["occupied"]:
        # belief 상에서도 장애물이면 근처 free 셀로 옮기는 예외처리
        start_min_r = max(0, start_pos[0] - 5)
        start_max_r = min(H-1, start_pos[0] + 5)
        start_min_c = max(0, start_pos[1] - 5)
        start_max_c = min(W-1, start_pos[1] + 5)

        valid = belief[start_min_r:start_max_r, start_min_c:start_max_c] == map_mask["free"]
        if np.any(valid):
            rr, cc = np.where(valid)
            start_pos = (rr[0] + start_min_r, cc[0] + start_min_c)
        else:
            print(f"[INFO] Agent {agent_id}: Start Pose is true obstacle (belief) and no nearby free cell.")
            return None
    # belief 기준으로는 free인데 inflated 때문에 막혀 있는 경우는 그대로 start_pos 유지

    # --------- End Pos 예외처리 ----------
    if belief[end_pos] == map_mask["occupied"]:
        # 마찬가지로 belief 기준이 우선
        end_min_r = max(0, end_pos[0] - 3)
        end_max_r = min(H-1, end_pos[0] + 3)
        end_min_c = max(0, end_pos[1] - 3)
        end_max_c = min(W-1, end_pos[1] + 3)

        valid = belief[end_min_r:end_max_r, end_min_c:end_max_c] == map_mask["free"]
        if np.any(valid):
            rr, cc = np.where(valid)
            end_pos = (rr[0] + end_min_r, cc[0] + end_min_c)
        else:
            print(f"[INFO] Agent {agent_id}: End Pose is true obstacle (belief) and no nearby free cell.")
            return None
    # 역시 belief 기준으로 free인데 inflated 때문에 막힌 경우는 일단 end_pos 유지

    # 시작/끝 노드 및 맵 정보 초기화
    start_node = AStarNode(start_pos)
    end_node = AStarNode(end_pos)

    # 2. open_list 및 closed_list 초기화
    open_list = []
    heapq.heappush(open_list, start_node)
    
    open_set_lookup = {start_node.position: start_node}
    closed_set = set()

    grid_rows, grid_cols = inflated_map.shape

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.position in closed_set:
            continue
        open_set_lookup.pop(current_node.position, None)
        closed_set.add(current_node.position)

        # 목표 도달 시 경로 역추적
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return np.array(path[::-1], dtype=np.int32)

        cr, cc = current_node.position

        # 8방향 이웃 탐색
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = cr + dr, cc + dc
            neighbor_pos = (nr, nc)

            # 그리드 범위 확인
            if not (0 <= nr < grid_rows and 0 <= nc < grid_cols):
                continue
            if neighbor_pos in closed_set:
                continue

            # ---- 여기서 "어느 맵을 사용할지" 결정 ----
            # start/goal 근처에서는 inflated 무시, 그 외에서는 inflated 사용
            ds2 = (nr - start_pos[0])**2 + (nc - start_pos[1])**2
            de2 = (nr - end_pos[0])**2   + (nc - end_pos[1])**2

            if (ds2 <= r2_start) or (de2 <= r2_end):
                # start 또는 end 주변: 원래 belief 기준으로만 장애물 체크
                occ_val = belief[nr, nc]
            else:
                # 그 외 구간: inflated_map 기준
                occ_val = inflated_map[nr, nc]

            if occ_val == map_mask["occupied"]:
                continue

            # 3. 비용 계산 및 노드 업데이트
            move_cost = 1.414 if abs(dr) == 1 and abs(dc) == 1 else 1.0

            # --- FREE / UNKNOWN 가중치 (원래 로직 그대로) ---
            cell_type = belief[nr, nc]  # 비용은 belief 기준
            if cell_type == map_mask["free"]:
                move_cost *= w_free
            elif cell_type == map_mask["unknown"]:
                move_cost *= w_unknown

            g_cost = current_node.g + move_cost
            
            if neighbor_pos not in open_set_lookup or g_cost < open_set_lookup[neighbor_pos].g:
                dx = nr - end_node.position[0]
                dy = nc - end_node.position[1]
                h_cost = np.sqrt(dx*dx + dy*dy)
                
                neighbor_node = AStarNode(neighbor_pos, current_node)
                neighbor_node.g = g_cost
                neighbor_node.h = h_cost
                neighbor_node.f = g_cost + h_cost
                
                heapq.heappush(open_list, neighbor_node)
                open_set_lookup[neighbor_pos] = neighbor_node

    print(f"[INFO] Agent {agent_id}: No path found")
    return None



def inflate_obstacles(map_info: MapInfo, inflation_radius_cells: int = 2) -> np.ndarray:
    """
    [성능 개선] Scipy를 사용하여 Belief map의 장애물을 효율적으로 팽창시킵니다.
    """
    belief_map = map_info.belief
    map_mask = map_info.map_mask
    
    if inflation_radius_cells <= 0:
        return np.copy(belief_map)
        
    # 1. 장애물만 1로 표시된 이진 맵 생성
    obstacle_mask = (belief_map == map_mask["occupied"])

    # 2. 팽창에 사용할 구조 요소(커널) 생성
    # inflation_radius_cells가 5이면 11x11 크기의 정사각형 커널
    structure_size = 2 * inflation_radius_cells + 1
    structure = np.ones((structure_size, structure_size))
    
    # 3. Scipy의 binary_dilation 함수를 사용하여 팽창 연산 수행
    dilated_obstacle_mask = binary_dilation(obstacle_mask, structure=structure)
    
    # 4. 원본 맵에 팽창된 장애물 영역을 덮어쓰기
    inflated_map = np.copy(belief_map)
    inflated_map[dilated_obstacle_mask] = map_mask["occupied"]
    
    return inflated_map

# -------------------------------
#   Helper: Bresenham line
# -------------------------------
def bresenham_line(r0: int, c0: int, r1: int, c1: int):
    """(row, col) 기준 Bresenham line"""
    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    step_r = 1 if r1 >= r0 else -1
    step_c = 1 if c1 >= c0 else -1

    err = dr - dc
    r, c = r0, c0

    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += step_r
        if e2 < dr:
            err += dr
            c += step_c

    return points  # list[(r,c)]


def has_line_of_sight(inflated_map: np.ndarray,
                      map_mask: dict,
                      start_rc: tuple,
                      end_rc: tuple):
    """
    inflated_map 상에서 start_rc → end_rc 직선 경로에
    occupied가 하나라도 있으면 False + 첫 충돌 인덱스 반환.
    """
    H, W = inflated_map.shape
    r0, c0 = start_rc
    r1, c1 = end_rc

    line = bresenham_line(r0, c0, r1, c1)
    for idx, (r, c) in enumerate(line):
        if not (0 <= r < H and 0 <= c < W):
            # 맵 밖이면 막힌 것으로 취급
            return False, idx, line
        if inflated_map[r, c] == map_mask["occupied"]:
            return False, idx, line
    return True, None, line

# ---------------------------------------
#   Main: 직선 + 장애물 우회 기반 경로 계획
# ---------------------------------------
def straight_then_detour_search(map_info: MapInfo,
                                start_pos: np.ndarray | tuple,
                                end_pos: np.ndarray | tuple,
                                agent_id: int,
                                inflation_radius_cells: int = 5,
                                max_detour_radius: int = 15) -> np.ndarray | None:
    """
    1) start_pos ~ end_pos 직선 상에 장애물이 없으면: Bresenham 직선 경로 반환
    2) 직선 경로가 inflated obstacle에 의해 막히면:
       - 첫 충돌 지점 근처에서 free cell pivot 하나를 BFS로 탐색
       - pivot은 start/end와 모두 line-of-sight가 있어야 함
       - 최종 경로: start -> pivot 직선 + pivot -> end 직선

    반환: (L,2) int32 array, 각 row가 (row, col).
    실패 시 None.
    """

    # 1. 입력을 튜플로 정리
    if isinstance(start_pos, np.ndarray):
        start_pos = tuple(start_pos.flatten())
    if isinstance(end_pos, np.ndarray):
        end_pos = tuple(end_pos.flatten())

    H, W = map_info.H, map_info.W
    map_mask = map_info.map_mask
    belief = map_info.belief

    # 2. 장애물 팽창
    inflated_map = inflate_obstacles(map_info, inflation_radius_cells=inflation_radius_cells)

    # 3. 시작/끝이 inflated obstacle 위에 있으면 근처 free 셀로 보정
    def _snap_to_free(pos, search_radius: int):
        r, c = pos
        if inflated_map[r, c] != map_mask["occupied"]:
            return pos

        r_min = max(0, r - search_radius)
        r_max = min(H - 1, r + search_radius)
        c_min = max(0, c - search_radius)
        c_max = min(W - 1, c + search_radius)

        local = inflated_map[r_min:r_max+1, c_min:c_max+1]
        free_mask = (local != map_mask["occupied"])
        if np.any(free_mask):
            rr, cc = np.where(free_mask)
            rr0, cc0 = rr[0] + r_min, cc[0] + c_min
            return (int(rr0), int(cc0))
        else:
            return None

    start_pos = _snap_to_free(start_pos, search_radius=5)
    end_pos   = _snap_to_free(end_pos,   search_radius=1)
    if start_pos is None or end_pos is None:
        print(f"[straight-detour] Agent {agent_id}: start or end is fully blocked around.")
        return None

    # 4. 직선 line-of-sight 체크
    los_ok, block_idx, line = has_line_of_sight(inflated_map, map_mask,
                                                start_rc=start_pos,
                                                end_rc=end_pos)
    if los_ok:
        # 모든 셀이 free/unknown → 그대로 사용
        return np.array(line, dtype=np.int32)

    # 5. 첫 충돌 지점 주변에서 pivot 탐색
    # block_idx: line 상에서 처음 막힌 셀의 인덱스
    # 그 직전 셀까지는 free였다고 가정
    # block_cell = line[block_idx]
    # prev_cell  = line[max(0, block_idx - 1)]
    block_cell = line[block_idx]

    # BFS로 block_cell 주변 free cell 탐색
    # 탐색은 일정 반경(max_detour_radius) 안에서만.
    visited = np.zeros((H, W), dtype=bool)
    q = deque()

    # block_cell은 obstacle이라 neighbor만 큐에 넣음
    br, bc = block_cell
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = br + dr, bc + dc
            if 0 <= nr < H and 0 <= nc < W:
                if inflated_map[nr, nc] != map_mask["occupied"]:
                    visited[nr, nc] = True
                    q.append((nr, nc, 1))  # (row, col, distance_from_block)

    best_pivot = None
    best_cost  = np.inf

    # 직선 길이를 기준으로 대략적인 거리 cost를 정의 (heuristic)
    def approx_cost(p):
        # start -> p -> end 의 유클리드 거리 합
        sr, sc = start_pos
        er, ec = end_pos
        pr, pc = p
        d1 = np.hypot(pr - sr, pc - sc)
        d2 = np.hypot(er - pr, ec - pc)
        return d1 + d2

    while q:
        r, c, d = q.popleft()
        if d > max_detour_radius:
            continue

        # (r,c)가 start, end와 둘 다 line-of-sight면 pivot 후보
        los_s, _, _ = has_line_of_sight(inflated_map, map_mask, start_rc=start_pos, end_rc=(r, c))
        if not los_s:
            # start에서 안 보이면 패스
            pass
        else:
            los_e, _, _ = has_line_of_sight(inflated_map, map_mask, start_rc=(r, c), end_rc=end_pos)
            if los_e:
                # 양쪽에서 다 보이는 pivot
                cost = approx_cost((r, c))
                if cost < best_cost:
                    best_cost = cost
                    best_pivot = (r, c)

        # 이웃 확장 (8방향)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if visited[nr, nc]:
                    continue
                if inflated_map[nr, nc] == map_mask["occupied"]:
                    continue
                visited[nr, nc] = True
                q.append((nr, nc, d + 1))

    if best_pivot is None:
        # pivot을 못 찾은 경우 → None (혹은 A* fallback을 걸어도 됨)
        print(f"[straight-detour] Agent {agent_id}: no valid pivot found around obstacle.")
        return None

    # 6. 최종 경로: start → pivot → end 를 모두 직선(bresenham)으로 이어붙임
    #    (중복되는 pivot 셀 한 번만 포함)
    s2p = bresenham_line(start_pos[0], start_pos[1], best_pivot[0], best_pivot[1])
    p2e = bresenham_line(best_pivot[0], best_pivot[1], end_pos[0], end_pos[1])

    # s2p 마지막과 p2e 첫 번째가 같은 셀이므로 하나만 유지
    full_path = s2p + p2e[1:]
    return np.array(full_path, dtype=np.int32)