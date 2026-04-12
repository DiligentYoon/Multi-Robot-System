import numpy as np

class Node:
    def __init__(self, bounds):
        self.bounds = bounds  # (r0, r1, c0, c1) - 이 노드가 담당하는 영역
        self.left = None      # 왼쪽 / 위쪽 자식 노드
        self.right = None     # 오른쪽 / 아래쪽 자식 노드
        self.axis = None      # 분할 축 (0: row(H), 1: col(W)) for 트리 탐색
        self.value = None     # 분할 값 for 트리 탐색
        self.information_gain = 0 # Node 고유 비용
        self.has_frontier = False
        
    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        r0, r1, c0, c1 = self.bounds
        h, w = (r1 - r0), (c1 - c0)
        return f"Node(bounds={self.bounds}, size={h}x{w}, is_leaf={self.is_leaf()})"


class RegionKDTree:
    def __init__(self, bounds, valid_threshold=0.2, S_min=32):
        """
            Args: 
                bounds: 전체 맵의 경계 (r0, r1, c0, c1)
                S_min : 분할을 멈추는 최소 축 길이
        """
        self.S_min = S_min
        self.valid_threshold = valid_threshold
        self.root = None     # 트리의 루트 노드
        self.leaves = []     # 최종 분할된 영역(리프 노드) 리스트
        
        # KD-Tree 빌드
        self.root = self._build_recursive(bounds)

    def _build_recursive(self, bounds):
        """
            KD-Tree 빌드 함수
            Time Complexity : O(n) = nlog(n)
        """
        r0, r1, c0, c1 = bounds
        h = r1 - r0
        w = c1 - c0

        # 1. 새 노드 생성
        node = Node(bounds)

        # 2. 종료 조건: S_min
        # 높이나 너비 둘 중 하나라도 S_min보다 작거나 같으면 분할을 멈춤.
        if (h <= self.S_min) or (w <= self.S_min):
            self.leaves.append(node) # 리프 노드 리스트에 추가
            return node # 자식이 없는 리프 노드 반환

        # 3. KD 분할: 긴 축을 이등분
        if h >= w:
            # Row(height)가 더 길거나 같으므로, row를 분할 (수평 분할)
            rm = (r0 + r1) // 2
            
            node.axis = 0      # 0번 축(row)
            node.value = rm    # 분할 값
            node.left = self._build_recursive((r0, rm, c0, c1))
            node.right = self._build_recursive((rm, r1, c0, c1))
        else:
            # Column(width)이 더 기므로, column을 분할 (수직 분할)
            cm = (c0 + c1) // 2
            
            node.axis = 1      # 1번 축(col)
            node.value = cm    # 분할 값
            node.left = self._build_recursive((r0, r1, c0, cm))
            node.right = self._build_recursive((r0, r1, cm, c1))

        return node

    def update_node_states(self, map_info, agents_pose,
                        alpha=1.0, beta=0.2, gamma=0.5, goal_w=10.0,
                        frontier_margin_cells: int = 20):
        """
        Frontier가 '직접 포함'된 리프 + Frontier에 '인접(8-이웃 margin)'한 리프까지 유지하도록 프루닝.
        frontier_margin_cells: 팽창 반경(셀 단위). 기본 1칸(8-이웃).
        """
        belief = map_info.belief
        belief_frontier = map_info.belief_frontier
        map_mask = map_info.map_mask
        valid_leaves = []

        # ----- 0) Frontier 팽창 마스크 구성 (numpy-only) -----
        # core: 실제 frontier 위치
        frontier_core = (belief_frontier == map_mask["frontier"]).astype(np.uint8)

        # dilation: frontier 주변 frontier_margin_cells 칸까지 포함
        def dilate_bool(mask: np.ndarray, r: int) -> np.ndarray:
            if r <= 0:
                return mask.astype(bool)
            pad = r
            m = np.pad(mask, pad_width=pad, mode='constant', constant_values=0)
            out = np.zeros_like(mask, dtype=np.uint8)
            # 2r+1 x 2r+1 윈도우 중 최대값(OR)을 취함
            for dr in range(-r, r+1):
                for dc in range(-r, r+1):
                    out |= m[pad+dr:pad+dr+mask.shape[0], pad+dc:pad+dc+mask.shape[1]]
            return out.astype(bool)

        frontier_dil = dilate_bool(frontier_core, frontier_margin_cells)

        def _recursive_search(node: Node):
            r0, r1, c0, c1 = node.bounds

            # 원래 프론티어 포함 여부 (순수)
            has_frontier_core = np.any(frontier_core[r0:r1, c0:c1])
            # 팽창된 프론티어(인접 포함)와의 교차 여부 → 프루닝 기준
            has_frontier_or_adj = np.any(frontier_dil[r0:r1, c0:c1])

            # 디버그/시각화용 플래그들
            node.has_frontier_core = bool(has_frontier_core)
            node.has_frontier      = bool(has_frontier_or_adj)

            # 프루닝: 인접 포함 마스크 기준으로 가지치기
            if not has_frontier_or_adj:
                return

            if node.is_leaf():
                # ===== 점수 계산(기존과 동일) =====
                h, w = (r1 - r0), (c1 - c0)
                area = max(1, h * w)

                unk  = np.sum(belief[r0:r1, c0:c1] == map_mask["unknown"])  / area
                occ  = np.sum(belief[r0:r1, c0:c1] == map_mask["occupied"]) / area
                goal = np.sum(belief[r0:r1, c0:c1] == map_mask["goal"])     / area

                r = (r0 + r1) / 2
                c = (c0 + c1) / 2
                cx, cy = map_info.grid_to_world(r, c)

                d_list = [float(np.hypot(cx - ax, cy - ay)) for (ax, ay) in agents_pose]
                d_norm = np.mean(d_list)

                J_now = alpha*unk - beta*occ - gamma*d_norm + goal_w*goal
                node.information_gain = J_now

                if J_now > self.valid_threshold:
                    valid_leaves.append(node)
                return

            # 재귀 하강
            if node.left:  _recursive_search(node.left)
            if node.right: _recursive_search(node.right)

        if self.root:
            _recursive_search(self.root)

        return valid_leaves