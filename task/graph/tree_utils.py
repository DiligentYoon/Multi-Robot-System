def split_and_score_local_region(
    local_cell_bounds, 
    agents_pose, 
    agents_yaw,            
    map_info, 
    S_min=5, max_local=50,     
    alpha=1.0, beta=0, gamma=0, goal_w=5.0,
    # --- heading-offset 보상 ---
    off_dist=0.1,          # 진행방향 오프셋(m)
    off_w=0,             # 보상 가중치
    off_sigma=0.15,        # RBF σ (m)
    # --- center 주변 obst 패널티 ---
    safe_win_half=5,
    safe_occ_penalty=1e6,
    # --- NEW: frontier 가중치 ---
    frontier_w=4.0,        # frontier 비율 가중치
):
    import numpy as np
    from math import cos, sin

    belief   = map_info.belief
    map_mask = map_info.map_mask
    H, W     = map_info.H, map_info.W

    # 에이전트 진행방향 오프셋 포인트 (world)
    agents_pose = np.asarray(agents_pose, dtype=float)               # (N,2)
    agents_yaw  = np.asarray(agents_yaw,  dtype=float).reshape(-1)   # (N,)
    dir_vecs    = np.stack([np.cos(agents_yaw), np.sin(agents_yaw)], axis=1)  # (N,2)
    offset_pts  = agents_pose + off_dist * dir_vecs                  # (N,2)

    def stats_local(r0, r1, c0, c1):
        sub = belief[r0:r1, c0:c1]
        area = max(1, (r1 - r0) * (c1 - c0))
        F = int(np.sum(sub == map_mask["frontier"]))
        f = F / area
        h, w = (r1 - r0), (c1 - c0)
        return f, h, w

    r0, r1, c0, c1 = local_cell_bounds
    # 경계 보정
    r0 = max(0, min(r0, H)); r1 = max(0, min(r1, H))
    c0 = max(0, min(c0, W)); c1 = max(0, min(c1, W))

    kept_regions = []
    kept_scores  = []
    stack = [(r0, r1, c0, c1)]

    while stack:
        if len(kept_regions) >= max_local:
            break

        r0, r1, c0, c1 = stack.pop()
        f_frontier_local, h, w = stats_local(r0, r1, c0, c1)

        if (h <= S_min) or (w <= S_min):
            h2, w2 = (r1 - r0), (c1 - c0)
            area = max(1, h2 * w2)

            sub  = belief[r0:r1, c0:c1]
            unk  = np.sum(sub == map_mask["unknown"])   / area
            occ  = np.sum(sub == map_mask["occupied"])  / area
            free = np.sum(sub == map_mask["free"])      / area
            goal = np.sum(sub == map_mask["goal"])      / area
            f_frontier = np.sum(sub == map_mask["frontier"]) / area

            # 셀 중심 (row,col) → world
            r_c = (r0 + r1) // 2
            c_c = (c0 + c1) // 2
            cx, cy = map_info.grid_to_world(r_c, c_c)

            # 기본 점수식에 frontier 항 추가
            # (원하면 -beta*occ, -gamma*d_norm 다시 넣어도 됨)
            d_list = [float(np.hypot(cx - ax, cy - ay)) for (ax, ay) in agents_pose]
            d_norm = np.mean(d_list)
            J_now  = (
                alpha * unk      +     # unknown 비율
                frontier_w * f_frontier +  # frontier 비율
                goal_w * goal    # goal 비율
                # - beta * occ
                # - gamma * d_norm
            )

            # === center 주변 obstacle 패널티 ===
            rmin = max(0, r_c - safe_win_half)
            rmax = min(H, r_c + safe_win_half)
            cmin = max(0, c_c - safe_win_half)
            cmax = min(W, c_c + safe_win_half)

            patch = belief[rmin:rmax, cmin:cmax]
            if np.any(patch == map_mask["occupied"]):
                J_now -= safe_occ_penalty

            # === heading-offset 근접 보상 (RBF, 에이전트 중 최댓값 사용) ===
            dxdy   = offset_pts - np.array([cx, cy])[None, :]
            d2_all = np.sum(dxdy*dxdy, axis=1)                 # (N,)
            rbf    = np.exp(- d2_all / (2.0 * (off_sigma**2)))
            off_bonus = off_w * float(np.max(rbf))
            J_now += off_bonus

            kept_regions.append((r0, r1, c0, c1))
            kept_scores.append(J_now)
            continue

        # KD 분할 (긴 축 기준)
        if h >= w:
            rm = (r0 + r1) // 2
            stack.append((r0, rm, c0, c1))
            stack.append((rm, r1, c0, c1))
        else:
            cm = (c0 + c1) // 2
            stack.append((r0, r1, c0, cm))
            stack.append((r0, r1, cm, c1))

    return kept_regions, kept_scores