import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes

from .base_planner import AbstractPlanner
from task.utils.transform_utils import assign_targets_hungarian


class TargetUnknownPlanner(AbstractPlanner):
    def plan(self, map_info, robot_locations, robot_velocities,
             num_agent, cfg) -> dict:

        bel = map_info.belief
        bf  = map_info.belief_frontier
        mm  = map_info.map_mask
        H, W = map_info.H, map_info.W

        FREE_LABEL     = mm["free"]
        UNKNOWN_LABEL  = mm["unknown"]
        OCCUPIED_LABEL = mm["occupied"]
        FRONTIER_LABEL = mm["frontier"]

        free_mask     = (bel == FREE_LABEL)
        unknown_mask  = (bel == UNKNOWN_LABEL)
        occ_mask      = (bel == OCCUPIED_LABEL)
        frontier_mask = (bf  == FRONTIER_LABEL)

        # union_mask: frontier ∪ occupied  (H,W), 0/1 또는 False/True
        union_mask = (occ_mask | frontier_mask).astype(np.uint8)

        # 1) 가장 큰 contour 찾기 (outer boundary)
        contours, _ = cv2.findContours(
                union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            raise ValueError("[target_unknwown] union_mask has no contour. Check occ/frontier.")

        largest = max(contours, key=cv2.contourArea)  # 가장 넓은 contour 하나 선택

        # 2) boundary mask (1-pixel contour)
        contours, _ = cv2.findContours(
            union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        boundary_mask = np.zeros_like(union_mask, dtype=np.uint8)

        # 면적이 너무 작은 noise contour는 걸러도 됨 (예: area > 10 같은 조건)
        for cnt in contours:
            if cv2.contourArea(cnt) < 5:   # 임계치는 맵 해상도에 맞게 튜닝
                continue
            cv2.drawContours(boundary_mask, [cnt], contourIdx=-1, color=1, thickness=1)

        boundary_bool = boundary_mask.astype(bool)

        inside_bool = binary_fill_holes(boundary_bool)

        # 3) radius만큼 contour를 확장해서 band 생성 (안쪽+바깥쪽 모두 포함)
        band_thickness = 0   # [cells] 튜닝 파라미터

        interest_mask = np.zeros_like(boundary_bool, dtype=bool)
        idx_rc = np.argwhere(boundary_bool)  # boundary에 해당하는 (row, col)들

        for r, c in idx_rc:
            r0 = max(0, r - band_thickness)
            r1 = min(H, r + band_thickness + 1)
            c0 = max(0, c - band_thickness)
            c1 = min(W, c + band_thickness + 1)
            interest_mask[r0:r1, c0:c1] = True

        # interest_mask == band_mask 로 사용
        band_mask = interest_mask.copy()

        if not np.any(band_mask):
            raise ValueError("[target_unknwown] band_mask is empty. Check frontier/occupied configuration.")

        # contour 포인트를 (row, col)로 변환
        # largest: shape (N,1,2) with (x,y) = (col,row)
        cnt_xy = largest[:, 0, :]              # (N,2) = (x,y)
        cnt_rc = np.stack([cnt_xy[:, 1],       # row = y
                        cnt_xy[:, 0]],      # col = x
                        axis=1).astype(float)   # (N,2) = (r,c)

        n_x = np.full((H, W), np.nan, dtype=float)
        n_y = np.full((H, W), np.nan, dtype=float)
        band_indices = np.argwhere(band_mask)

        # ------------------------------------------------------------
        # 여기서부터: inside_bool을 이용해 "테두리 바깥 방향" 법선 벡터 n_x, n_y 계산
        # ------------------------------------------------------------
        # inside=1, outside=0
        solid = inside_bool.astype(np.float32)   # (H,W), 0 or 1

        # 너무 날카로운 edge를 조금 부드럽게 → 정상적인 gradient field
        solid_blur = cv2.GaussianBlur(solid, (5, 5), 0)

        # row, col 방향 gradient (중심차분)
        grad_r = np.zeros_like(solid_blur, dtype=float)  # d/d(row)
        grad_c = np.zeros_like(solid_blur, dtype=float)  # d/d(col)

        grad_r[1:-1, :] = 0.5 * (solid_blur[2:, :] - solid_blur[:-2, :])
        grad_c[:, 1:-1] = 0.5 * (solid_blur[:, 2:] - solid_blur[:, :-2])

        # band 바깥은 신경 안 쓰므로 NaN 처리
        grad_r[~band_mask] = np.nan
        grad_c[~band_mask] = np.nan

        # magnitude
        mag = np.sqrt(grad_r**2 + grad_c**2)

        # outward normal = -grad / ||grad||   (grad는 inside(1) 쪽 → 안쪽, 우리는 바깥 방향이 필요)
        n_x = np.full((H, W), np.nan, dtype=float)
        n_y = np.full((H, W), np.nan, dtype=float)

        valid = band_mask & (mag > 1e-6)
        n_x[valid] = -grad_r[valid] / mag[valid]
        n_y[valid] = -grad_c[valid] / mag[valid]

        # ------------------------------------------------------------
        # 7) 모든 band cell에서 outward normal 방향으로 offset 나간 지점을 타깃 후보로 생성
        #    (cos_align, 점수 기반 필터링은 사용하지 않음)
        # ------------------------------------------------------------
        band_indices = np.argwhere(band_mask)

        offset_cells = 10   # grid 상에서 몇 cell 나갈지 (튜닝 파라미터)
        targets_raw = []

        for r, c in band_indices:
            nx = n_x[r, c]
            ny = n_y[r, c]

            if occ_mask[r, c]:
                continue

            if np.isnan(nx) or np.isnan(ny):
                continue

            # outward normal 방향으로 offset_cells 만큼 나간 지점
            r_t = int(round(r + offset_cells * nx))
            c_t = int(round(c + offset_cells * ny))

            # 맵 밖이면 스킵
            if not (0 <= r_t < H and 0 <= c_t < W):
                continue

            # contour 내부로 들어간 셀은 제외
            if inside_bool[r_t, c_t]:
                continue

            if not unknown_mask[r_t, c_t]:
                continue

            r0 = max(0, r_t - 3)
            r1 = min(H, r_t + 3)
            c0 = max(0, c_t - 3)
            c1 = min(W, c_t + 3)

            if np.any(occ_mask[r0:r1, c0:c1]):
                continue

            targets_raw.append([r_t, c_t])

        if len(targets_raw) == 0:
            raise ValueError("[target_unknwown] No valid targets generated along outward normals (band → offset).")

        # ---------------------------------------
        # 1) targets_raw -> binary cand_mask 생성
        # ---------------------------------------
        targets_raw = np.asarray(targets_raw, dtype=int)

        if targets_raw.shape[0] == 0:
            raise ValueError("[target_unknown] targets_raw is empty before clustering.")

        targets_rc_all = np.unique(targets_raw, axis=0)

        cand_mask = np.zeros((H, W), dtype=np.uint8)
        cand_mask[targets_rc_all[:, 0], targets_rc_all[:, 1]] = 1

        # ---------------------------------------
        # 2) cand_mask inflate (장애물은 항상 제외)
        # ---------------------------------------
        inflate_radius = 2  # 필요 시 cfg로 빼도 됨
        if inflate_radius > 0:
            k_size = 2 * inflate_radius + 1
            kernel = np.ones((k_size, k_size), np.uint8)
            cand_mask = cv2.dilate(cand_mask, kernel, iterations=1)

        # 장애물 셀은 무조건 후보에서 제거
        cand_mask[occ_mask] = 0

        if not np.any(cand_mask):
            print("[target_unknown] cand_mask empty after dilation & obstacle removal; fallback to raw targets.")
            cand_mask[targets_rc_all[:, 0], targets_rc_all[:, 1]] = 1  # raw targets만 사용

        # ---------------------------------------
        # 3) 8-연결 컴포넌트로 클러스터링
        # ---------------------------------------
        num_labels, labels = cv2.connectedComponents(cand_mask, connectivity=8)

        if num_labels <= 1:
            # 클러스터가 없거나 전체가 한 덩어리
            largest_cluster_mask = cand_mask.astype(bool)
            areas = np.array([np.count_nonzero(largest_cluster_mask)], dtype=int)
            best_label = 1
        else:
            # -----------------------------
            # 1) 각 라벨별 area, center 계산
            # -----------------------------
            num_comp = num_labels - 1  # 라벨 1..num_labels-1
            areas   = np.zeros(num_comp, dtype=int)
            centers = np.zeros((num_comp, 2), dtype=float)  # (row, col)

            for idx, lbl in enumerate(range(1, num_labels)):
                mask_lbl = (labels == lbl)
                area_lbl = np.count_nonzero(mask_lbl)
                areas[idx] = area_lbl

                if area_lbl > 0:
                    rc = np.argwhere(mask_lbl)   # shape (area_lbl, 2) = (row,col)
                    centers[idx] = rc.mean(axis=0)
                else:
                    centers[idx] = np.array([np.nan, np.nan])

            # -----------------------------
            # 2) area 기준 임계값 적용
            # -----------------------------
            area_max = int(areas.max()) if areas.size > 0 else 0
            # 셀 개수 기준 임계값 (config 에서 가져오고, 없으면 기본값 사용)
            area_thresh = 200

            if area_max < area_thresh:
                # 전체적으로 다 작으면: 그냥 가장 넓은 클러스터 선택 (기존 방식)
                best_idx = int(np.argmax(areas))   # 0..num_comp-1
            else:
                # -----------------------------
                # 3) 넓이가 충분히 큰 클러스터들 중에서
                #    로봇 군집 중심과 가장 가까운 클러스터 선택
                # -----------------------------
                # 로봇 군집 중심 (world -> grid(row,col))
                cluster_center_world = robot_locations.mean(axis=0)  # (x,y)
                r_c, c_c = map_info.world_to_grid(
                    cluster_center_world[0], cluster_center_world[1]
                )
                robot_center_rc = np.array([r_c, c_c], dtype=float)

                # area_thresh 이상인 라벨들만 후보
                candidate_idx = np.where(areas >= area_thresh)[0]

                if candidate_idx.size == 0:
                    # 방어적: 혹시라도 없으면 다시 기존 방식
                    best_idx = int(np.argmax(areas))
                else:
                    # 각 후보의 중심까지 grid 상 거리
                    cand_centers = centers[candidate_idx]          # (K,2)
                    dists = np.linalg.norm(
                        cand_centers - robot_center_rc[None, :], axis=1
                    )                                              # (K,)
                    best_idx = int(candidate_idx[np.argmin(dists)])  # 0..num_comp-1

            # best_idx 는 0..num_comp-1, 실제 라벨 번호는 1..num_labels-1
            best_label = 1 + best_idx
            largest_cluster_mask = (labels == best_label)

        print(
                f"[target_unknown] {num_comp} clusters, "
                f"area_max={area_max}, area_thresh={area_thresh}, "
                f"chosen_label={best_label}, chosen_area={areas[best_idx]}"
            )

        # ---------------------------------------
        # 4) largest_cluster_mask의 "모든 셀"을 샘플링 후보로 사용
        # ---------------------------------------
        largest_cluster_rc = np.argwhere(largest_cluster_mask)  # (M,2) = (row,col)

        if largest_cluster_rc.shape[0] == 0:
            print("[target_unknown] largest cluster has no cells; fallback to targets_rc_all.")
            largest_cluster_rc = targets_rc_all.copy()

        targets_base = largest_cluster_rc   # 샘플링 기준 배열은 이것 하나만 사용
        M = targets_base.shape[0]

        # ---------------------------------------
        # 5) targets_base에서 에이전트 수만큼 샘플링 (spread 제한 포함)
        # ---------------------------------------
        rng = np.random.default_rng(seed=cfg.seed)
        max_pairwise_cells = 80.0  # 필요 시 cfg에서 가져오기
        max_trials = 100

        sampled_idx = None

        for _ in range(max_trials):
            if M >= num_agent:
                candidate_idx = rng.choice(M, size=num_agent, replace=False)
            else:
                candidate_idx = rng.choice(M, size=num_agent, replace=True)

            cand = targets_base[candidate_idx]   # (N,2), 여기서 N=num_agent

            diff = cand[None, :, :] - cand[:, None, :]
            dist_pair = np.linalg.norm(diff, axis=2)
            spread = float(dist_pair.max())

            if spread > max_pairwise_cells:
                continue
            else:
                sampled_idx = candidate_idx
                break

        if sampled_idx is None:
            print(f"[target_unknown] could not sample {num_agent} targets within "
                f"max_pairwise_cells={max_pairwise_cells} from largest cluster. "
                f"Falling back to core-based sampling.")

            K_max = min(num_agent, M)
            core_idx = None
            K_core = 0

            for K in range(K_max, 0, -1):
                if M < K:
                    continue
                cand_idx = rng.choice(M, size=K, replace=False)
                cand_pts = targets_base[cand_idx]

                diff = cand_pts[None, :, :] - cand_pts[:, None, :]
                dist_pair = np.linalg.norm(diff, axis=2)
                spread_K = float(dist_pair.max())

                if spread_K <= max_pairwise_cells:
                    core_idx = cand_idx
                    K_core = K
                    break

            if core_idx is None:
                base_idx = int(rng.integers(0, M))
                core_idx = np.array([base_idx], dtype=int)
                K_core = 1
                print(f"[target_unknown] fallback: using 1 core target index={base_idx} (largest cluster).")

            if K_core >= num_agent:
                sampled_idx = rng.choice(core_idx, size=num_agent, replace=False)
            else:
                sampled_idx = rng.choice(core_idx, size=num_agent, replace=True)

        # 여기서 sampled_idx 는 반드시 0 <= idx < M 범위
        targets_rc = targets_base[sampled_idx]

        # ------------------------------------------------------------
        # 8) Heat map: band=1.0, 타깃 후보=0.5
        # ------------------------------------------------------------
        inflate_radius = 2   # band를 시각화용으로 조금 부풀리기
        k_size = 2 * inflate_radius + 1
        kernel = np.ones((k_size, k_size), np.uint8)

        band_inflated_u8 = cv2.dilate(band_mask.astype(np.uint8), kernel, iterations=1)
        band_inflated = band_inflated_u8.astype(bool)

        heat = np.full((H, W), np.nan, dtype=float)

        # 1) band 전체를 1.0으로 칠하기
        heat[band_inflated] = 1.0

        heat[largest_cluster_mask] = 0.5

        # ------------------------------------------------------------
        # 9) world 좌표 변환 (root 선택 및 시각화용)
        # ------------------------------------------------------------
        target_world = []
        for (r_c, c_c) in targets_rc:
            x, y = map_info.grid_to_world(r_c, c_c)
            target_world.append([x, y])
        target_world = np.asarray(target_world, dtype=float)

        # ------------------------------------------------------------
        # 10) root 선택 + D-MST 업데이트 (region 모드와 동일 전략)
        # ------------------------------------------------------------
        if target_world.shape[0] > 0:
            radius_root = 0.5
            counts     = np.zeros(num_agent, dtype=int)
            mean_dists = np.zeros(num_agent, dtype=float)

            for i in range(num_agent):
                pos_i = robot_locations[i]
                dists = np.linalg.norm(target_world - pos_i, axis=1)

                counts[i]     = np.sum(dists <= radius_root)
                mean_dists[i] = np.mean(dists)

            if np.any(counts > 0):
                root_id = int(np.argmax(counts))
            else:
                root_id = int(np.argmin(mean_dists))
        else:
            root_id = 0  # fallback

        # ------------------------------------------------------------
        # 11) Hungarian matching으로 에이전트 ↔ 타깃 할당
        #      targets_rc: (row, col) 셀 좌표
        # ------------------------------------------------------------
        assigned_rc = assign_targets_hungarian(
            map_info, robot_locations, targets_rc, num_agent
        )

        assigned_rc = np.asarray(assigned_rc, dtype=int)
        # Hungarian에서 (row, col)로 나왔다고 가정 → (col, row)로 맞춤
        assigned_rc = assigned_rc[:, ::-1].copy()

        print('Team decision')

        return {
            "assigned_rc": assigned_rc,
            "root_id"    : root_id,
            "viz": {
                "targets_prob_heat": heat,
                "assigned_rc_viz"  : assigned_rc.copy(),
                "cluster_infos"    : {},
            }
        }
