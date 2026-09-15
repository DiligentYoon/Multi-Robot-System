import numpy as np
import cv2

from .base_planner import AbstractPlanner
from task.utils.transform_utils import assign_targets_hungarian


class TargetUnknownPlanner(AbstractPlanner):

    def __init__(self):
        super().__init__()

        # ---------- ray marching ----------
        self.push_ratio       = 0.35   # push = push_ratio * sensor_cells
        self.push_min         = 2      # [cells]
        self.march_range_mult = 1.5    # max_steps = push + mult * sensor_cells
        self.march_substep    = 0.5    # 대각선 관통 방지 
        self.allow_shallow    = True
        self.fallback_push    = 2
        self.seed_stride      = 4      # band seed 다운샘플링 [cells]

        # ---------- clearance ----------
        self.d_obs_default    = 0.05   # [m]
        self.clearance_min    = 2      # [cells]

        # ---------- normal ----------
        self.grad_blur_ksize  = 11     # 벽 두께보다 크게 (홀수)

        # ---------- clustering ----------
        self.min_contour_area = 5
        self.link_extra       = 2      # link 반경 = seed_stride + link_extra
        self.link_min         = 4      # link 반경 하한 [cells]
        self.area_ratio       = 0.3    # area_thresh = ratio * area_max
        self.area_min         = 20     # area_thresh 하한 [cells]

        # ---------- sampling ----------
        self.min_pair_m       = 0.08
        self.d_safe_default   = 0.10
        self.max_pairwise_cells = 80.0
        self.max_trials       = 100
        self.radius_root      = 0.5

        # ---------- viz ----------
        self.viz_inflate_radius = 2
        self.viz_cand_radius    = 2

        # ---------- 실행 중 확정되는 파생값 [cells] ----------
        self.res_m           = None
        self.sensor_cells    = None
        self.push            = None
        self.max_steps       = None
        self.clearance_cells = None
        self.min_pair_cells  = None
        self._param_key      = None

        # ---------- 지속 상태 ----------
        self.rng = None                # plan()마다 재생성 금지 (deadlock 방지)

    # ==================================================================
    # 파라미터 / rng
    # ==================================================================
    def _resolve_params(self, map_info, cfg):
        """해상도·센서 기반 cell 단위 파라미터 확정 (변경 시에만 재계산)."""
        res    = float(map_info.res_m)
        srng   = float(cfg.sensor_range)
        d_obs  = float(getattr(cfg, "d_obs",  self.d_obs_default))
        d_safe = float(getattr(cfg, "d_safe", self.d_safe_default))

        key = (res, srng, d_obs, d_safe)
        if key == self._param_key:
            return
        self._param_key = key

        self.res_m           = res
        self.sensor_cells    = max(1, int(round(srng / res)))
        self.push            = max(self.push_min,
                                   int(round(self.push_ratio * self.sensor_cells)))
        self.max_steps       = self.push + int(round(self.march_range_mult * self.sensor_cells))
        self.clearance_cells = max(self.clearance_min, int(round(d_obs / res)))

        pair_m = self.min_pair_m if self.min_pair_m is not None else d_safe
        self.min_pair_cells = max(2, int(round(pair_m / res)))

        print(f"[target_unknown] params: res={res}m sensor={self.sensor_cells}cell "
              f"push={self.push} max_steps={self.max_steps} "
              f"clearance={self.clearance_cells} min_pair={self.min_pair_cells} "
              f"stride={self.seed_stride}")

    def _ensure_rng(self, cfg):
        if self.rng is None:
            self.rng = np.random.default_rng(getattr(cfg, "seed", None))

    # ==================================================================
    # ray marching
    # ==================================================================
    def _march_to_unknown(self, r0, c0, nr, nc,
                          occ_mask, unknown_mask, known_free_mask, H, W,
                          push=None, max_steps=None):
        """
        (r0,c0)에서 normal (nr,nc) 방향으로 1 cell 이하 간격으로 전진.
          - occupied를 만나면 폐기          → occlusion 차단
          - unknown 진입 후 known 복귀 폐기 → 얇은 벽 관통 차단
          - unknown 진입 지점에서 push만큼 더 들어간 셀을 타깃으로 반환

        Args:
            nr : normal의 row 성분 (= n_x)
            nc : normal의 col 성분 (= n_y)
        Returns:
            ((r,c), reason) 또는 (None, reason)
        """
        push      = self.push      if push      is None else push
        max_steps = self.max_steps if max_steps is None else max_steps
        substep   = self.march_substep

        entered_s = None
        deepest   = None
        last_rc   = None

        n_sub = int(round(max_steps / substep))
        for k in range(1, n_sub + 1):
            s  = k * substep
            rr = int(round(r0 + s * nr))
            cc = int(round(c0 + s * nc))

            if (rr, cc) == last_rc:          # substep으로 인한 같은 셀 재방문
                continue
            last_rc = (rr, cc)

            if not (0 <= rr < H and 0 <= cc < W):
                return None, "oob"
            if occ_mask[rr, cc]:
                return None, "occ_hit"       # 벽에 막힘

            if entered_s is None:
                if unknown_mask[rr, cc]:
                    entered_s = s
                    deepest   = (rr, cc)
            else:
                if known_free_mask[rr, cc]:
                    return None, "reenter"   # 얇은 벽 스치고 넘어감
                deepest = (rr, cc)
                if s - entered_s >= push:
                    return (rr, cc), "ok"

        if entered_s is None:
            return None, "no_unknown"
        if self.allow_shallow and deepest is not None:
            return deepest, "shallow"
        return None, "short"

    # ==================================================================
    # farthest-point sampling (중복 없이 최대 분산, spread 제약 준수)
    # ==================================================================
    def _fps_sample(self, pts, k, rng, max_spread):
        """
        pts: (M,2) float. anchor 반경 max_spread/2 안에서 FPS.
        anchor 기준 반경 R 안의 점들은 서로 최대 2R = max_spread 이내이므로
        spread 제약이 자동으로 보장된다.
        """
        M = pts.shape[0]
        R = max_spread * 0.5

        subset = None
        for _ in range(20):                       # anchor 재시도
            a = int(rng.integers(0, M))
            d_a = np.linalg.norm(pts - pts[a], axis=1)
            cand = np.where(d_a <= R)[0]
            if cand.size >= k:
                subset = cand
                break
            if subset is None or cand.size > subset.size:
                subset = cand                     # 최선의 차선책 보관

        if subset is None or subset.size == 0:
            subset = np.arange(M)

        sub_pts = pts[subset]
        n = sub_pts.shape[0]

        idx = [int(rng.integers(0, n))]
        d = np.linalg.norm(sub_pts - sub_pts[idx[0]], axis=1)
        while len(idx) < min(k, n):
            nxt = int(np.argmax(d))
            idx.append(nxt)
            d = np.minimum(d, np.linalg.norm(sub_pts - sub_pts[nxt], axis=1))

        while len(idx) < k:                       # n < k 인 경우만 (불가피한 중복)
            idx.append(idx[len(idx) % n])

        return subset[np.array(idx, dtype=int)]

    # ==================================================================
    # plan
    # ==================================================================
    def plan(self, map_info, robot_locations, robot_velocities,
             num_agent, cfg) -> dict:

        self._resolve_params(map_info, cfg)
        self._ensure_rng(cfg)

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

        # START 라벨 보존 때문에 free도 unknown도 아닌 셀이 있을 수 있음
        known_free_mask = ~unknown_mask & ~occ_mask

        union_mask = (occ_mask | frontier_mask).astype(np.uint8)

        # ------------------------------------------------------------
        # 1) sanity check
        # ------------------------------------------------------------
        contours, _ = cv2.findContours(
            union_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            raise ValueError("[target_unknown] union_mask has no contour. Check occ/frontier.")

        boundary_mask = np.zeros_like(union_mask, dtype=np.uint8)
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_contour_area:
                continue
            cv2.drawContours(boundary_mask, [cnt], contourIdx=-1, color=1, thickness=1)
        boundary_bool = boundary_mask.astype(bool)   # 시각화/디버그용

        # ------------------------------------------------------------
        # 2) inside_bool = known 영역
        # ------------------------------------------------------------
        inside_bool = ~unknown_mask

        # ------------------------------------------------------------
        # 3) band = frontier
        # ------------------------------------------------------------
        band_mask = frontier_mask & ~occ_mask
        if not np.any(band_mask):
            raise ValueError("[target_unknown] band_mask is empty. Check frontier/occupied configuration.")

        # ------------------------------------------------------------
        # 4) outward normal = -grad(known)
        # ------------------------------------------------------------
        solid = inside_bool.astype(np.float32)

        ksz = self.grad_blur_ksize
        if ksz % 2 == 0:
            ksz += 1
        solid_blur = cv2.GaussianBlur(solid, (ksz, ksz), 0)

        grad_r = np.zeros_like(solid_blur, dtype=float)
        grad_c = np.zeros_like(solid_blur, dtype=float)
        grad_r[1:-1, :] = 0.5 * (solid_blur[2:, :] - solid_blur[:-2, :])
        grad_c[:, 1:-1] = 0.5 * (solid_blur[:, 2:] - solid_blur[:, :-2])

        grad_r[~band_mask] = np.nan
        grad_c[~band_mask] = np.nan
        mag = np.sqrt(grad_r**2 + grad_c**2)

        n_x = np.full((H, W), np.nan, dtype=float)   # row 성분
        n_y = np.full((H, W), np.nan, dtype=float)   # col 성분

        valid = band_mask & (mag > 1e-6)
        n_x[valid] = -grad_r[valid] / mag[valid]
        n_y[valid] = -grad_c[valid] / mag[valid]

        # ------------------------------------------------------------
        # 5) seed 다운샘플링 + ray marching
        # ------------------------------------------------------------
        band_indices_all = np.argwhere(band_mask)
        stride = max(1, int(self.seed_stride))
        band_indices = band_indices_all[::stride]

        march = self._march_to_unknown
        rad   = self.clearance_cells

        targets_raw   = []
        occ_hit_seeds = []
        reject        = {}

        for r, c in band_indices:
            nr = n_x[r, c]
            nc = n_y[r, c]
            if np.isnan(nr) or np.isnan(nc):
                continue

            hit, why = march(r, c, nr, nc,
                             occ_mask, unknown_mask, known_free_mask, H, W)
            reject[why] = reject.get(why, 0) + 1

            if hit is None:
                if why == "occ_hit":
                    occ_hit_seeds.append([r, c])
                continue
            r_t, c_t = hit

            if inside_bool[r_t, c_t]:
                continue
            if not unknown_mask[r_t, c_t]:
                continue

            r0 = max(0, r_t - rad); r1 = min(H, r_t + rad + 1)
            c0 = max(0, c_t - rad); c1 = min(W, c_t + rad + 1)
            if np.any(occ_mask[r0:r1, c0:c1]):
                reject["clearance"] = reject.get("clearance", 0) + 1
                continue

            targets_raw.append([r_t, c_t])

        print(f"[target_unknown] band={len(band_indices_all)} seeds={len(band_indices)} "
              f"kept={len(targets_raw)} | "
              + " ".join(f"{k}={v}" for k, v in sorted(reject.items())))

        # ------------------------------------------------------------
        # 6) 빈 결과 fallback
        # ------------------------------------------------------------
        if len(targets_raw) == 0:
            for r, c in band_indices:
                nr, nc = n_x[r, c], n_y[r, c]
                if np.isnan(nr) or np.isnan(nc):
                    continue
                hit, _ = march(r, c, nr, nc,
                               occ_mask, unknown_mask, known_free_mask, H, W,
                               push=self.fallback_push)
                if hit is not None and unknown_mask[hit[0], hit[1]]:
                    targets_raw.append(list(hit))
            if targets_raw:
                print(f"[target_unknown] relaxed push fallback -> {len(targets_raw)} targets.")

        if len(targets_raw) == 0:
            print("[target_unknown] no reachable unknown; fallback to frontier cells.")
            fb = np.argwhere(frontier_mask & ~occ_mask)
            if fb.shape[0] == 0:
                raise ValueError("[target_unknown] no frontier at all — fully explored?")
            targets_raw = fb.tolist()

        # ---------------------------------------
        # 7) cand_raw(원본) / link_mask(연결 판정용) 분리   ★ 수정 2
        # ---------------------------------------
        targets_raw    = np.asarray(targets_raw, dtype=int)
        targets_rc_all = np.unique(targets_raw, axis=0)

        cand_raw = np.zeros((H, W), dtype=np.uint8)
        cand_raw[targets_rc_all[:, 0], targets_rc_all[:, 1]] = 1
        cand_raw[occ_mask] = 0

        cand_raw_bool = cand_raw.astype(bool)
        if not np.any(cand_raw_bool):
            raise ValueError("[target_unknown] cand_raw empty after obstacle removal.")

        # 연결 판정용: seed 간격보다 넉넉히 부풀려 과분할 방지
        link_r = max(self.link_min, stride + self.link_extra)
        k_link = np.ones((2 * link_r + 1, 2 * link_r + 1), np.uint8)
        link_mask = cv2.dilate(cand_raw, k_link, iterations=1)
        link_mask[occ_mask] = 0

        # ---------------------------------------
        # 8) 8-연결 클러스터링 (라벨은 link, 타깃은 cand_raw)
        # ---------------------------------------
        num_labels, labels = cv2.connectedComponents(link_mask, connectivity=8)

        if num_labels <= 1:
            largest_cluster_mask = cand_raw_bool
            areas       = np.array([np.count_nonzero(largest_cluster_mask)], dtype=int)
            best_label  = 1
            num_comp, best_idx = 1, 0
            area_max    = int(areas[0])
            area_thresh = max(self.area_min, int(self.area_ratio * area_max))
        else:
            num_comp = num_labels - 1
            areas   = np.zeros(num_comp, dtype=int)
            centers = np.zeros((num_comp, 2), dtype=float)
            cell_lists = [None] * num_comp

            for idx, lbl in enumerate(range(1, num_labels)):
                # area/center는 부풀린 셀이 아니라 '실제 후보' 기준
                mask_lbl = (labels == lbl) & cand_raw_bool
                rc = np.argwhere(mask_lbl)
                cell_lists[idx] = rc
                areas[idx] = rc.shape[0]
                centers[idx] = rc.mean(axis=0) if rc.shape[0] > 0 else np.array([np.nan, np.nan])

            area_max = int(areas.max()) if areas.size > 0 else 0
            # ★ 수정: 절대 게이트 → 상대 게이트 (항상 거리 비교가 일어나도록)
            area_thresh = max(self.area_min, int(self.area_ratio * area_max))

            candidate_idx = np.where(areas >= area_thresh)[0]
            if candidate_idx.size == 0:
                best_idx = int(np.argmax(areas))
            else:
                # 로봇 각각의 셀 좌표 (군집 평균이 아닌 개별 로봇 기준)
                robot_cells = np.asarray(
                    [map_info.world_to_grid(p[0], p[1]) for p in robot_locations],
                    dtype=float)                                    # (N,2) = (row,col)

                # centroid가 아닌 '최근접 셀' 거리 (C자 클러스터 왜곡 방지)
                dists = np.empty(candidate_idx.size, dtype=float)
                for j, ci in enumerate(candidate_idx):
                    rc = cell_lists[ci]
                    sub = rc[::10] if rc.shape[0] > 200 else rc     # 비용 절감
                    d = np.linalg.norm(
                        sub[:, None, :] - robot_cells[None, :, :], axis=2)
                    dists[j] = float(d.min())

                best_idx = int(candidate_idx[np.argmin(dists)])

            best_label = 1 + best_idx
            largest_cluster_mask = (labels == best_label) & cand_raw_bool

        print(f"[target_unknown] {num_comp} clusters, "
              f"areas={sorted(areas.tolist(), reverse=True)[:8]}, "
              f"area_max={area_max}, area_thresh={area_thresh}, "
              f"chosen_label={best_label}, chosen_area={areas[best_idx]}")

        # ---------------------------------------
        # 9) 샘플링   ★ 수정 1
        # ---------------------------------------
        largest_cluster_rc = np.argwhere(largest_cluster_mask)
        if largest_cluster_rc.shape[0] == 0:
            print("[target_unknown] largest cluster has no cells; fallback to targets_rc_all.")
            largest_cluster_rc = targets_rc_all.copy()

        targets_base = largest_cluster_rc
        M = targets_base.shape[0]

        rng                = self.rng
        max_pairwise_cells = self.max_pairwise_cells
        min_pair_cells     = self.min_pair_cells

        sampled_idx = None
        if M >= num_agent:
            iu = np.triu_indices(num_agent, k=1)
            for _ in range(self.max_trials):
                candidate_idx = rng.choice(M, size=num_agent, replace=False)
                cand = targets_base[candidate_idx]

                diff = cand[None, :, :] - cand[:, None, :]
                dist_pair = np.linalg.norm(diff, axis=2)

                spread  = float(dist_pair.max())
                min_sep = float(dist_pair[iu].min()) if num_agent > 1 else np.inf

                if spread <= max_pairwise_cells and min_sep >= min_pair_cells:
                    sampled_idx = candidate_idx
                    break

        # ★ replace=True 제거 → FPS로 중복 없이 최대 분산
        if sampled_idx is None:
            sampled_idx = self._fps_sample(
                targets_base.astype(float), num_agent, rng, max_pairwise_cells)

            cand = targets_base[sampled_idx]
            diff = cand[None, :, :] - cand[:, None, :]
            dp = np.linalg.norm(diff, axis=2)
            iu = np.triu_indices(num_agent, k=1)
            print(f"[target_unknown] FPS fallback (M={M}): "
                  f"spread={dp.max():.1f} min_sep="
                  f"{(dp[iu].min() if num_agent > 1 else 0):.1f} "
                  f"(required >= {min_pair_cells})")

        targets_rc = targets_base[sampled_idx]

        # ------------------------------------------------------------
        # 10) Heat map (디버그)
        #     1.00 band / 0.85 occ_hit / 0.65 후보 / 0.50 선택된 클러스터
        # ------------------------------------------------------------
        k_band = np.ones((2 * self.viz_inflate_radius + 1,) * 2, np.uint8)
        band_inflated = cv2.dilate(band_mask.astype(np.uint8), k_band,
                                   iterations=1).astype(bool)

        k_cand = np.ones((2 * self.viz_cand_radius + 1,) * 2, np.uint8)
        cand_viz = cv2.dilate(cand_raw, k_cand, iterations=1).astype(bool)
        chosen_viz = cv2.dilate(largest_cluster_mask.astype(np.uint8), k_cand,
                                iterations=1).astype(bool)

        heat = np.full((H, W), np.nan, dtype=float)
        heat[band_inflated] = 1.00
        if occ_hit_seeds:
            occ_hit_seeds = np.asarray(occ_hit_seeds, dtype=int)
            heat[occ_hit_seeds[:, 0], occ_hit_seeds[:, 1]] = 0.85
        heat[cand_viz]   = 0.65
        heat[chosen_viz] = 0.50

        # ------------------------------------------------------------
        # 11) world 변환 + root 선택
        # ------------------------------------------------------------
        target_world = np.asarray(
            [map_info.grid_to_world(r_c, c_c) for (r_c, c_c) in targets_rc],
            dtype=float)

        if target_world.shape[0] > 0:
            radius_root = self.radius_root
            counts     = np.zeros(num_agent, dtype=int)
            mean_dists = np.zeros(num_agent, dtype=float)

            for i in range(num_agent):
                dists = np.linalg.norm(target_world - robot_locations[i], axis=1)
                counts[i]     = np.sum(dists <= radius_root)
                mean_dists[i] = np.mean(dists)

            root_id = int(np.argmax(counts)) if np.any(counts > 0) \
                      else int(np.argmin(mean_dists))
        else:
            root_id = 0

        # ------------------------------------------------------------
        # 12) Hungarian matching
        #     assign_targets_hungarian는 입력 원소를 그대로 반환 → (row, col)
        # ------------------------------------------------------------
        assigned_rc = assign_targets_hungarian(
            map_info, robot_locations, targets_rc, num_agent)

        assigned_rc = np.asarray(assigned_rc, dtype=int)
        assigned_rc = assigned_rc[:, ::-1].copy()   # (row,col) → (col,row)

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