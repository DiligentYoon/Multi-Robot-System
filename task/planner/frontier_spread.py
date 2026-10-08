"""Deterministic bounded search for compact, separated frontier target sets."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_scores(scores, candidate_ids):
    """Return candidate IDs in agent order, with the original score objective."""
    rows, cols = linear_sum_assignment(-scores[:, candidate_ids])
    assigned = np.empty(scores.shape[0], dtype=int)
    assigned[rows] = np.asarray(candidate_ids)[cols]
    return assigned


def geometry(points):
    if len(points) <= 1: return 0., None
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    pairs = distances[np.triu_indices(len(points), 1)]
    return float(pairs.max()), float(pairs.min())


def select_frontier_targets(points, scores, resolution, max_spread_m=.8, min_pair_m=.08, max_anchor_trials=100):
    """Keep the original assignment if admissible; otherwise search safe frontiers.

    This is a bounded heuristic, not a globally optimal constrained assignment.
    Every selected pair respects the spread cap. If strict search fails, relax
    separation, then repeat safe targets only when no full unique set is found.
    Such fallback decisions are returned explicitly for audit; no episode is dropped.
    """
    points, scores = np.asarray(points, float), np.asarray(scores, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError('Expected nonempty frontier points with shape (M, 2).')
    if scores.ndim != 2 or scores.shape[1] != len(points) or scores.shape[0] == 0:
        raise ValueError('Expected scores with shape (N, M), N >= 1.')
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError('Frontier points and scores must be finite.')
    if not np.isfinite([resolution, max_spread_m, min_pair_m]).all() or resolution <= 0 or max_spread_m <= 0 or min_pair_m < 0:
        raise ValueError('Resolution/spread must be positive; separation must be nonnegative.')
    if int(max_anchor_trials) != max_anchor_trials or max_anchor_trials < 1:
        raise ValueError('max_anchor_trials must be a positive integer.')
    n, m = scores.shape
    # Match TargetUnknownPlanner's physical-to-cell minimum-separation conversion.
    min_cells = max(2, int(round(min_pair_m / resolution)))
    max_cells = max_spread_m / resolution
    initial = assign_scores(scores, np.arange(m)) if m >= n else np.argmax(scores, axis=1)
    spread, separation = geometry(points[initial])
    fallback_reason, attempts = '', 0
    if spread > max_cells + 1e-9 or (separation is not None and separation < min_cells - 1e-9):
        # Stable ties use the np.argwhere frontier order; no random fallback.
        ranked = np.argsort(-scores.max(axis=0), kind='stable')
        anchors = list(dict.fromkeys([*initial.tolist(), *ranked.tolist()]))[:int(max_anchor_trials)]

        def search(required_sep):
            nonlocal attempts
            best_full, best_value, best_partial = None, -np.inf, None
            for anchor in anchors:
                attempts += 1
                selected, remaining = [anchor], list(range(n))
                remaining.remove(int(np.argmax(scores[:, anchor])))
                available = np.linalg.norm(points - points[anchor], axis=1) <= max_cells + 1e-9
                available &= np.linalg.norm(points - points[anchor], axis=1) >= required_sep - 1e-9
                available[anchor] = False
                while len(selected) < n and available.any():
                    candidates = np.flatnonzero(available)
                    local = scores[np.ix_(remaining, candidates)]
                    agent_pos, candidate_pos = np.unravel_index(np.argmax(local), local.shape)
                    chosen = int(candidates[candidate_pos]); selected.append(chosen)
                    remaining.pop(agent_pos)
                    distance = np.linalg.norm(points - points[chosen], axis=1)
                    available &= (distance <= max_cells + 1e-9) & (distance >= required_sep - 1e-9)
                    available[chosen] = False
                if best_partial is None or len(selected) > len(best_partial): best_partial = selected
                if len(selected) == n:
                    assigned = assign_scores(scores, selected)
                    value = float(scores[np.arange(n), assigned].sum())
                    if value > best_value: best_full, best_value = assigned, value
            return best_full, best_partial

        chosen, partial = search(min_cells)
        if chosen is None:
            chosen, partial = search(0.)
            fallback_reason = 'insufficient_safe_frontiers' if m < n else 'bounded_strict_search_exhausted'
            if chosen is None:
                repeated = np.resize(np.asarray(partial, int), n)
                chosen = assign_scores(scores, repeated)
                fallback_reason += ';no_full_unique_set_found'
        initial = chosen
    spread, separation = geometry(points[initial])
    satisfied = spread <= max_cells + 1e-9 and (separation is None or separation >= min_cells - 1e-9)
    audit = {'spread_control': True, 'safe_frontier_count': m, 'num_targets': n,
             'max_spread_m': float(max_spread_m), 'requested_min_pair_m': float(min_pair_m),
             'effective_min_pair_m': float(min_cells * resolution), 'target_spread_m': spread * resolution,
             'target_min_sep_m': None if separation is None else separation * resolution,
             'unique_targets': int(np.unique(initial).size), 'geometry_satisfied': bool(satisfied),
             'fallback_used': bool(fallback_reason), 'fallback_reason': fallback_reason,
             'anchor_attempts': attempts, 'assignment_score': float(scores[np.arange(n), initial].sum())}
    return initial, audit
