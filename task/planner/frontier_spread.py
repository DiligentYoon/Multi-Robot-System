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

# def select_frontier_targets(points, scores, resolution, max_spread_m=0.8,
#                             min_pair_m=0.08, max_anchor_trials=100):
#     """Distance-constrained frontier assignment with first-feasible selection.

#     Keep the original assignment if both distance conditions hold.
#     Otherwise, return the first complete set found by bounded search.
#     If search fails, repeat the largest partial set, relaxing minimum
#     separation while preserving the maximum pairwise distance.
#     """
#     points, scores = np.asarray(points, float), np.asarray(scores, float)
#     if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
#         raise ValueError('Expected nonempty frontier points with shape (M, 2).')
#     if scores.ndim != 2 or scores.shape[1] != len(points) or scores.shape[0] == 0:
#         raise ValueError('Expected scores with shape (N, M), N >= 1.')
#     if not np.isfinite(points).all() or not np.isfinite(scores).all():
#         raise ValueError('Frontier points and scores must be finite.')
#     if not np.isfinite([resolution, max_spread_m, min_pair_m]).all():
#         raise ValueError('Distance parameters must be finite.')
#     if resolution <= 0 or max_spread_m <= 0 or min_pair_m < 0:
#         raise ValueError('Resolution/spread must be positive; separation must be nonnegative.')
#     if int(max_anchor_trials) != max_anchor_trials or max_anchor_trials < 1:
#         raise ValueError('max_anchor_trials must be a positive integer.')

#     n, m = scores.shape
#     min_cells = max(2, int(round(min_pair_m / resolution)))
#     max_cells = max_spread_m / resolution
#     initial = assign_scores(scores, np.arange(m)) if m >= n else np.argmax(scores, axis=1)
#     spread, separation = geometry(points[initial])
#     attempts, fallback_reason = 0, ''

#     if spread > max_cells + 1e-9 or (
#         separation is not None and separation < min_cells - 1e-9
#     ):
#         ranked = np.argsort(-scores.max(axis=0), kind='stable')
#         anchors = list(dict.fromkeys([*initial.tolist(), *ranked.tolist()]))[:int(max_anchor_trials)]
#         chosen, best_partial = None, []

#         for anchor in anchors:
#             attempts += 1
#             selected, remaining = [anchor], list(range(n))
#             remaining.remove(int(np.argmax(scores[:, anchor])))
#             distance = np.linalg.norm(points - points[anchor], axis=1)
#             available = (distance <= max_cells + 1e-9) & (distance >= min_cells - 1e-9)
#             available[anchor] = False

#             while len(selected) < n and available.any():
#                 candidates = np.flatnonzero(available)
#                 local = scores[np.ix_(remaining, candidates)]
#                 agent_pos, candidate_pos = np.unravel_index(np.argmax(local), local.shape)
#                 candidate = int(candidates[candidate_pos])
#                 selected.append(candidate)
#                 remaining.pop(agent_pos)
#                 distance = np.linalg.norm(points - points[candidate], axis=1)
#                 available &= (distance <= max_cells + 1e-9) & (distance >= min_cells - 1e-9)
#                 available[candidate] = False

#             if len(selected) > len(best_partial):
#                 best_partial = selected
#             if len(selected) == n:
#                 chosen = assign_scores(scores, selected)
#                 break

#         if chosen is None:
#             repeated = np.resize(np.asarray(best_partial, int), n)
#             chosen = assign_scores(scores, repeated)
#             fallback_reason = (
#                 'no_full_unique_set_found_in_bounded_search;'
#                 'minimum_separation_relaxed_by_repetition'
#             )
#         initial = chosen

#     spread, separation = geometry(points[initial])
#     spread_satisfied = spread <= max_cells + 1e-9
#     separation_satisfied = separation is None or separation >= min_cells - 1e-9
#     audit = {
#         'spread_control': True,
#         'safe_frontier_count': m,
#         'num_targets': n,
#         'max_spread_m': float(max_spread_m),
#         'requested_min_pair_m': float(min_pair_m),
#         'effective_min_pair_m': float(min_cells * resolution),
#         'target_spread_m': spread * resolution,
#         'target_min_sep_m': None if separation is None else separation * resolution,
#         'unique_targets': int(np.unique(initial).size),
#         'spread_satisfied': bool(spread_satisfied),
#         'separation_satisfied': bool(separation_satisfied),
#         'geometry_satisfied': bool(spread_satisfied and separation_satisfied),
#         'fallback_used': bool(fallback_reason),
#         'fallback_reason': fallback_reason,
#         'anchor_attempts': attempts,
#         'assignment_score': float(scores[np.arange(n), initial].sum())
#     }
#     return initial, audit
def select_frontier_targets(points, scores, resolution, max_spread_m=0.8, min_pair_m=0.08):
    """Filter frontier assignments by pairwise distance, with Frontier fallback."""
    points, scores = np.asarray(points, float), np.asarray(scores, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError('Expected nonempty frontier points with shape (M, 2).')
    if scores.ndim != 2 or scores.shape[1] != len(points) or scores.shape[0] == 0:
        raise ValueError('Expected scores with shape (N, M), N >= 1.')
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError('Frontier points and scores must be finite.')
    if not np.isfinite([resolution, max_spread_m, min_pair_m]).all():
        raise ValueError('Distance parameters must be finite.')
    if resolution <= 0 or max_spread_m <= 0 or min_pair_m < 0:
        raise ValueError('Resolution/spread must be positive; separation must be nonnegative.')

    n, m = scores.shape
    min_cells = max(2, int(round(min_pair_m / resolution)))
    max_cells = max_spread_m / resolution
    original = assign_scores(scores, np.arange(m)) if m >= n else np.argmax(scores, axis=1)
    assigned = original.copy()
    active = np.arange(m)
    removed, reassignments = [], 0
    fallback_reason = ''

    while True:
        selected = points[assigned]
        distances = np.linalg.norm(selected[:, None] - selected[None, :], axis=2)
        rows, cols = np.triu_indices(n, 1)
        invalid = (distances[rows, cols] > max_cells + 1e-9) | (
            distances[rows, cols] < min_cells - 1e-9
        )
        violations = np.flatnonzero(invalid)
        if len(violations) == 0:
            break

        # Remove the lower-scoring target of the first violating agent pair.
        pair = violations[0]
        i, j = int(rows[pair]), int(cols[pair])
        score_i, score_j = scores[i, assigned[i]], scores[j, assigned[j]]
        if score_i < score_j:
            rejected = int(assigned[i])
        elif score_j < score_i:
            rejected = int(assigned[j])
        else:
            rejected = int(max(assigned[i], assigned[j]))

        active = active[active != rejected]
        removed.append(rejected)
        if len(active) < n:
            assigned = original.copy()
            fallback_reason = 'insufficient_candidates_after_distance_filtering'
            break

        assigned = assign_scores(scores, active)
        reassignments += 1

    spread, separation = geometry(points[assigned])
    spread_satisfied = spread <= max_cells + 1e-9
    separation_satisfied = separation is None or separation >= min_cells - 1e-9
    audit = {
        'spread_control': True,
        'safe_frontier_count': m,
        'num_targets': n,
        'max_spread_m': float(max_spread_m),
        'requested_min_pair_m': float(min_pair_m),
        'effective_min_pair_m': float(min_cells * resolution),
        'target_spread_m': spread * resolution,
        'target_min_sep_m': None if separation is None else separation * resolution,
        'unique_targets': int(np.unique(assigned).size),
        'spread_satisfied': bool(spread_satisfied),
        'separation_satisfied': bool(separation_satisfied),
        'geometry_satisfied': bool(spread_satisfied and separation_satisfied),
        'fallback_used': bool(fallback_reason),
        'fallback_reason': fallback_reason,
        'removed_candidate_count': len(removed),
        'reassignment_count': reassignments,
        'assignment_score': float(scores[np.arange(n), assigned].sum())
    }
    return assigned, audit