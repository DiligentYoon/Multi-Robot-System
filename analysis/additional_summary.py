"""Revision evaluation from saved logs; the simulator and CMR definition are unchanged.

python -m analysis.additional --root results/quantitative/round_0 --out_dir additional_results

"""

import argparse

import hashlib

import json

import re

import subprocess

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import numpy as np

import pandas as pd

from PIL import Image

from .quantitative_summary import DIR_RE, parse_summary, wilson_ci

VARIANT_RE = re.compile(r'^agent_(\d+)_(.+)$')

KNOWN_REASONS = ['goal', 'obstacle_collision', 'robot_collision', 'frozen', 'timeout']

METHOD_NAMES = {'default': 'Full', 'safe_ablation': 'w/o ICA', 'conn_ablation': 'w/o CM', 'team_ablation': 'Frontier', 'frontier_spread': 'Frontier (spread control)'}

METRICS = ['control_compute_mean_ms', 'control_compute_p95_ms', 'coordination_mean_ms', 'coordination_p95_ms', 'routing_mean_ms', 'max_active_obstacle_constraints', 'cmr_pct', 'spr_pct', 'mcm_m2', 'free_coverage', 'fce_mean', 'conn_violation_total_s', 'conn_violation_longest_s', 'conn_exceedance_max_m', 'conn_negative_mean_m2', 'runtime_mean_ms', 'runtime_p95_ms', 'runtime_max_ms', 'runtime_over_100ms_pct', 'first_arrival_logged_s', 'goal_fraction_first', 'goal_distance_mean_m', 'goal_distance_max_m']

CORE_COLS = ['step', 'time', 'x', 'y', 'a_nom', 'w_nom']

METRICS += ['planner_fallback_pct', 'planner_geometry_violation_pct']

TIMING_COLUMNS = {'cbf_solve': 'solve_s', 'coordination': 'coordination_s', 'routing': 'routing_s'}

def timing_statistics(values):
    """One sample per logged team step, or per flagged coordination update."""

    values = np.asarray(values, float)
    if not len(values):
        return dict(samples=0, mean_ms=np.nan, p95_ms=np.nan, max_ms=np.nan, total_s=0.)
    return dict(samples=len(values), mean_ms=float(values.mean() * 1000), p95_ms=float(np.percentile(values, 95) * 1000), max_ms=float(values.max() * 1000), total_s=float(values.sum()))

def read_batch_timings(ep, rec):
    samples = {key: np.array([]) for key in TIMING_COLUMNS}
    issues = []
    path = ep / 'batch_log.csv'
    if not path.exists():
        return samples, ['missing_batch_timing_log']
    try:
        data = pd.read_csv(path, skipinitialspace=True)
        data.columns = data.columns.str.strip()
        if data.empty or any(c not in data for c in ['step', 'time']):
            return samples, ['invalid_batch_timing_index']
        index = data[['step', 'time']].apply(pd.to_numeric, errors='coerce').to_numpy(float)

        if not np.isfinite(index).all() or not np.array_equal(index[:, 0], np.arange(len(data))) or (len(data) > 1 and np.any(np.diff(index[:, 1]) <= 0)):
            return samples, ['invalid_batch_timing_index']

        if rec.get('log_status') == 'ok':
            expected_time = np.arange(rec['log_steps']) * rec['dt_s']
            if len(data) != rec['log_steps'] or not np.allclose(index[:, 1], expected_time):
                return samples, ['batch_timing_unaligned']

        for key, column in TIMING_COLUMNS.items():
            if column not in data:
                issues.append('missing_' + column); continue
            values = pd.to_numeric(data[column], errors='coerce').to_numpy(float)

            if not np.isfinite(values).all() or np.any(values < 0):
                issues.append('invalid_' + column); continue
            
            if key == 'coordination':
                if 'coordination_updated' not in data:
                    issues.append('missing_coordination_updated'); continue
                flags = data['coordination_updated'].astype(str).str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})

                if flags.isna().any():
                    issues.append('invalid_coordination_updated'); continue
                values = values[flags.to_numpy() == 1]
            samples[key] = values
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        issues.append('unreadable_batch_timing_log')

    return samples, issues

def aggregate_timings(primary, timing_samples):

    rows = []

    for keys, group in primary.groupby(['run', 'variant', 'num_agent'], dropna=False):

        row = dict(zip(['run', 'variant', 'num_agent'], keys))

        row.update(method=group.method.iloc[0], n_episodes=len(group))

        for key in TIMING_COLUMNS:

            arrays = [timing_samples[p][key] for p in group.episode_dir]

            values = np.concatenate(arrays) if arrays else np.array([])

            row[key + '_n_episodes_with_samples'] = sum(bool(len(a)) for a in arrays)

            row.update({key + '_' + name: value for name, value in timing_statistics(values).items()})

        rows.append(row)

    return pd.DataFrame(rows)

def timing_table(timings):

    lines = ['| Method | N | Component | Episodes with samples | Samples | Mean (ms) | p95 (ms) | Max (ms) |',

             '|---|---:|---|---:|---:|---:|---:|---:|']

    for _, row in timings.iterrows():

        for key in TIMING_COLUMNS:

            lines.append(f"| {row.method} | {int(row.num_agent)} | {key} | {int(row[key + '_n_episodes_with_samples'])}/{int(row.n_episodes)} | {int(row[key + '_samples'])} | {row[key + '_mean_ms']:.3f} | {row[key + '_p95_ms']:.3f} | {row[key + '_max_ms']:.3f} |")

    return '\n'.join(lines).replace('nan', 'NA')

def connectivity_summary(groups):
    columns = ['run', 'variant', 'method', 'num_agent', 'n_episodes', 'n_valid_logs',
               'conn_violation_episode_count', 'cmr_pct_mean',
               'conn_violation_total_s_mean', 'conn_violation_total_s_std_over_seeds',
               'conn_violation_total_s_n_episodes', 'conn_violation_longest_s_mean',
               'conn_violation_longest_s_std_over_seeds', 'conn_violation_longest_s_max',
               'conn_exceedance_max_m_mean', 'conn_exceedance_global_max_m',
               'mcm_m2_mean', 'mcm_global_min_m2']
    result = groups.reindex(columns=columns).copy()
    result['conn_negative_global_max_m2'] = np.maximum(-result.mcm_global_min_m2, 0.)
    return result


def connectivity_table(data):
    def count(value):
        return str(int(value)) if pd.notna(value) else 'NA'
    lines = ['| Method | N | Valid / evaluated | Violation episodes | Mean total violation (s) | Mean longest run (s) | Global longest run (s) | Max range excess (m) | Max negative margin (m²) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, row in data.iterrows():
        lines.append(f"| {row.method} | {int(row.num_agent)} | {count(row.n_valid_logs)}/{int(row.n_episodes)} | {count(row.conn_violation_episode_count)} | {row.conn_violation_total_s_mean:.3f} | {row.conn_violation_longest_s_mean:.3f} | {row.conn_violation_longest_s_max:.3f} | {row.conn_exceedance_global_max_m:.4f} | {row.conn_negative_global_max_m2:.4f} |")
    return '\n'.join(lines).replace('nan', 'NA')


def planner_statistics(path):

    data = pd.read_csv(path)

    if data.empty: raise ValueError('Empty planner audit')

    flags = {}

    for key in ['fallback_used', 'geometry_satisfied']:
        flags[key] = data[key].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
        if flags[key].isna().any(): raise ValueError('Invalid planner audit flags')
    return {'planner_decisions': len(data), 'planner_fallback_decisions': int(flags['fallback_used'].sum()),
            'planner_geometry_violations': int((~flags['geometry_satisfied']).sum()),
            'planner_fallback_pct': 100 * flags['fallback_used'].mean(),
            'planner_geometry_violation_pct': 100 * (~flags['geometry_satisfied']).mean()}

def longest_run(mask):
    transitions = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8)))
    return int(np.max(transitions[1::2] - transitions[::2])) if len(transitions) else 0

def step_statistics(conn, obs, avoid, dt, eps=1e-3, radius=0.8):
    """The eps and any-agent reduction match main_driver.compute_cbf_violation_rates."""
    h = np.min(conn, axis=0)
    violations = h < -eps
    unsafe = np.any((obs < -eps) | (avoid < -eps), axis=0)
    excess = np.maximum(np.sqrt(np.maximum(radius * radius - h, 0)) - radius, 0)

    return {'cmr_pct': 100 * (1 - violations.mean()), 'spr_pct': 100 * (1 - unsafe.mean()),
            'mcm_m2': float(h.min()), 'conn_violation_steps': int(violations.sum()),
            'conn_violation_total_s': float(violations.sum() * dt),
            'conn_violation_longest_s': longest_run(violations) * dt,
            'conn_exceedance_max_m': float(excess[violations].max()) if violations.any() else 0.,
            'conn_negative_mean_m2': float((-h[violations]).mean()) if violations.any() else np.nan,
            'conn_margin_mean_m2': float(h.mean()), 'conn_margin_p05_m2': float(np.percentile(h, 5))}

class GoalMap:
    def __init__(self, path, resolution):
        self.path, self.resolution = str(path), resolution
        self.rgb = np.asarray(Image.open(path).convert('RGB'))
        self.goal = np.all(self.rgb == [180, 50, 200], axis=2)
        self.h, self.w = self.goal.shape
        self.free_cells = int(np.any(self.rgb != [0, 0, 0], axis=2).sum())
        self.obstacle_fraction = float(np.all(self.rgb == [0, 0, 0], axis=2).mean())
        r, c = np.where(self.goal)
        self.lo = np.column_stack([c, self.h - 1 - r]) * resolution
        self.hi = self.lo + resolution

    def inside(self, positions):
        # Match the float32, 0.5 m padded MapInfo.world_to_grid_np used by the simulator.
        pad = int(round(.5 / self.resolution))
        xy = np.asarray(positions, np.float32)
        col = np.clip((xy[..., 0] + np.float32(.5)) / self.resolution, 0, self.w + 2 * pad - 1).astype(np.int64) - pad
        row = self.h + 2 * pad - 1 - np.clip((xy[..., 1] + np.float32(.5)) / self.resolution, 0, self.h + 2 * pad - 1).astype(np.int64) - pad

        valid = (row >= 0) & (row < self.h) & (col >= 0) & (col < self.w)
        result = np.zeros(valid.shape, bool)
        result[valid] = self.goal[row[valid], col[valid]]

        return result

    def distances(self, positions):
        # Exact Euclidean distance to the union of goal cells, not to the goal centre.
        p = np.asarray(positions, float)[:, None, :]
        delta = np.maximum(np.maximum(self.lo[None] - p, p - self.hi[None]), 0)

        return np.sqrt((delta * delta).sum(axis=2)).min(axis=1)

def discover_maps(project, resolution, manifest_path=None):
    if manifest_path:
        data = json.loads(Path(manifest_path).read_text())
        return {(tag, int(idx)): GoalMap(project / path, resolution) for tag, maps in data.items() for idx, path in maps.items()}
    maps = {}

    for tag in ['i_shape', 'square']:
        for f in sorted((project / 'maps').rglob(f'{tag}/map_*.png')):
            m = re.search(r'map_(\d+)\.png$', f.name)
            if m: maps[(tag, int(m.group(1)))] = GoalMap(f, resolution)

    custom = project / 'maps' / 'custom'
    if custom.exists():
        for f in custom.glob('map_*.png'):
            maps[('custom', int(f.stem.split('_')[-1]))] = GoalMap(f, resolution)

    for f in sorted((project / 'maps' / '01_case_study').glob('*.png')):
        m = re.match(r'(\d+)_', f.name)
        if m: maps.setdefault(('custom', int(m.group(1))), GoalMap(f, resolution))

    return maps

def episode_identity(ep, root):

    for parent in [ep.parent, *ep.parents]:

        m = VARIANT_RE.match(parent.name)

        if m:

            run = parent.parent.relative_to(root).as_posix() if parent.parent.is_relative_to(root) else parent.parent.name

            return run or '.', m.group(2), int(m.group(1))

    return '.', 'unspecified', None

def evaluate_episode(summary_path, root, maps, eps, radius, a_max, w_max):
    ep = summary_path.parent
    rec = parse_summary(str(summary_path))
    if rec is None: raise ValueError(f'Unreadable summary: {summary_path}')
    run, variant, count = episode_identity(ep, root)
    match = DIR_RE.match(ep.name)

    if match:
        rec['seed'] = rec['seed'] if rec['seed'] is not None else int(match['seed'])
        rec['episode_index'] = rec['episode_index'] if rec['episode_index'] is not None else int(match['idx'])
        if rec['map_tag'] in ['', 'unknown']: rec['map_tag'] = match['map_tag']

    rec['num_agent'] = rec['num_agent'] if rec['num_agent'] is not None else count
    rec.update(run=run, variant=variant, method=METHOD_NAMES.get(variant, variant), episode_dir=ep.relative_to(root).as_posix(), primary=True)
    rec['summary_cmr_pct'] = 100 * (1 - rec['cbf_conn_violation_rate'])

    for metric in METRICS: rec.setdefault(metric, np.nan)
    rec['cmr_vs_summary_delta_pp'] = np.nan
    rec['trajectory_sha256'] = None

    issues = []
    if count is not None and count != rec['num_agent']: issues.append('num_agent_directory_mismatch')
    if rec['stop_reason'] not in KNOWN_REASONS: issues.append('unknown_stop_reason')
    files = sorted(ep.glob('agent_*_log.csv'), key=lambda f: int(f.stem.split('_')[1]))
    expected = set(range(int(rec['num_agent']))) if rec['num_agent'] is not None else set()
    found = {int(f.stem.split('_')[1]) for f in files}

    if found != expected or not files:
        rec['log_status'] = 'missing_agents'; rec['cmr_pct'] = rec['summary_cmr_pct']; rec['issues'] = ';'.join(issues + ['missing_agent_csv'])
        return rec, np.array([])

    frames = [pd.read_csv(f, skipinitialspace=True) for f in files]
    for frame in frames: frame.columns = frame.columns.str.strip()
    needed = CORE_COLS + ['a_safe', 'w_safe', 'obs_avoid', 'agent_avoid', 'agent_conn', 'solve_time']
    if any(any(c not in f for c in needed) for f in frames):
        rec['log_status'] = 'missing_columns'; rec['cmr_pct'] = rec['summary_cmr_pct']; rec['issues'] = ';'.join(issues + ['missing_log_columns'])
        return rec, np.array([])

    valid = all(len(f) == len(frames[0]) and len(f) > 0 and np.isfinite(f[needed].to_numpy()).all() and np.array_equal(f['step'], frames[0]['step']) and np.array_equal(f['time'], frames[0]['time']) for f in frames)
    time = frames[0]['time'].to_numpy(float)

    dt = float(np.median(np.diff(time))) if len(time) > 1 else np.nan
    valid = valid and np.isfinite(dt) and dt > 0 and np.allclose(np.diff(time), dt, rtol=1e-8, atol=1e-9) and np.array_equal(frames[0]['step'].to_numpy(), np.arange(len(time)))

    if not valid:
        rec['log_status'] = 'unaligned_or_invalid'; rec['cmr_pct'] = rec['summary_cmr_pct']; rec['issues'] = ';'.join(issues + ['unaligned_or_invalid_log'])

        return rec, np.array([])

    rec.update(log_status='ok', log_steps=len(time), dt_s=dt, recorded_window_s=len(time) * dt, logged_last_time_s=float(time[-1]))
    matrices = [np.stack([f[c].to_numpy(float) for f in frames]) for c in ['agent_conn', 'obs_avoid', 'agent_avoid']]
    rec.update(step_statistics(*matrices, dt, eps, radius))

    delta = rec['cmr_pct'] - rec['summary_cmr_pct']; rec['cmr_vs_summary_delta_pp'] = delta
    if np.isfinite(delta) and abs(delta) > .00011: issues.append('cmr_summary_mismatch')
    for col, key in [('obs_avoid', 'cbf_obs_violation_rate'), ('agent_avoid', 'cbf_avoid_violation_rate')]:
        rate = np.any(np.stack([f[col].to_numpy() for f in frames]) < -eps, axis=0).mean()
        if np.isfinite(rec[key]) and abs(rate - rec[key]) > 1.1e-6: issues.append(f'{key}_summary_mismatch')

    if rec['cbf_total_steps'] != len(time): issues.append('summary_step_count_mismatch')
    effort = np.concatenate([np.hypot((f['a_safe'] - f['a_nom']) / a_max, (f['w_safe'] - f['w_nom']) / w_max) for f in frames])
    rec.update(fce_mean=float(effort.mean()), fce_p95=float(np.percentile(effort, 95)), fce_max=float(effort.max()), fce_n_updates=len(effort))
    runtime = frames[0]['solve_time'].to_numpy(float)

    if any(not np.array_equal(f['solve_time'], runtime) for f in frames):
        issues.append('runtime_not_identical_across_agent_files'); runtime = np.array([])

    if len(runtime):
        rec.update(runtime_samples=len(runtime), runtime_mean_ms=float(runtime.mean() * 1000), runtime_p95_ms=float(np.percentile(runtime, 95) * 1000), runtime_max_ms=float(runtime.max() * 1000), runtime_over_100ms_pct=float((runtime > .1).mean() * 100))

    fingerprint = hashlib.sha256()
    for f in frames: fingerprint.update(f[CORE_COLS].to_numpy(np.float64).tobytes())
    rec['trajectory_sha256'] = fingerprint.hexdigest()
    goal_map = maps.get((rec['map_tag'], rec['episode_index']))
    if goal_map is None: issues.append('goal_map_unresolved')
    elif goal_map.free_cells != rec['total_gt_free_cells']: issues.append('goal_map_free_cell_count_mismatch')
    elif not goal_map.goal.any(): issues.append('goal_mask_empty')
    else:
        rec.update(map_path=goal_map.path, map_width_m=goal_map.w * goal_map.resolution, map_height_m=goal_map.h * goal_map.resolution, obstacle_area_fraction_with_boundary=goal_map.obstacle_fraction)
        xy = np.stack([f[['x', 'y']].to_numpy(float) for f in frames], axis=1)
        inside = goal_map.inside(xy); arrival = np.flatnonzero(inside.any(axis=1))
        if rec['stop_reason'] == 'goal':
            if not len(arrival): issues.append('goal_summary_without_goal_mask_arrival')
            else:
                first = int(arrival[0]); distance = goal_map.distances(xy[first])
                rec.update(first_arrival_step=first, first_arrival_logged_s=float(time[first]), goal_count_first=int(inside[first].sum()), goal_fraction_first=float(inside[first].mean()), goal_distance_mean_m=float(distance.mean()), goal_distance_max_m=float(distance.max()))
                if first != len(time) - 1: issues.append('goal_arrival_before_logged_end')
        elif len(arrival): issues.append('non_goal_summary_with_goal_mask_arrival')

    batch_path = ep / 'batch_log.csv'
    if batch_path.exists():
        batch = pd.read_csv(batch_path)
        required = ['step', 'time', 'control_compute_s', 'coordination_s', 'coordination_updated', 'routing_s']

        if any(c not in batch for c in required) or len(batch) != len(time) or not np.array_equal(batch['step'], frames[0]['step']) or not np.allclose(batch['time'], time):
            issues.append('batch_log_unaligned_or_incomplete')

        elif not np.isfinite(batch[required].to_numpy(float)).all(): issues.append('batch_log_nonfinite')

        else:
            updates = batch.loc[batch.coordination_updated.eq(1), 'coordination_s']
            rec.update(control_compute_mean_ms=batch.control_compute_s.mean() * 1000,
                       control_compute_p95_ms=batch.control_compute_s.quantile(.95) * 1000,
                       coordination_mean_ms=updates.mean() * 1000,
                       coordination_p95_ms=updates.quantile(.95) * 1000,
                       routing_mean_ms=batch.routing_s.mean() * 1000)

            active_cols = [c for c in batch if c.startswith('active_obs_')]
            if active_cols: rec['max_active_obstacle_constraints'] = batch[active_cols].max().max()
    planner_path = ep / 'planner_log.csv'

    if planner_path.exists():
        try: rec.update(planner_statistics(planner_path))
        except (ValueError, KeyError, pd.errors.ParserError): issues.append('invalid_planner_audit')
    rec['issues'] = ';'.join(issues)

    return rec, runtime

def primary_selection(df, deterministic_variants):
    df = df.copy(); df['primary'] = True
    audit = []
    for keys, group in df[df.variant.isin(deterministic_variants)].groupby(['run', 'variant', 'num_agent', 'map_tag', 'episode_index'], dropna=False):
        identical = (group.log_status.eq('ok').all() and group.trajectory_sha256.notna().all() and group.trajectory_sha256.nunique() == 1 and group.stop_reason.nunique() == 1)
        audit.append(dict(zip(['run', 'variant', 'num_agent', 'map_tag', 'episode_index'], keys), n_saved=len(group), identical_trajectory=identical, n_unique_trajectory=group.trajectory_sha256.nunique()))
        if len(group) > 1 and identical:
            retain = group.sort_values(['seed', 'episode_dir']).index[0]
            df.loc[group.index.difference([retain]), 'primary'] = False
        elif len(group) > 1:
            df.loc[group.index, 'issues'] = df.loc[group.index, 'issues'].map(lambda s: ';'.join(filter(None, [s, 'deterministic_repeats_not_identical'])))

    return df, pd.DataFrame(audit)

def aggregate(primary, raw, runtimes, root):
    rows = []
    for keys, g in primary.groupby(['run', 'variant', 'num_agent'], dropna=False):
        row = dict(zip(['run', 'variant', 'num_agent'], keys)); row['method'] = g.method.iloc[0]
        row.update(n_episodes=len(g), n_maps=g[['map_tag', 'episode_index']].drop_duplicates().shape[0], n_seeds=g.seed.nunique(), n_valid_logs=int(g.log_status.eq('ok').sum()))
        stored = raw[(raw.run == keys[0]) & (raw.variant == keys[1]) & (raw.num_agent == keys[2])]
        row['n_saved_episodes'] = len(stored); row['n_duplicate_repeats_removed'] = len(stored) - len(g)
        
        for key in ['planner_decisions', 'planner_fallback_decisions', 'planner_geometry_violations']:
            row[key + '_total'] = g[key].sum(min_count=1) if key in g else np.nan

        row['conn_violation_episode_count'] = int(g.get('conn_violation_steps', pd.Series(dtype=float)).gt(0).sum()) if g.log_status.eq('ok').all() else np.nan
        row['conn_violation_longest_s_max'] = g.conn_violation_longest_s.max()
        row['mcm_global_min_m2'] = g.mcm_m2.min()
        row['conn_exceedance_global_max_m'] = g.conn_exceedance_max_m.max()

        for reason in KNOWN_REASONS: row[reason + '_count'] = int(g.stop_reason.eq(reason).sum())
        row['other_count'] = int((~g.stop_reason.isin(KNOWN_REASONS)).sum())
        row['success_rate_pct'] = 100 * row['goal_count'] / len(g)

        lo, hi = wilson_ci(row['goal_count'], len(g)); row.update(sr_ci95_low_pct=100 * lo, sr_ci95_high_pct=100 * hi)
        for metric in METRICS:
            if metric not in g: continue
            values = g.groupby('seed', dropna=False)[metric].mean().dropna()
            row[metric + '_mean'] = float(values.mean()) if len(values) else np.nan
            row[metric + '_std_over_seeds'] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[metric + '_n_episodes'] = int(g[metric].notna().sum())

        row.update(batch_runtime_samples=0, batch_runtime_over_100ms_count=0, batch_runtime_mean_ms=np.nan, batch_runtime_p95_ms=np.nan, batch_runtime_max_ms=np.nan, batch_runtime_over_100ms_pct=np.nan)

        pooled = [runtimes.get(str(root / p), np.array([])) for p in g.episode_dir]
        pooled = np.concatenate(pooled) if pooled else np.array([])

        if len(pooled):
            row.update(batch_runtime_samples=len(pooled), batch_runtime_over_100ms_count=int((pooled > .1).sum()), batch_runtime_mean_ms=float(pooled.mean() * 1000), batch_runtime_p95_ms=float(np.percentile(pooled, 95) * 1000), batch_runtime_max_ms=float(pooled.max() * 1000), batch_runtime_over_100ms_pct=float((pooled > .1).mean() * 100))

        rows.append(row)

    return pd.DataFrame(rows)

def map_comparison(df):
    rows = []
    for keys, g in df.groupby(['run', 'variant', 'num_agent', 'map_tag', 'episode_index'], dropna=False):
        rows.append(dict(zip(['run', 'variant', 'num_agent', 'map_tag', 'episode_index'], keys), n_episodes=len(g), success_rate_pct=100 * g.stop_reason.eq('goal').mean(), cmr_pct=g.cmr_pct.mean(), spr_pct=g.spr_pct.mean(), mcm_m2=g.mcm_m2.mean(), map_path=g.get('map_path', pd.Series(dtype=str)).iloc[0] if 'map_path' in g else None, obstacle_area_fraction_with_boundary=g.get('obstacle_area_fraction_with_boundary', pd.Series(dtype=float)).mean(), goal_fraction_first=g.goal_fraction_first.mean(), goal_distance_max_m=g.goal_distance_max_m.mean()))

    by_map = pd.DataFrame(rows)
    baseline = by_map[by_map.variant.eq('team_ablation')]
    full = by_map[by_map.variant.eq('default')]
    keys = ['run', 'num_agent', 'map_tag', 'episode_index']
    paired = full.merge(baseline, on=keys, suffixes=('_full', '_frontier'))

    if not paired.empty: paired['success_delta_pp'] = paired.success_rate_pct_full - paired.success_rate_pct_frontier
    return by_map, paired

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--project_root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--map_manifest', default=None)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--radius', type=float, default=.8)
    parser.add_argument('--resolution', type=float, default=.01)
    parser.add_argument('--a_max', type=float, default=.1)
    parser.add_argument('--w_max', type=float, default=1.)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--deterministic_variants', nargs='*', default=['team_ablation', 'frontier_spread'])

    args = parser.parse_args()

    if args.a_max <= 0 or args.w_max <= 0 or args.radius <= 0 or args.resolution <= 0 or args.eps < 0 or args.workers < 1: parser.error('Invalid analysis parameters')

    root, project, out = Path(args.root).resolve(), Path(args.project_root).resolve(), Path(args.out_dir).resolve()

    paths = sorted(root.rglob('termination_summary.txt'))

    if not paths: parser.error('No termination_summary.txt found')

    maps = discover_maps(project, args.resolution, args.map_manifest)

    records, runtimes = [], {}

    def work(path): return evaluate_episode(path, root, maps, args.eps, args.radius, args.a_max, args.w_max)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        for i, (path, (rec, runtime)) in enumerate(zip(paths, executor.map(work, paths)), 1):

            records.append(rec); runtimes[str(path.parent)] = runtime

            if i % 60 == 0 or i == len(paths): print(f'[analysis] {i}/{len(paths)} episodes', flush=True)

    timing_samples = {}

    for rec in records:

        samples, timing_issues = read_batch_timings(root / rec['episode_dir'], rec)

        timing_samples[rec['episode_dir']] = samples

        rec['issues'] = ';'.join(filter(None, [rec.get('issues', ''), *timing_issues]))

        for key, values in samples.items():

            rec.update({key + '_' + name: value for name, value in timing_statistics(values).items()})

    df = pd.DataFrame(records)

    if df[['num_agent', 'seed', 'episode_index']].isna().any().any(): raise ValueError('Unresolved episode identity; no records silently dropped')

    for column in ['num_agent', 'seed', 'episode_index']: df[column] = df[column].astype(int)

    df, repeat_audit = primary_selection(df, args.deterministic_variants)

    valid = df.loc[
        df.primary & df.log_status.eq('ok') & df.conn_violation_longest_s.notna()
    ].copy()
    keys = ['run', 'variant', 'num_agent']
    max_duration = valid.groupby(keys)['conn_violation_longest_s'].transform('max')
    worst_conn = valid.loc[
        np.isclose(valid.conn_violation_longest_s, max_duration, rtol=0, atol=1e-9)
        & max_duration.gt(0),
        keys + ['method', 'map_tag', 'episode_index', 'seed', 'stop_reason',
                'conn_violation_longest_s', 'conn_violation_total_s',
                'conn_exceedance_max_m', 'episode_dir']
    ].sort_values(keys + ['map_tag', 'episode_index', 'seed'])

    groups = aggregate(df[df.primary], df, runtimes, root)

    timings = aggregate_timings(df[df.primary], timing_samples)

    groups = groups.merge(timings.drop(columns=['method', 'n_episodes']), on=['run', 'variant', 'num_agent'], validate='one_to_one')

    connectivity = connectivity_summary(groups)
    groups['conn_negative_global_max_m2'] = connectivity['conn_negative_global_max_m2'].to_numpy()
    by_map, paired = map_comparison(df[df.primary])

    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / 'per_episode.csv', index=False)

    groups.to_csv(out / 'group_summary.csv', index=False)

    timings.to_csv(out / 'timing_summary.csv', index=False)
    connectivity.to_csv(out / 'connectivity_summary.csv', index=False)

    by_map.to_csv(out / 'by_map.csv', index=False)

    paired.to_csv(out / 'paired_full_frontier.csv', index=False)

    repeat_audit.to_csv(out / 'deterministic_repeat_audit.csv', index=False)

    worst_conn.to_csv(out / 'connectivity_longest_cases.csv', index=False)
    print('\nEPISODES WITH THE LONGEST CONTINUOUS CONNECTIVITY VIOLATION')
    print(worst_conn.to_string(index=False) if not worst_conn.empty else 'No violation cases.')

    df.loc[df.issues.ne(''), ['run', 'variant', 'num_agent', 'episode_dir', 'log_status', 'issues', 'cmr_vs_summary_delta_pp']].to_csv(out / 'data_audit.csv', index=False)

    manifest = [dict(map_tag=tag, episode_index=idx, map_path=g.path, width_m=g.w * args.resolution, height_m=g.h * args.resolution, free_cells=g.free_cells, goal_cells=int(g.goal.sum()), obstacle_area_fraction_with_boundary=g.obstacle_fraction) for (tag, idx), g in sorted(maps.items())]

    pd.DataFrame(manifest).to_csv(out / 'map_manifest.csv', index=False)

    commit = subprocess.run(['git', '-C', str(project), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or 'unavailable'

    metadata = dict(source_commit=commit, parameters=vars(args), n_saved_episodes=len(df), n_primary_episodes=int(df.primary.sum()), n_issue_episodes=int(df.issues.ne('').sum()), continuous_aggregation='mean within each seed, then mean and sample std across seeds', time_definition='CSV logged time; duration is count*dt', runtime_definition='one solve_time per aligned team step, not per agent', cmr_definition='100*(1-mean(any(agent_conn < -eps)))', goal_definition='float32 padded MapInfo grid conversion; Euclidean distance to union of goal cells')

    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

    print('\nCONNECTIVITY VIOLATION DURATION AND MAGNITUDE')
    print(connectivity_table(connectivity))
    print('\nBATCH TIMING (pooled primary samples, milliseconds)')

    print(timing_table(timings))

    print(groups[['method', 'num_agent', 'goal_count', 'n_episodes', 'success_rate_pct', 'cmr_pct_mean', 'goal_fraction_first_mean']].to_string(index=False))

    print(f'[analysis] Saved results to {out}; issue episodes={metadata["n_issue_episodes"]}', flush=True)

if __name__ == '__main__': main()
