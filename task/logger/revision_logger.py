"""One batch-level record per step, alongside the unchanged per-agent CSVs."""
import csv
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path


def _plain(value):
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if hasattr(value, 'tolist'): return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)


class RevisionLogger:
    def __init__(self, out_dir, env, config):
        self.out = Path(out_dir); self.rows = []; self.planner_rows = []
        project = Path(__file__).resolve().parents[2]
        try: commit = subprocess.run(['git', '-C', str(project), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        except OSError: commit = ''
        paths = ['main_driver.py', 'task/env/cbf_env.py', 'task/env/cbf_env_cfg.py', 'task/models/hocbf.py',
                 'task/planner/frontier_planner.py', 'task/planner/frontier_spread.py', 'task/planner/unknown_target_planner.py',
                 'task/logger/revision_logger.py', 'analysis/run_frontier_spread.py']
        hashes = {p: hashlib.sha256((project / p).read_bytes()).hexdigest() for p in paths if (project / p).exists()}
        versions = {}
        for package in ['numpy', 'torch', 'cvxpy', 'cvxpylayers', 'ecos', 'scipy']:
            try: versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError: versions[package] = None
        try:
            import torch
            gpu = torch.cuda.get_device_name(env.cfg.device) if str(env.cfg.device).startswith('cuda') and torch.cuda.is_available() else None
        except (ImportError, RuntimeError, AttributeError): gpu = None
        data = {'source_commit': commit or None, 'code_sha256': hashes, 'versions': versions,
                'python': platform.python_version(), 'platform': platform.platform(), 'processor': platform.processor(), 'gpu': gpu,
                'config': _plain(config), 'effective_env_config': _plain(vars(env.cfg)),
                'planner_parameters': {k: _plain(v) for k, v in vars(env.planner).items() if isinstance(v, (str, int, float, bool)) or v is None},
                'timing_definitions': {'solve_s': 'existing batch solve path, including fallback', 'control_compute_s': 'nominal control through env.step; excludes logging/printing/figures', 'coordination_s': 'planner and MST update during env.step, excludes reset', 'routing_s': 'route_all during env.step'},
                'state_phase': 'active constraints use input info at t; event flags use post-step state t+1'}
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'run_metadata.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        initial = getattr(env, 'infos', {}).get('planner', {})
        if initial: self.planner_rows.append({'step': -1, 'time': 0., 'phase': 'reset', **initial})

    def record(self, step, env, info, next_info, solve_info, control_compute_s):
        timing = next_info.get('timing', {})
        planner = next_info.get('planner', {})
        row = {'step': step, 'time': step * env.dt, 'control_compute_s': control_compute_s,
               'solve_s': solve_info['computing_time'], 'coordination_updated': int(timing.get('coordination_updated', False)),
               'coordination_s': timing.get('coordination_s', 0.), 'routing_s': timing.get('routing_s', 0.)}
        for key in ['target_spread_m', 'target_min_sep_m', 'geometry_satisfied', 'fallback_used']:
            row[key] = planner.get(key)
        if planner and timing.get('coordination_updated', False):
            self.planner_rows.append({'step': step, 'time': (step + 1) * env.dt, 'phase': 'post_step', **planner})
        safety = info['safety']
        for i in range(env.num_agent):
            row.update({f'active_obs_{i}': min(len(safety['p_obs'][i]), env.cfg.max_obs),
                        f'active_neighbor_{i}': min(len(safety['p_agents'][i]), env.cfg.max_agents - 1),
                        f'active_conn_{i}': int(len(safety['p_c_agent'][i]) > 0),
                        f'reached_goal_{i}': int(env.is_reached_goal[i].item()),
                        f'obstacle_collision_{i}': int(env.is_collided_obstacle[i].item()),
                        f'robot_collision_{i}': int(env.is_collided_drone[i].item())})
        self.rows.append(row)

    def save(self):
        for filename, rows in [('batch_log.csv', self.rows), ('planner_log.csv', self.planner_rows)]:
            if not rows: continue
            with (self.out / filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
