"""Run a smoke test, two-map pilot, or the existing 12-map frontier comparison."""
import argparse
import copy
import json
import os
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
METHODS = {'full': ('default', 'target_unknown'), 'frontier': ('team_ablation', 'target_frontier'),
           'frontier_spread': ('frontier_spread', 'target_frontier_spread')}


def resolve_map(tag, index):
    filename = f'map_{index:03d}.png'
    choices = [PROJECT / 'maps' / tag / filename, PROJECT / 'maps/02_quantitative_study' / tag / filename]
    if tag == 'custom': choices.extend(sorted((PROJECT / 'maps/01_case_study').glob(f'{index:02d}_*.png')))
    for path in choices:
        if path.is_file(): return path
    raise FileNotFoundError(f'Cannot resolve {tag}/{filename}; checked existing map directories.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke', 'pilot', 'full'], default='pilot')
    parser.add_argument('--methods', nargs='+', choices=list(METHODS), default=['frontier_spread'])
    parser.add_argument('--agents', nargs='+', type=int)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--config', type=Path, default=PROJECT / 'config/config.yaml')
    parser.add_argument('--out_dir', type=Path, default=PROJECT / 'results/frontier_spread_pilot')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--max_spread_m', type=float, default=.8)
    parser.add_argument('--min_pair_m', type=float, default=.08)
    parser.add_argument('--max_anchor_trials', type=int, default=100)
    args = parser.parse_args()
    agents = args.agents or ([3, 5, 7] if args.stage == 'full' else [5])
    steps = args.steps if args.steps is not None else (120 if args.stage == 'smoke' else 10000)
    if steps < 1 or min(agents) < 1 or min(args.seeds) < 0: parser.error('Use positive steps/agent counts and nonnegative seeds.')
    if args.max_spread_m <= 0 or args.min_pair_m < 0 or args.max_anchor_trials < 1:
        parser.error('Use positive spread/trials and nonnegative separation.')
    maps = [('i_shape', 3), ('square', 34)]
    if args.stage == 'full':
        maps = [('i_shape', i) for i in [3, 15, 44, 68, 84, 89]] + [('square', i) for i in [34, 49, 69]] + [('custom', i) for i in [1, 2, 3]]
    cfg = yaml.safe_load(args.config.resolve().read_text())
    out = args.out_dir.resolve()
    cases = []
    for method in dict.fromkeys(args.methods):
        variant, mode = METHODS[method]
        for n in dict.fromkeys(agents):
            for seed in dict.fromkeys(args.seeds):
                for tag, idx in maps:
                    dest = out / f'agent_{n}_{variant}' / f'{tag}_seed_{seed}_{idx:03d}'
                    if dest.exists(): raise FileExistsError(f'Refusing to overwrite existing episode: {dest}')
                    cases.append({'method': method, 'mode': mode, 'variant': variant, 'num_agent': n,
                                  'seed': seed, 'map_tag': tag, 'episode_index': idx, 'map_path': str(resolve_map(tag, idx)), 'output': str(dest)})
    # Simulator imports remain deferred so --help and map validation need no torch.
    from main_driver import run_single_simulation
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f'frontier_{args.stage}_manifest.json'
    if manifest.exists(): raise FileExistsError(f'Refusing to overwrite run manifest: {manifest}')
    manifest.write_text(json.dumps({'stage': args.stage, 'steps': steps, 'max_spread_m': args.max_spread_m,
        'min_pair_m': args.min_pair_m, 'max_anchor_trials': args.max_anchor_trials, 'cases': cases}, indent=2) + '\n')
    os.chdir(PROJECT)
    for case in cases:
        run_cfg = copy.deepcopy(cfg)
        run_cfg['env'].update(num_agent=case['num_agent'], assign_mode=case['mode'],
            frontier_max_spread_m=args.max_spread_m, frontier_min_pair_m=args.min_pair_m,
            frontier_max_anchor_trials=args.max_anchor_trials)
        if args.device: run_cfg['env']['device'] = args.device
        run_single_simulation(run_cfg, case['seed'], case['episode_index'], case['map_tag'], steps,
            root_out_dir=str(Path(case['output']).parent), frame_interval=None, gif_interval=None,
            map_filepath=case['map_path'], raise_on_error=True)
        if not (Path(case['output']) / 'batch_log.csv').is_file():
            raise RuntimeError(f'No batch log produced; inspect simulator error above: {case["output"]}')
    print(f'Completed {len(cases)} episodes. Summarize with:')
    print(f'python -m analysis.revision_summary --root {out} --out_dir {out}/evaluation')


if __name__ == '__main__': main()
