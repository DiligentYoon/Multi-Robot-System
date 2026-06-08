# A Hierarchical Connectivity-Preserving HOCBF Framework for Safety-Critical Cooperative Navigation in Obstacle-Dense Corridors

Multiple robots explore and navigate an unknown environment using boundary-based team-level planning, with a High Order Control Barrier Functions (HOCBFs)-based safety filter that enforces collision avoidance and connectivity maintenance in real time.

## Demo

<img src="results/demo/demo_video.gif" width="100%">

---

## Setup

```bash
# Standard (CUDA 12.1)
conda env create -f safe_control.yml
conda activate safe_control

# RTX 50 series GPU
conda env create -f safe_control_rtx50.yml
conda activate safe_control
```

---

## Running

### Single simulation

```bash
python main_driver.py
```

Edit `config/config.yaml` to choose a map, set the number of agents, or tune safety parameters, then run the command above.

### Batch validation across map sets

Call `run_validation()` inside `main_driver.py` (uncomment or set the entry point). Results for each episode are saved under `results/<run_tag>/`.

### Generate maps

```bash
python -m maps.src.main --type all --num 50          # all 5 types, 50 each
python -m maps.src.main --type i_shape --num 50
python -m maps.src.main --type square  --num 50
python -m maps.src.main --type random_field --num 10
python -m maps.src.main --type bottleneck            # fixed map, overwrite
python -m maps.src.main --type maze                  # fixed map, overwrite
```

### Post-hoc analysis

```bash
python analysis/quantitative_summary.py   # aggregate metrics across episodes
python analysis/plot_timeseries.py        # per-agent CBF & control time-series
```

---

## Key Config Options (`config/config.yaml`)

| Parameter | What it controls |
|-----------|-----------------|
| `env.num_agent` | Number of robots |
| `env.map.map_filepath` | Map PNG to load |
| `env.graph_mode` | Connectivity graph: `"mst"` (default) or `"nn_tree"` |
| `env.assign_mode` | Frontier assignment: `"target_unknown"` or `"target_frontier"` (ablation) |
| `model.safety.d_obs` | Obstacle safety margin (m) |
| `model.safety.d_safe` | Inter-agent safety margin (m) |
| `model.safety.d_max` | Max allowed connectivity edge length (m) |
| `visualization.show_trajectory` | Draw path history in frames |

---

## Outputs

Each episode writes to `results/<run_tag>/<map_tag>_NNN/`:

```
agent_0_log.csv          # step, time, x, y, nominal/safe control, CBF values
agent_1_log.csv
...
frame_0000.png           # GT map (left) + belief map (right) snapshots
<map_tag>_NNN.gif        # full episode animation
termination_summary.txt  # termination reason, coverage %, CBF violation rates
```

Termination reasons: `goal` / `obstacle_collision` / `robot_collision` / `frozen` / `timeout`

---

## Project Structure

```
Multi_Robot_System/
├── config/config.yaml
├── main_driver.py              # entry point
├── visualization.py
├── task/
│   ├── env/cbf_env.py          # simulation environment
│   ├── models/hocbf.py         # HOCBF safety layer (QP via CvxpyLayers for parallel solving)
│   ├── graph/graph.py          # connectivity graph (MST / NN-tree)
│   ├── planner/                # boundary detection + A* routing
│   ├── logger/sim_logger.py
│   └── utils/
├── maps/                       # map datasets + generation scripts
├── results/
│   └── demo/demo_video.mp4
└── analysis/
```

---

## Citation

Update soon..
