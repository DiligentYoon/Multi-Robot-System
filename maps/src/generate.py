"""Map generation CLI.

Usage (from project root):
    python -m maps.src.generate --type all --num 50
    python -m maps.src.generate --type i_shape --num 50 --seed_start 10001
    python -m maps.src.generate --type bottleneck
"""

import os
import sys
import argparse
import yaml

from .map_types.i_shape      import create_i_map
from .map_types.square       import create_square_map
from .map_types.bottleneck   import create_bottleneck_1x5_map
from .map_types.random_field import create_random_field_5x5_map
from .map_types.maze         import create_maze_5x5_map
from .utils import save_map_as_image

# ---------------------------------------------------------------------------
# Map registry
# ---------------------------------------------------------------------------
# Each entry describes one map type.
#   fn       : generator function
#   out_dir  : output directory (relative to project root)
#   width_m  : map width  in meters  (x-axis)
#   height_m : map height in meters  (y-axis)
#   random   : True  → generates --num maps with incrementing seeds
#              False → always writes a single deterministic map (overwrite)
#   seed_base: starting seed for random maps
#   filename : fixed output filename for deterministic maps
# ---------------------------------------------------------------------------
MAP_REGISTRY = {
    "i_shape": {
        "fn":       create_i_map,
        "out_dir":  "maps/i_shape",
        "width_m":  5.0,
        "height_m": 1.0,
        "random":   True,
        "seed_base": 10001,
    },
    "square": {
        "fn":       create_square_map,
        "out_dir":  "maps/square",
        "width_m":  5.0,
        "height_m": 5.0,
        "random":   True,
        "seed_base": 20001,
    },
    "bottleneck": {
        "fn":       create_bottleneck_1x5_map,
        "out_dir":  "maps/custom",
        "width_m":  5.0,
        "height_m": 1.0,
        "random":   False,
        "filename": "map_002.png",
    },
    "random_field": {
        "fn":       create_random_field_5x5_map,
        "out_dir":  "maps/custom",
        "width_m":  5.0,
        "height_m": 5.0,
        "random":   True,
        "seed_base": 30001,
    },
    "maze": {
        "fn":       create_maze_5x5_map,
        "out_dir":  "maps/custom",
        "width_m":  5.0,
        "height_m": 5.0,
        "random":   False,
        "filename": "map_004.png",
    },
}


def _load_resolution(config_path="config/config.yaml"):
    """Read resolution from config/config.yaml."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return float(cfg["env"]["map"].get("resolution", 0.01))


def _next_index(out_dir):
    """Return the next available zero-padded map index in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(out_dir)
        if f.startswith("map_") and f.endswith(".png")
    ]
    if not existing:
        return 0
    indices = []
    for name in existing:
        try:
            indices.append(int(name[4:7]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def _generate_random(cfg, res, num, seed_start):
    """Generate `num` random maps and save with sequential filenames."""
    out_dir  = cfg["out_dir"]
    width_m  = cfg["width_m"]
    height_m = cfg["height_m"]
    fn       = cfg["fn"]

    start_idx = _next_index(out_dir)

    for i in range(num):
        seed = seed_start + i
        gt   = fn(width_m, height_m, res, seed)
        filename = f"map_{start_idx + i:03d}.png"
        save_map_as_image(gt, out_dir, filename)
        if (i + 1) % 10 == 0 or (i + 1) == num:
            print(f"  [{i+1}/{num}] {filename}")


def _generate_fixed(cfg, res):
    """Generate one deterministic map and overwrite the fixed filename."""
    out_dir  = cfg["out_dir"]
    width_m  = cfg["width_m"]
    height_m = cfg["height_m"]
    fn       = cfg["fn"]
    filename = cfg["filename"]

    # Fixed maps may not accept seed — pass only accepted kwargs
    import inspect
    sig = inspect.signature(fn)
    kwargs = {}
    if "width_m"  in sig.parameters: kwargs["width_m"]  = width_m
    if "height_m" in sig.parameters: kwargs["height_m"] = height_m
    if "res"      in sig.parameters: kwargs["res"]       = res

    gt = fn(**kwargs)
    save_map_as_image(gt, out_dir, filename)
    print(f"  {filename} (overwrite)")


def main():
    parser = argparse.ArgumentParser(description="Map generation tool")
    parser.add_argument("--type", default="all",
                        choices=list(MAP_REGISTRY.keys()) + ["all"],
                        help="Map type to generate (default: all)")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of maps for random types (default: 50)")
    parser.add_argument("--seed_start", type=int, default=None,
                        help="Override starting seed for random types")
    args = parser.parse_args()

    # Resolve project root (two levels up from this file: maps/src/main.py)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(project_root)

    res = _load_resolution("config/config.yaml")
    print(f"Resolution: {res} m/cell")

    types_to_run = list(MAP_REGISTRY.keys()) if args.type == "all" else [args.type]

    for map_type in types_to_run:
        cfg = MAP_REGISTRY[map_type]
        print(f"\n--- {map_type} ---")

        if cfg["random"]:
            seed_start = args.seed_start if args.seed_start is not None else cfg["seed_base"]
            _generate_random(cfg, res, args.num, seed_start)
        else:
            _generate_fixed(cfg, res)

    print("\nDone.")


if __name__ == "__main__":
    main()
