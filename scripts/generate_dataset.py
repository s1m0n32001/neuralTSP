"""
Generate TSP datasets and save them as .npz files.

Usage examples
--------------
# Generate train/val/test splits with default settings (100 pts, 10x10 grid):
    python scripts/generate_dataset.py --out data/

# Custom settings:
    python scripts/generate_dataset.py \\
        --out data/ \\
        --n_points 50 \\
        --grid_size 5 \\
        --train 100000 \\
        --val   10000  \\
        --test  10000  \\
        --seed  42

Output
------
    data/train.npz
    data/val.npz
    data/test.npz

Each .npz contains:
    coords    (n_instances, n_points, 2)  float32
    cell_ids  (n_instances, n_points)     int32
    grid_size scalar int
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from neuraltsp.config import DataConfig
from neuraltsp.data.generator import generate_instance, instances_to_arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate TSP datasets")
    p.add_argument("--out", type=Path, default=Path("data"),
                   help="Output directory (default: data/)")
    p.add_argument("--n_points", type=int, default=100,
                   help="Cities per TSP instance (default: 100)")
    p.add_argument("--grid_size", type=int, default=10,
                   help="Grid cells along each axis (default: 10)")
    p.add_argument("--train", type=int, default=100_000,
                   help="Number of training instances (default: 100000)")
    p.add_argument("--val", type=int, default=10_000,
                   help="Number of validation instances (default: 10000)")
    p.add_argument("--test", type=int, default=10_000,
                   help="Number of test instances (default: 10000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed (default: 42)")
    return p.parse_args()


def generate_split(
    name: str,
    n_instances: int,
    cfg: DataConfig,
    out_dir: Path,
    rng: np.random.Generator,
) -> None:
    out_path = out_dir / f"{name}.npz"
    instances = [
        generate_instance(cfg.n_points, cfg.grid_size, rng)
        for _ in tqdm(range(n_instances), desc=name, unit="inst")
    ]
    arrays = instances_to_arrays(instances)
    np.savez_compressed(
        out_path,
        coords=arrays["coords"],
        cell_ids=arrays["cell_ids"],
        grid_size=np.int32(cfg.grid_size),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  saved {out_path}  ({size_mb:.1f} MB)")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(
        n_points=args.n_points,
        grid_size=args.grid_size,
        seed=args.seed,
    )

    print(f"Config: {cfg}")
    print(f"Output: {args.out.resolve()}\n")

    # Use a single global RNG so splits are reproducible and non-overlapping.
    rng = np.random.default_rng(cfg.seed)

    splits = [("train", args.train), ("val", args.val), ("test", args.test)]
    for name, n in splits:
        generate_split(name, n, cfg, args.out, rng)

    print("\nDone.")


if __name__ == "__main__":
    main()
