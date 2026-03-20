"""
Visualise a single TSP instance from a dataset file.

Usage
-----
    python scripts/visualize.py data/val.npz --idx 0 --out plot.png

With no --out, the plot is shown interactively.
"""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise a TSP instance")
    p.add_argument("npz", type=Path, help="Path to .npz dataset file")
    p.add_argument("--idx", type=int, default=0, help="Instance index (default: 0)")
    p.add_argument("--out", type=Path, default=None,
                   help="Save path for the figure (PNG/PDF). If omitted, show interactively.")
    p.add_argument("--highlight_cell", type=int, default=None,
                   help="Highlight a specific cell id (optional)")
    return p.parse_args()


def plot_instance(
    coords: np.ndarray,    # (N, 2)
    cell_ids: np.ndarray,  # (N,)
    grid_size: int,
    highlight_cell: int | None = None,
    title: str = "",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))

    # --- draw grid ---
    step = 1.0 / grid_size
    for i in range(grid_size + 1):
        ax.axhline(i * step, color="lightgray", linewidth=0.8, zorder=0)
        ax.axvline(i * step, color="lightgray", linewidth=0.8, zorder=0)

    # --- optionally shade a cell ---
    if highlight_cell is not None:
        row, col = divmod(highlight_cell, grid_size)
        rect = patches.Rectangle(
            (col * step, row * step), step, step,
            linewidth=1.5, edgecolor="royalblue",
            facecolor="royalblue", alpha=0.15, zorder=1,
        )
        ax.add_patch(rect)

    # --- colour points by cell ---
    n_cells = grid_size * grid_size
    cmap = plt.cm.get_cmap("tab20", n_cells)
    colors = [cmap(int(c)) for c in cell_ids]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=30, zorder=2, linewidths=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title or f"TSP instance  (N={len(coords)}, grid={grid_size}×{grid_size})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)

    coords = data["coords"][args.idx]      # (N, 2)
    cell_ids = data["cell_ids"][args.idx]  # (N,)
    grid_size = int(data["grid_size"]) if "grid_size" in data else 10

    fig = plot_instance(
        coords, cell_ids, grid_size,
        highlight_cell=args.highlight_cell,
        title=f"{args.npz.name}  [idx={args.idx}]",
    )

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
