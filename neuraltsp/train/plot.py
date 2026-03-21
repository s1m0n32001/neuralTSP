"""
Tour plotting utilities for training logs.

`plot_tour` draws a TSP instance together with the predicted tour:
  - faint grid overlay
  - cities coloured by grid cell
  - tour edges drawn as a colour-gradient path (blue → red = start → end)
  - start city marked with a white star
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_tour(
    coords: np.ndarray,    # (N, 2)
    cell_ids: np.ndarray,  # (N,)
    tour: list[int],       # ordered city indices
    grid_size: int,
    title: str = "",
    path: Path | str | None = None,
) -> plt.Figure:
    """
    Plot a predicted TSP tour.

    Parameters
    ----------
    coords    : (N, 2) city coordinates in [0, 1)
    cell_ids  : (N,)   grid-cell index per city
    tour      : visit order (length N, open tour — closing edge drawn automatically)
    grid_size : cells per axis
    title     : figure title
    path      : if given, save the figure here; otherwise return without saving

    Returns
    -------
    fig : matplotlib Figure
    """
    N = len(tour)
    fig, ax = plt.subplots(figsize=(7, 7))

    # --- grid ---
    step = 1.0 / grid_size
    for k in range(grid_size + 1):
        ax.axhline(k * step, color="#e0e0e0", linewidth=0.7, zorder=0)
        ax.axvline(k * step, color="#e0e0e0", linewidth=0.7, zorder=0)

    # --- tour edges with colour gradient (blue=start → red=end) ---
    cmap_edge = plt.cm.coolwarm
    closed_tour = tour + [tour[0]]   # close the loop
    for step_idx in range(N):
        a = coords[closed_tour[step_idx]]
        b = coords[closed_tour[step_idx + 1]]
        color = cmap_edge(step_idx / N)
        ax.annotate(
            "",
            xy=b, xytext=a,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=0.9,
                mutation_scale=8,
            ),
            zorder=1,
        )

    # --- cities coloured by cell ---
    n_cells = grid_size * grid_size
    cmap_city = plt.cm.get_cmap("tab20", n_cells)
    city_colors = [cmap_city(int(c)) for c in cell_ids]
    ax.scatter(coords[:, 0], coords[:, 1],
               c=city_colors, s=25, zorder=2, linewidths=0)

    # --- mark start city ---
    start = coords[tour[0]]
    ax.scatter(*start, s=120, c="white", edgecolors="black",
               linewidths=1.2, marker="*", zorder=3)

    length = _tour_length(coords, tour)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title or f"N={N}  len={length:.4f}  grid={grid_size}×{grid_size}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def _tour_length(coords: np.ndarray, tour: list[int]) -> float:
    pts = coords[tour]
    diffs = np.diff(pts, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum()
                 + np.sqrt(((pts[-1] - pts[0]) ** 2).sum()))
