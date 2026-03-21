"""
TSP instance generation with grid-cell assignment.

Each instance consists of N points uniformly sampled in [0, 1)^2.
The unit square is partitioned into a grid_size x grid_size grid of equal
square cells.  Every point is assigned to exactly one cell:

    cell_row = floor(y * grid_size)
    cell_col = floor(x * grid_size)
    cell_id  = cell_row * grid_size + cell_col   ∈ [0, grid_size^2)

The cell_ids array lets downstream code quickly retrieve which points share a
cell, which is the key ingredient for triangular attention.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from neuraltsp.config import DataConfig


def assign_cells(coords: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Map (x, y) coordinates to integer cell ids.

    Parameters
    ----------
    coords : (N, 2) float array, values in [0, 1)
    grid_size : number of cells along each axis

    Returns
    -------
    cell_ids : (N,) int array in [0, grid_size**2)
    """
    # clip to [0, grid_size - 1] to handle the rare x==1.0 or y==1.0
    col = np.clip((coords[:, 0] * grid_size).astype(int), 0, grid_size - 1)
    row = np.clip((coords[:, 1] * grid_size).astype(int), 0, grid_size - 1)
    return row * grid_size + col


def cell_to_rowcol(cell_id: int, grid_size: int) -> tuple[int, int]:
    """Convert a flat cell id to (row, col)."""
    return divmod(cell_id, grid_size)


def generate_instance(
    n_points: int,
    grid_size: int,
    rng: Generator,
) -> dict[str, np.ndarray]:
    """
    Generate a single TSP instance.

    Returns
    -------
    dict with keys:
        'coords'   : (N, 2) float32 – point coordinates in [0, 1)
        'cell_ids' : (N,)   int32   – flat grid-cell index for each point
    """
    coords = rng.random((n_points, 2)).astype(np.float32)
    cell_ids = assign_cells(coords, grid_size).astype(np.int32)
    return {"coords": coords, "cell_ids": cell_ids}


def generate_dataset(
    cfg: DataConfig,
    n_instances: int,
) -> list[dict[str, np.ndarray]]:
    """
    Generate a list of TSP instances according to *cfg*.

    Parameters
    ----------
    cfg : DataConfig
    n_instances : number of instances to generate

    Returns
    -------
    list of dicts, each as returned by generate_instance
    """
    rng = np.random.default_rng(cfg.seed)
    return [
        generate_instance(cfg.n_points, cfg.grid_size, rng)
        for _ in range(n_instances)
    ]


def instances_to_arrays(
    instances: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack a list of instances into batched arrays for efficient storage."""
    return {
        "coords": np.stack([inst["coords"] for inst in instances]),    # (B, N, 2)
        "cell_ids": np.stack([inst["cell_ids"] for inst in instances]),  # (B, N)
    }
