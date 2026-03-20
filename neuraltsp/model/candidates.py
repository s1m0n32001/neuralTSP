"""
Spatial candidate selection for TSP autoregressive decoding.

At each step the current city is P.  Candidate cities for the next hop are
drawn from all cells using a distance-based thinning scheme:

    Chebyshev distance d from P's cell → keep ceil(count / 2**d) points,
    but always at least 1 from every non-empty cell.

d=0 (same cell)  → all remaining points
d=1 (adjacent)   → ceil(n/2)
d=2              → ceil(n/4)
…

The selection within each cell (for d > 0) is a random draw without
replacement.
"""

from __future__ import annotations

import math

import numpy as np


def _chebyshev(r1: int, c1: int, r2: int, c2: int) -> int:
    return max(abs(r1 - r2), abs(c1 - c2))


def build_candidate_indices(
    remaining: np.ndarray,      # 1-D array of remaining (unvisited) point indices
    coords: np.ndarray,         # (N, 2) full coordinate array (all points, including visited)
    cell_ids: np.ndarray,       # (N,)  full cell-id array
    current_idx: int,           # index of current point P in [0, N)
    grid_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return an array of candidate point indices for the next step.

    Parameters
    ----------
    remaining   : indices of unvisited points (P itself is excluded by the caller)
    coords      : (N, 2) coordinates of all points
    cell_ids    : (N,)   cell ids of all points
    current_idx : index of current city P
    grid_size   : G (cells per axis)
    rng         : numpy RNG for random within-cell sampling

    Returns
    -------
    candidates : 1-D int array of selected candidate indices (unordered)
    """
    current_cell = int(cell_ids[current_idx])
    cur_row, cur_col = divmod(current_cell, grid_size)

    # group remaining points by cell
    cell_to_pts: dict[int, list[int]] = {}
    for idx in remaining:
        cid = int(cell_ids[idx])
        cell_to_pts.setdefault(cid, []).append(int(idx))

    candidates: list[int] = []

    for cid, pts in cell_to_pts.items():
        row, col = divmod(cid, grid_size)
        d = _chebyshev(cur_row, cur_col, row, col)

        if d == 0:
            # same cell – take all
            candidates.extend(pts)
        else:
            # take ceil(n / 2**d), minimum 1
            n_take = max(1, math.ceil(len(pts) / (2 ** d)))
            perm = rng.permutation(len(pts))[:n_take]
            candidates.extend(pts[i] for i in perm)

    return np.array(candidates, dtype=np.int64)
