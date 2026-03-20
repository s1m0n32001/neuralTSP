import math

import numpy as np
import pytest

from neuraltsp.model.candidates import build_candidate_indices


def make_grid(n_points, grid_size, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.random((n_points, 2)).astype(np.float32)
    from neuraltsp.data.generator import assign_cells
    cell_ids = assign_cells(coords, grid_size).astype(np.int32)
    return coords, cell_ids


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_candidates_are_subset_of_remaining():
    rng = np.random.default_rng(1)
    coords, cell_ids = make_grid(50, 5)
    remaining = np.arange(1, 50)          # all except point 0 (current)
    cands = build_candidate_indices(remaining, coords, cell_ids, 0, 5, rng)
    assert set(cands.tolist()).issubset(set(remaining.tolist()))


def test_current_point_not_in_candidates():
    rng = np.random.default_rng(2)
    coords, cell_ids = make_grid(30, 4)
    remaining = np.arange(1, 30)
    cands = build_candidate_indices(remaining, coords, cell_ids, 0, 4, rng)
    assert 0 not in cands


def test_no_duplicates():
    rng = np.random.default_rng(3)
    coords, cell_ids = make_grid(40, 5)
    remaining = np.arange(1, 40)
    cands = build_candidate_indices(remaining, coords, cell_ids, 0, 5, rng)
    assert len(cands) == len(set(cands.tolist()))


# ---------------------------------------------------------------------------
# Same-cell: all remaining points in the same cell must be included
# ---------------------------------------------------------------------------

def test_same_cell_all_included():
    """All remaining points in the current cell must always appear."""
    rng = np.random.default_rng(4)
    coords, cell_ids = make_grid(100, 10)

    current_idx = 0
    my_cell = int(cell_ids[current_idx])
    remaining = np.arange(1, 100)

    cands = build_candidate_indices(remaining, coords, cell_ids, current_idx, 10, rng)
    cand_set = set(cands.tolist())

    same_cell_pts = [i for i in remaining if cell_ids[i] == my_cell]
    for pt in same_cell_pts:
        assert pt in cand_set, f"Point {pt} (same cell) missing from candidates"


# ---------------------------------------------------------------------------
# Distance thinning: for d>0, at most ceil(count/2^d) points per cell
# ---------------------------------------------------------------------------

def test_distance_thinning_upper_bound():
    """
    For each foreign cell at Chebyshev distance d, at most ceil(n/2^d)
    points should be selected (n = remaining in that cell).
    """
    rng = np.random.default_rng(5)
    grid_size = 4
    coords, cell_ids = make_grid(200, grid_size)

    current_idx = 0
    my_cell = int(cell_ids[current_idx])
    cur_row, cur_col = divmod(my_cell, grid_size)

    remaining = np.arange(1, 200)
    cands = build_candidate_indices(remaining, coords, cell_ids, current_idx, grid_size, rng)
    cand_set = set(cands.tolist())

    # check per foreign cell
    from neuraltsp.model.candidates import _chebyshev
    cell_to_remaining: dict[int, list[int]] = {}
    for i in remaining:
        cid = int(cell_ids[i])
        cell_to_remaining.setdefault(cid, []).append(i)

    for cid, pts in cell_to_remaining.items():
        if cid == my_cell:
            continue
        row, col = divmod(cid, grid_size)
        d = _chebyshev(cur_row, cur_col, row, col)
        max_take = max(1, math.ceil(len(pts) / (2 ** d)))
        selected = [p for p in pts if p in cand_set]
        assert len(selected) <= max_take, (
            f"Cell {cid} (d={d}): selected {len(selected)} > max {max_take}"
        )


# ---------------------------------------------------------------------------
# At least one per non-empty cell
# ---------------------------------------------------------------------------

def test_at_least_one_per_nonempty_cell():
    rng = np.random.default_rng(6)
    coords, cell_ids = make_grid(80, 5)
    remaining = np.arange(1, 80)
    cands = build_candidate_indices(remaining, coords, cell_ids, 0, 5, rng)
    cand_set = set(cands.tolist())

    occupied_cells = set(int(cell_ids[i]) for i in remaining)
    for cid in occupied_cells:
        pts_in_cell = [i for i in remaining if cell_ids[i] == cid]
        selected = [p for p in pts_in_cell if p in cand_set]
        assert len(selected) >= 1, f"Cell {cid} has {len(pts_in_cell)} remaining but 0 selected"


# ---------------------------------------------------------------------------
# Edge: only one point remaining
# ---------------------------------------------------------------------------

def test_single_remaining():
    rng = np.random.default_rng(7)
    coords, cell_ids = make_grid(10, 3)
    remaining = np.array([5])
    cands = build_candidate_indices(remaining, coords, cell_ids, 0, 3, rng)
    assert list(cands) == [5]
