"""
Bidirectional autoregressive tour prediction.

Two pointers grow the tour simultaneously from a shared start city:

    left_path  = [P0, L1, L2, ...]
    right_path = [P0, R1, R2, ...]

At each step:
  1. Left pointer picks its next city from its own spatial candidates
     (computed from left's current position over the shared remaining pool).
     The other-pointer token is the right pointer's current position.
  2. Left's choice is removed from the pool.
  3. Right pointer picks its next city from its own spatial candidates
     (computed from right's current position over the now-updated pool).
     The other-pointer token is the left pointer's NEW position.
  4. Right's choice is removed from the pool.

The loop runs while |remaining| >= 2.  If exactly 1 city is left after the
loop it is the forced closing city — both pointer ends connect to it without
a model prediction.

Final tour assembly:
    tour = reverse(left_path) + right_path[1:] + [closing_city]  (if any)

This gives a closed, non-repeating tour of all N cities.

Examples
--------
N=5 (even steps):  left=[P0,L1,L2], right=[P0,R1,R2], no closing
    tour = [L2, L1, P0, R1, R2]

N=6 (odd steps):   left=[P0,L1,L2], right=[P0,R1,R2], closing=X
    tour = [L2, L1, P0, R1, R2, X]
"""

from __future__ import annotations

import numpy as np
import torch

from neuraltsp.model.candidates import build_candidate_indices
from neuraltsp.model.model import TSPTransformer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick(candidate_logits: torch.Tensor, greedy: bool) -> int:
    """Argmax or multinomial sample over candidate logits. Returns local index."""
    if greedy:
        return int(candidate_logits.argmax().item())
    probs = torch.softmax(candidate_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _pointer_step(
    model: TSPTransformer,
    current_idx: int,
    other_idx: int,
    remaining: np.ndarray,
    coords: np.ndarray,
    cell_ids: np.ndarray,
    grid_size: int,
    rng: np.random.Generator,
    device: torch.device,
    greedy: bool,
) -> tuple[int, np.ndarray]:
    """
    One pointer picks its next city.

    Returns (chosen_city_idx, updated_remaining).
    """
    candidates = build_candidate_indices(
        remaining, coords, cell_ids, current_idx, grid_size, rng
    )
    coords_t, is_other_t = TSPTransformer.build_inputs(
        current_idx, other_idx, candidates, coords, device
    )
    logits = model(coords_t, is_other_t)   # (1, 2 + K)
    chosen_local = _pick(logits[0, 2:], greedy)
    chosen_idx = int(candidates[chosen_local])
    remaining = remaining[remaining != chosen_idx]
    return chosen_idx, remaining


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_tour(
    model: TSPTransformer,
    coords: np.ndarray,       # (N, 2)
    cell_ids: np.ndarray,     # (N,)
    grid_size: int,
    rng: np.random.Generator,
    device: torch.device,
    greedy: bool = False,
) -> list[int]:
    """
    Predict a full TSP tour using bidirectional construction.

    Parameters
    ----------
    model     : TSPTransformer
    coords    : (N, 2) city coordinates
    cell_ids  : (N,)  grid-cell index per city
    grid_size : cells per axis
    rng       : numpy RNG for candidate sampling and start-city choice
    device    : torch device
    greedy    : if True use argmax; if False sample from softmax

    Returns
    -------
    tour : list of N city indices (closed tour, start city first in the
           assembled order)
    """
    N = len(coords)
    if N == 1:
        return [0]

    start_idx = int(rng.integers(N))
    left_path  = [start_idx]
    right_path = [start_idx]
    remaining  = np.array([i for i in range(N) if i != start_idx], dtype=np.int64)

    while len(remaining) >= 2:
        # --- left pointer picks ---
        left_choice, remaining = _pointer_step(
            model, left_path[-1], right_path[-1],
            remaining, coords, cell_ids, grid_size, rng, device, greedy,
        )
        left_path.append(left_choice)

        if len(remaining) == 0:
            break

        # --- right pointer picks (other = left's NEW position) ---
        right_choice, remaining = _pointer_step(
            model, right_path[-1], left_path[-1],
            remaining, coords, cell_ids, grid_size, rng, device, greedy,
        )
        right_path.append(right_choice)

    # --- assemble tour ---
    tour = list(reversed(left_path)) + right_path[1:]
    if len(remaining) == 1:
        tour.append(int(remaining[0]))

    return tour


def tour_length(tour: list[int], coords: np.ndarray) -> float:
    """Total Euclidean length of a closed tour (last city back to first)."""
    pts = coords[tour]
    diffs = np.diff(pts, axis=0)
    leg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    closing = np.sqrt(((pts[-1] - pts[0]) ** 2).sum())
    return float(leg_lengths.sum() + closing)
