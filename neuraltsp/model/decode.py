"""
Autoregressive tour prediction.

`predict_tour` runs the model step-by-step from a random starting city,
always sampling (or greedily picking) the next city from the candidate set
built by `build_candidate_indices`, until all cities have been visited.

Sequence layout at each step:  [P, A, Q1, ..., QK]
  P  = current city   (position 0, always masked to -inf)
  A  = start city     (position 1, always masked to -inf during decoding)
  Qi = candidates     (positions 2 …, softmax is taken over these)
"""

from __future__ import annotations

import numpy as np
import torch

from neuraltsp.model.candidates import build_candidate_indices
from neuraltsp.model.model import TSPTransformer


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
    Predict a full TSP tour for a single instance.

    Parameters
    ----------
    model     : TSPTransformer in eval or train mode (gradients respected)
    coords    : (N, 2) city coordinates
    cell_ids  : (N,)  grid cell index per city
    grid_size : cells per axis
    rng       : numpy RNG used for candidate sampling (and start-city choice)
    device    : torch device
    greedy    : if True use argmax; if False sample from the softmax distribution

    Returns
    -------
    tour : list of N city indices in visit order (open tour, start city first)
    """
    N = len(coords)

    # --- pick a random start city ---
    start_idx = int(rng.integers(N))
    current_idx = start_idx
    tour = [start_idx]

    # remaining: all cities except the start (numpy array for fast ops)
    remaining = np.array([i for i in range(N) if i != start_idx], dtype=np.int64)

    while len(remaining) > 0:
        # --- build spatial candidates from unvisited cities ---
        candidates = build_candidate_indices(
            remaining, coords, cell_ids, current_idx, grid_size, rng
        )
        # candidates is a 1-D int64 array of length K

        # --- build model input tensors ---
        # sequence: [current (P), start (A), *candidates]
        coords_t, is_start_t = TSPTransformer.build_inputs(
            current_idx, start_idx, candidates, coords, device
        )

        # --- forward pass ---
        logits = model(coords_t, is_start_t)   # (1, 2+K)

        # mask P (pos 0) and A (pos 1) — they are never next-city candidates
        logits[0, 0] = float("-inf")
        logits[0, 1] = float("-inf")

        candidate_logits = logits[0, 2:]       # (K,)

        # --- choose next city ---
        if greedy:
            chosen_local = int(candidate_logits.argmax().item())
        else:
            probs = torch.softmax(candidate_logits, dim=-1)
            chosen_local = int(torch.multinomial(probs, num_samples=1).item())

        chosen_idx = int(candidates[chosen_local])

        # --- update state ---
        tour.append(chosen_idx)
        current_idx = chosen_idx
        remaining = remaining[remaining != chosen_idx]

    return tour


def tour_length(tour: list[int], coords: np.ndarray) -> float:
    """Total Euclidean length of a tour (closed: last city back to first)."""
    pts = coords[tour]
    diffs = np.diff(pts, axis=0)
    leg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    closing = np.sqrt(((pts[-1] - pts[0]) ** 2).sum())
    return float(leg_lengths.sum() + closing)
