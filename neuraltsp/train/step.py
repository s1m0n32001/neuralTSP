"""
Single training step: simulated-annealing style update.

Algorithm
---------
1.  Obtain the current tour (from cache or fresh bidirectional prediction).
2.  Apply a random 2-opt move: pick positions i < j and reverse new_tour[i:j+1].
3.  Accept if the new tour is shorter; otherwise accept with probability
    exp(-(E_new - E_old) / T).
4.  If accepted, compute the loss for the two pointer predictions that the
    2-opt move affects, and backprop.

Loss after a 2-opt move at (i, j)
-----------------------------------
The reversed segment is new_tour[i : j+1].  In the bidirectional construction,
the natural training signal is to place the two pointers at the segment
boundaries and ask the model to predict inward:

    left pointer  at new_tour[i-1]  →  should predict new_tour[i]
    right pointer at new_tour[j+1]  →  should predict new_tour[j]

    remaining for both = new_tour[i : j+1]  (the reversed segment)
    left's "other"  = new_tour[j+1]  (right pointer's position)
    right's "other" = new_tour[i-1]  (left pointer's position)

Each term is normalised by log(K) where K = number of candidates seen by that
pointer.  A term is skipped if the correct city is absent from the candidate
set, if K == 1, or if the segment touches a tour boundary (i == 0 or
j == N-1, so one pointer has no boundary city to condition on).

Example: ABCDEFGH, 2-opt at i=2, j=5 → ABFEDCGH
    left  : current=B, other=G, correct=F, remaining=[F,E,D,C]
    right : current=G, other=B, correct=C, remaining=[F,E,D,C]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from neuraltsp.model.candidates import build_candidate_indices
from neuraltsp.model.decode import predict_tour, tour_length
from neuraltsp.model.model import TSPTransformer


@dataclass
class StepResult:
    accepted: bool
    E_old: float
    E_new: float
    # Current tour after the step: new_tour if accepted, original tour if not.
    # The caller should write this back to the PathCache.
    tour: list[int]
    loss: float | None = None       # None if not accepted or all terms skipped
    n_loss_terms: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pointer_loss(
    model: TSPTransformer,
    current_idx: int,       # this pointer's current city
    other_idx: int,         # the other pointer's current city
    correct_next: int,      # city this pointer should predict
    remaining: np.ndarray,  # shared remaining pool (both pointers see this)
    coords: np.ndarray,
    cell_ids: np.ndarray,
    grid_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Cross-entropy loss for one pointer prediction, normalised by log(K).
    Returns None if the correct city is absent from candidates or K == 1.
    """
    candidates = build_candidate_indices(
        remaining, coords, cell_ids, current_idx, grid_size, rng
    )
    cand_list = candidates.tolist()

    if correct_next not in cand_list:
        return None

    K = len(cand_list)
    if K == 1:
        return None

    correct_local = cand_list.index(correct_next)

    coords_t, is_other_t = TSPTransformer.build_inputs(
        current_idx, other_idx, candidates, coords, device
    )
    logits = model(coords_t, is_other_t)   # (1, 2 + K)

    candidate_logits = logits[:, 2:]       # (1, K)
    target = torch.tensor([correct_local], dtype=torch.long, device=device)

    ce = F.cross_entropy(candidate_logits, target)
    return ce / math.log(K)


def _swap_losses(
    model: TSPTransformer,
    new_tour: list[int],
    i: int,
    j: int,
    coords: np.ndarray,
    cell_ids: np.ndarray,
    grid_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Compute up to two pointer-prediction loss terms for the 2-opt move at (i, j).

    Returns an empty list if the segment touches the tour boundary (i == 0 or
    j == N-1), since there would be no boundary city to serve as pointer context.
    """
    N = len(new_tour)
    if i == 0 or j == N - 1:
        return []

    remaining = np.array(new_tour[i:j + 1], dtype=np.int64)
    left_current  = new_tour[i - 1]   # B
    right_current = new_tour[j + 1]   # G
    left_correct  = new_tour[i]        # F  (first city of reversed segment)
    right_correct = new_tour[j]        # C  (last city of reversed segment)

    loss_terms: list[torch.Tensor] = []

    # left pointer: other = right's position
    term = _pointer_loss(
        model, left_current, right_current, left_correct,
        remaining, coords, cell_ids, grid_size, rng, device,
    )
    if term is not None:
        loss_terms.append(term)

    # right pointer: other = left's position
    term = _pointer_loss(
        model, right_current, left_current, right_correct,
        remaining, coords, cell_ids, grid_size, rng, device,
    )
    if term is not None:
        loss_terms.append(term)

    return loss_terms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def training_step(
    model: TSPTransformer,
    optimizer: torch.optim.Optimizer,
    coords: np.ndarray,       # (N, 2)
    cell_ids: np.ndarray,     # (N,)
    grid_size: int,
    rng: np.random.Generator,
    device: torch.device,
    temperature: float,
    cached_tour: list[int] | None = None,
) -> StepResult:
    """
    One SA training iteration on a single TSP instance.

    Parameters
    ----------
    model        : TSPTransformer
    optimizer    : torch optimiser
    coords       : (N, 2) city coordinates
    cell_ids     : (N,)   grid-cell index per city
    grid_size    : cells per axis
    rng          : numpy RNG
    device       : torch device
    temperature  : SA temperature T ≥ 0
    cached_tour  : if given, skip autoregressive prediction and use this tour
    """
    N = len(coords)

    # ------------------------------------------------------------------ #
    # 1. Obtain current tour                                              #
    # ------------------------------------------------------------------ #
    model.eval()
    if cached_tour is not None:
        tour = cached_tour
    else:
        with torch.no_grad():
            tour = predict_tour(model, coords, cell_ids, grid_size, rng, device)

    E_old = tour_length(tour, coords)

    if N < 4:
        # Too few cities for a meaningful 2-opt with interior boundaries.
        return StepResult(accepted=False, E_old=E_old, E_new=E_old, tour=tour)

    # ------------------------------------------------------------------ #
    # 2. 2-opt move                                                       #
    # ------------------------------------------------------------------ #
    # Sample from [1, N-1) so the reversed segment always has a boundary
    # city on each side (new_tour[i-1] and new_tour[j+1] always exist).
    # This guarantees _swap_losses can compute both pointer losses.
    i, j = sorted(rng.choice(np.arange(1, N - 1), size=2, replace=False).tolist())
    new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]

    E_new = tour_length(new_tour, coords)

    # ------------------------------------------------------------------ #
    # 3. Accept / reject                                                  #
    # ------------------------------------------------------------------ #
    delta = E_new - E_old
    if delta < 0:
        accepted = True
    elif temperature > 0:
        accepted = bool(rng.random() < math.exp(-delta / temperature))
    else:
        accepted = False

    if not accepted:
        return StepResult(accepted=False, E_old=E_old, E_new=E_new, tour=tour)

    # ------------------------------------------------------------------ #
    # 4. Compute loss and backprop                                        #
    # ------------------------------------------------------------------ #
    model.train()

    loss_terms = _swap_losses(
        model, new_tour, i, j, coords, cell_ids, grid_size, rng, device
    )

    if not loss_terms:
        return StepResult(accepted=True, E_old=E_old, E_new=E_new,
                          tour=new_tour, n_loss_terms=0)

    total_loss = sum(loss_terms)  # type: ignore[arg-type]

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return StepResult(
        accepted=True,
        E_old=E_old,
        E_new=E_new,
        tour=new_tour,
        loss=total_loss.item(),
        n_loss_terms=len(loss_terms),
    )
