"""
Single training step: simulated-annealing style update.

Algorithm
---------
1.  Predict a full tour with the current model (no gradient).
2.  Randomly swap two cities at positions i < j in the tour.
3.  Accept the swap if it shortens the tour, otherwise accept with
    probability exp(-(E_new - E_old) / T).
4.  If accepted, compute up to four cross-entropy loss terms and backprop.

The four loss terms (for swapped positions i and j)
-----------------------------------------------------
For each position p ∈ {i, j} compute both a forward and a reverse term:

  Forward at p  — predict new_tour[p] given the prefix new_tour[:p]:
      current = new_tour[p-1],  start = new_tour[0]
      remaining = new_tour[p:]
      (skipped if p == 0)

  Reverse at p  — predict new_tour[p] given the suffix new_tour[p+1:] read
                  right-to-left (i.e. traverse the tour backwards from the end):
      current = new_tour[p+1],  start = new_tour[-1]
      remaining = new_tour[:p+1]   (cities not yet visited in reverse)
      (skipped if p == N-1)

Example: path ABCDEFGH, 2-opt at i=2, j=5 → AB + reverse(CDEF) + GH = ABFEDCGH
  1. forward  i=2 : current=B, start=A, correct=F, remaining=[F,E,D,C,G,H]
  2. forward  j=5 : current=D, start=A, correct=C, remaining=[C,G,H]
  3. reverse  i=2 : current=E, start=H, correct=F, remaining=[F,B,A]
  4. reverse  j=5 : current=G, start=H, correct=C, remaining=[C,D,E,F,B,A]

Each term is normalised by log(K) (K = number of candidates) so that
terms with different candidate set sizes are on a comparable scale.
A term is skipped entirely if the correct city is absent from the
candidate set, or if K == 1 (trivially correct, zero information).
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
    loss: float | None = None          # None if no loss term was computed
    n_loss_terms: int = 0              # 0 if not accepted or all terms skipped


def _loss_at_step(
    model: TSPTransformer,
    new_tour: list[int],
    p: int,                            # position in tour we're training on
    coords: np.ndarray,
    cell_ids: np.ndarray,
    grid_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Compute the normalised cross-entropy loss for predicting new_tour[p]
    given that we have visited new_tour[:p].

    Returns None if the correct next city is not in the candidate set
    (in which case the caller skips this term).
    """
    current_idx = new_tour[p - 1]
    correct_next = new_tour[p]
    start_idx = new_tour[0]

    remaining = np.array(new_tour[p:], dtype=np.int64)

    candidates = build_candidate_indices(
        remaining, coords, cell_ids, current_idx, grid_size, rng
    )
    cand_list = candidates.tolist()

    if correct_next not in cand_list:
        return None

    K = len(cand_list)
    if K == 1:
        # Only one choice; loss is trivially 0, skip.
        return None

    correct_local = cand_list.index(correct_next)

    coords_t, is_other_t = TSPTransformer.build_inputs(
        current_idx, start_idx, candidates, coords, device
    )
    logits = model(coords_t, is_other_t)   # (1, 2 + K)

    # candidate logits are at positions [2, 2+K); slice avoids in-place masking
    candidate_logits = logits[:, 2:]       # (1, K)
    target = torch.tensor([correct_local], dtype=torch.long, device=device)

    ce = F.cross_entropy(candidate_logits, target)
    return ce / math.log(K)


def _reverse_loss_at_step(
    model: TSPTransformer,
    new_tour: list[int],
    p: int,                            # position in the forward tour
    coords: np.ndarray,
    cell_ids: np.ndarray,
    grid_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Compute the normalised cross-entropy loss for predicting new_tour[p]
    when traversing the tour in reverse (right to left).

    The reverse tour starts at new_tour[-1] and proceeds backwards.
    At the step corresponding to position p:
      - already visited (in reverse): new_tour[p+1 :]
      - current city : new_tour[p+1]
      - correct next : new_tour[p]
      - remaining    : new_tour[:p+1]  (not yet visited in reverse)
      - start of reverse tour : new_tour[-1]

    Returns None if p == N-1 (nothing after p to condition on),
    or if the correct city is absent from the candidate set, or K == 1.
    """
    N = len(new_tour)
    if p == N - 1:
        return None

    current_idx = new_tour[p + 1]
    correct_next = new_tour[p]
    start_idx = new_tour[-1]

    remaining = np.array(new_tour[: p + 1], dtype=np.int64)

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
        current_idx, start_idx, candidates, coords, device
    )
    logits = model(coords_t, is_other_t)   # (1, 2 + K)

    candidate_logits = logits[:, 2:]       # (1, K)
    target = torch.tensor([correct_local], dtype=torch.long, device=device)

    ce = F.cross_entropy(candidate_logits, target)
    return ce / math.log(K)


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
    One training iteration on a single TSP instance.

    Parameters
    ----------
    model       : TSPTransformer
    optimizer   : torch optimiser (zero_grad / step called here)
    coords      : (N, 2) city coordinates
    cell_ids    : (N,)   grid-cell index per city
    grid_size   : cells per axis
    rng         : numpy RNG (controls tour prediction sampling + candidate sampling)
    device      : torch device
    temperature : SA temperature T ≥ 0; if 0 only improvements are accepted
    cached_tour : if provided, skip autoregressive prediction and use this tour
                  as the starting point for the SA move

    Returns
    -------
    StepResult whose `.tour` field holds the current tour after the step
    (new_tour if the swap was accepted, original tour otherwise).
    The caller should write this back to the PathCache.
    """
    N = len(coords)

    # ------------------------------------------------------------------ #
    # 1. Obtain current tour — from cache or fresh prediction             #
    # ------------------------------------------------------------------ #
    model.eval()
    if cached_tour is not None:
        tour = cached_tour
    else:
        with torch.no_grad():
            tour = predict_tour(model, coords, cell_ids, grid_size, rng, device)

    E_old = tour_length(tour, coords)

    # ------------------------------------------------------------------ #
    # 2. 2-opt move: reverse the segment between two random positions    #
    # ------------------------------------------------------------------ #
    i, j = sorted(rng.choice(N, size=2, replace=False).tolist())
    new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]

    E_new = tour_length(new_tour, coords)

    # ------------------------------------------------------------------ #
    # 3. Accept / reject (simulated annealing)                            #
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
    # 4. Compute up to four loss terms and backprop                       #
    # ------------------------------------------------------------------ #
    model.train()

    loss_terms: list[torch.Tensor] = []

    for p in [i, j]:
        # forward: skip position 0 (no previous city)
        if p > 0:
            term = _loss_at_step(model, new_tour, p, coords, cell_ids, grid_size, rng, device)
            if term is not None:
                loss_terms.append(term)

        # reverse: skip position N-1 (no next city to condition on)
        term = _reverse_loss_at_step(model, new_tour, p, coords, cell_ids, grid_size, rng, device)
        if term is not None:
            loss_terms.append(term)

    if not loss_terms:
        return StepResult(accepted=True, E_old=E_old, E_new=E_new, tour=new_tour, n_loss_terms=0)

    total_loss = sum(loss_terms)   # type: ignore[arg-type]

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
