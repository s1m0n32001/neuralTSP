"""
Validation-set evaluation.

`evaluate_dataset` runs greedy decoding on every instance in a dataset
and returns the tour lengths as a numpy array.  Greedy (argmax) is used
so results are deterministic and comparable across epochs.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from neuraltsp.data.dataset import TSPDataset
from neuraltsp.model.decode import predict_tour, tour_length
from neuraltsp.model.model import TSPTransformer


def evaluate_dataset(
    model: TSPTransformer,
    dataset: TSPDataset,
    grid_size: int,
    device: torch.device,
    seed: int = 0,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute greedy-decoded tour lengths for every instance in *dataset*.

    Parameters
    ----------
    model        : TSPTransformer (set to eval mode internally)
    dataset      : TSPDataset
    grid_size    : cells per axis
    device       : torch device
    seed         : RNG seed — only affects candidate sampling order;
                   greedy argmax makes the start city and decision deterministic
    show_progress: show a tqdm bar

    Returns
    -------
    lengths : (n_instances,) float64 array of closed-tour Euclidean lengths
    """
    model.eval()
    rng = np.random.default_rng(seed)
    lengths = np.empty(len(dataset), dtype=np.float64)

    it = tqdm(range(len(dataset)), desc="val", leave=False) if show_progress else range(len(dataset))

    with torch.no_grad():
        for i in it:
            item = dataset[i]
            coords = item["coords"].numpy()
            cell_ids = item["cell_ids"].numpy()

            tour = predict_tour(
                model, coords, cell_ids, grid_size,
                rng=rng, device=device, greedy=True,
            )
            lengths[i] = tour_length(tour, coords)

    return lengths
