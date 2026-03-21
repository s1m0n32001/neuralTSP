"""
PyTorch Dataset for TSP instances stored as .npz files.

File layout produced by scripts/generate_dataset.py:
    coords   : (n_instances, n_points, 2)  float32
    cell_ids : (n_instances, n_points)     int32
    meta     : scalar attrs  – n_points, grid_size, seed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TSPDataset(Dataset):
    """
    Dataset of TSP instances loaded from an .npz file.

    Each item is a dict:
        'coords'   : (N, 2) float32 tensor  – point coordinates in [0, 1)
        'cell_ids' : (N,)   int32  tensor   – grid-cell index per point
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        data = np.load(path)
        # keep on CPU as numpy; convert to tensor lazily in __getitem__
        self._coords = data["coords"]      # (B, N, 2)
        self._cell_ids = data["cell_ids"]  # (B, N)

        self.n_points = int(self._coords.shape[1])
        self.grid_size = int(data["grid_size"]) if "grid_size" in data else None

    def __len__(self) -> int:
        return len(self._coords)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "coords": torch.from_numpy(self._coords[idx]),
            "cell_ids": torch.from_numpy(self._cell_ids[idx]),
        }

    def __repr__(self) -> str:
        return (
            f"TSPDataset(n_instances={len(self)}, "
            f"n_points={self.n_points}, "
            f"grid_size={self.grid_size})"
        )
