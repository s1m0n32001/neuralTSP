import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from neuraltsp.data.dataset import TSPDataset


def make_npz(tmp_path: Path, n_instances=8, n_points=20, grid_size=4) -> Path:
    rng = np.random.default_rng(0)
    coords = rng.random((n_instances, n_points, 2)).astype(np.float32)
    cell_ids = rng.integers(0, grid_size ** 2, size=(n_instances, n_points)).astype(np.int32)
    path = tmp_path / "test.npz"
    np.savez_compressed(path, coords=coords, cell_ids=cell_ids, grid_size=np.int32(grid_size))
    return path


@pytest.fixture
def npz_path(tmp_path):
    return make_npz(tmp_path)


def test_len(npz_path):
    ds = TSPDataset(npz_path)
    assert len(ds) == 8


def test_getitem_keys(npz_path):
    ds = TSPDataset(npz_path)
    item = ds[0]
    assert set(item.keys()) == {"coords", "cell_ids"}


def test_getitem_shapes(npz_path):
    ds = TSPDataset(npz_path)
    item = ds[0]
    assert item["coords"].shape == (20, 2)
    assert item["cell_ids"].shape == (20,)


def test_getitem_types(npz_path):
    ds = TSPDataset(npz_path)
    item = ds[0]
    assert item["coords"].dtype == torch.float32
    assert item["cell_ids"].dtype == torch.int32


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        TSPDataset("/nonexistent/path.npz")


def test_grid_size_stored(npz_path):
    ds = TSPDataset(npz_path)
    assert ds.grid_size == 4
