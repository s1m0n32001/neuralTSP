import numpy as np
import pytest

from neuraltsp.config import DataConfig
from neuraltsp.data.generator import (
    assign_cells,
    generate_dataset,
    generate_instance,
    instances_to_arrays,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# assign_cells
# ---------------------------------------------------------------------------

def test_assign_cells_shape(rng):
    coords = rng.random((50, 2)).astype(np.float32)
    ids = assign_cells(coords, grid_size=10)
    assert ids.shape == (50,)


def test_assign_cells_range():
    grid_size = 5
    coords = np.array([[0.0, 0.0], [0.999, 0.999], [0.5, 0.5]], dtype=np.float32)
    ids = assign_cells(coords, grid_size)
    assert ids.min() >= 0
    assert ids.max() < grid_size ** 2


def test_assign_cells_boundary():
    # x=1.0 or y=1.0 should not produce out-of-range ids
    coords = np.array([[1.0, 1.0]], dtype=np.float32)
    ids = assign_cells(coords, grid_size=4)
    assert 0 <= ids[0] < 16


def test_assign_cells_correct():
    # point at (0.15, 0.25), grid_size=10 → col=1, row=2, cell=21
    coords = np.array([[0.15, 0.25]], dtype=np.float32)
    ids = assign_cells(coords, grid_size=10)
    assert ids[0] == 21


# ---------------------------------------------------------------------------
# generate_instance
# ---------------------------------------------------------------------------

def test_generate_instance_keys(rng):
    inst = generate_instance(n_points=20, grid_size=4, rng=rng)
    assert set(inst.keys()) == {"coords", "cell_ids"}


def test_generate_instance_shapes(rng):
    inst = generate_instance(n_points=30, grid_size=5, rng=rng)
    assert inst["coords"].shape == (30, 2)
    assert inst["cell_ids"].shape == (30,)


def test_generate_instance_coord_range(rng):
    inst = generate_instance(n_points=1000, grid_size=10, rng=rng)
    assert inst["coords"].min() >= 0.0
    assert inst["coords"].max() < 1.0


def test_generate_instance_cell_range(rng):
    grid_size = 7
    inst = generate_instance(n_points=200, grid_size=grid_size, rng=rng)
    assert inst["cell_ids"].min() >= 0
    assert inst["cell_ids"].max() < grid_size ** 2


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

def test_generate_dataset_length():
    cfg = DataConfig(n_points=10, grid_size=3, seed=1)
    dataset = generate_dataset(cfg, n_instances=5)
    assert len(dataset) == 5


def test_generate_dataset_reproducible():
    cfg = DataConfig(n_points=20, grid_size=4, seed=99)
    d1 = generate_dataset(cfg, n_instances=3)
    d2 = generate_dataset(cfg, n_instances=3)
    np.testing.assert_array_equal(d1[0]["coords"], d2[0]["coords"])


# ---------------------------------------------------------------------------
# instances_to_arrays
# ---------------------------------------------------------------------------

def test_instances_to_arrays_shapes():
    cfg = DataConfig(n_points=15, grid_size=3, seed=7)
    instances = generate_dataset(cfg, n_instances=4)
    arrays = instances_to_arrays(instances)
    assert arrays["coords"].shape == (4, 15, 2)
    assert arrays["cell_ids"].shape == (4, 15)
