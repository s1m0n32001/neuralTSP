import numpy as np
import torch
import pytest

from neuraltsp.config import ModelConfig
from neuraltsp.data.generator import generate_instance
from neuraltsp.model.decode import predict_tour, tour_length


@pytest.fixture
def model():
    from neuraltsp.model.model import TSPTransformer
    cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0)
    return TSPTransformer(cfg).eval()


def make_instance(n=20, grid_size=4, seed=0):
    rng = np.random.default_rng(seed)
    return generate_instance(n, grid_size, rng)


def test_tour_visits_all_cities(model):
    inst = make_instance(n=20, grid_size=4)
    rng = np.random.default_rng(0)
    tour = predict_tour(model, inst["coords"], inst["cell_ids"], 4, rng,
                        device=torch.device("cpu"))
    assert sorted(tour) == list(range(20))


def test_tour_no_duplicates(model):
    inst = make_instance(n=15, grid_size=3)
    rng = np.random.default_rng(1)
    tour = predict_tour(model, inst["coords"], inst["cell_ids"], 3, rng,
                        device=torch.device("cpu"))
    assert len(tour) == len(set(tour))


def test_greedy_tour_deterministic(model):
    inst = make_instance(n=10, grid_size=2)
    # greedy + same rng seed → same tour
    tour1 = predict_tour(model, inst["coords"], inst["cell_ids"], 2,
                         np.random.default_rng(7), torch.device("cpu"), greedy=True)
    tour2 = predict_tour(model, inst["coords"], inst["cell_ids"], 2,
                         np.random.default_rng(7), torch.device("cpu"), greedy=True)
    assert tour1 == tour2


def test_tour_length_positive(model):
    inst = make_instance(n=12, grid_size=3)
    rng = np.random.default_rng(2)
    tour = predict_tour(model, inst["coords"], inst["cell_ids"], 3, rng,
                        device=torch.device("cpu"))
    length = tour_length(tour, inst["coords"])
    assert length > 0.0


def test_tour_length_triangle_inequality(model):
    """Each leg must be shorter than the total tour."""
    inst = make_instance(n=10, grid_size=2)
    rng = np.random.default_rng(3)
    tour = predict_tour(model, inst["coords"], inst["cell_ids"], 2, rng,
                        device=torch.device("cpu"))
    length = tour_length(tour, inst["coords"])
    coords = inst["coords"]
    for i in range(len(tour)):
        a, b = coords[tour[i]], coords[tour[(i + 1) % len(tour)]]
        leg = float(np.sqrt(((a - b) ** 2).sum()))
        assert leg < length + 1e-6
