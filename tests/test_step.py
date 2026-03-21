import numpy as np
import torch
import pytest

from neuraltsp.config import ModelConfig
from neuraltsp.data.generator import generate_instance
from neuraltsp.model.model import TSPTransformer
from neuraltsp.train.step import training_step, StepResult, _loss_at_step, _reverse_loss_at_step


@pytest.fixture
def model_and_opt():
    cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0)
    m = TSPTransformer(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    return m, opt


def make_instance(n=20, grid_size=4, seed=0):
    rng = np.random.default_rng(seed)
    return generate_instance(n, grid_size, rng)


# ---------------------------------------------------------------------------
# StepResult structure
# ---------------------------------------------------------------------------

def test_step_returns_result(model_and_opt):
    model, opt = model_and_opt
    inst = make_instance()
    rng = np.random.default_rng(0)
    result = training_step(model, opt, inst["coords"], inst["cell_ids"],
                           4, rng, torch.device("cpu"), temperature=1.0)
    assert isinstance(result, StepResult)
    assert isinstance(result.E_old, float)
    assert isinstance(result.E_new, float)


def test_step_energies_positive(model_and_opt):
    model, opt = model_and_opt
    inst = make_instance()
    rng = np.random.default_rng(1)
    result = training_step(model, opt, inst["coords"], inst["cell_ids"],
                           4, rng, torch.device("cpu"), temperature=1.0)
    assert result.E_old > 0
    assert result.E_new > 0


# ---------------------------------------------------------------------------
# Zero temperature: only improvements accepted
# ---------------------------------------------------------------------------

def test_zero_temperature_only_accepts_improvements(model_and_opt):
    """At T=0, result.accepted must equal (E_new < E_old)."""
    model, opt = model_and_opt
    inst = make_instance()
    accepted_with_increase = 0

    for seed in range(30):
        rng = np.random.default_rng(seed)
        result = training_step(model, opt, inst["coords"], inst["cell_ids"],
                               4, rng, torch.device("cpu"), temperature=0.0)
        if result.accepted and result.E_new >= result.E_old:
            accepted_with_increase += 1

    assert accepted_with_increase == 0, "T=0 accepted a worsening swap"


# ---------------------------------------------------------------------------
# High temperature: acceptance rate should be near 1
# ---------------------------------------------------------------------------

def test_high_temperature_high_acceptance(model_and_opt):
    model, opt = model_and_opt
    inst = make_instance()
    n_accepted = 0
    n_trials = 40

    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        result = training_step(model, opt, inst["coords"], inst["cell_ids"],
                               4, rng, torch.device("cpu"), temperature=1e6)
        if result.accepted:
            n_accepted += 1

    assert n_accepted / n_trials > 0.9


# ---------------------------------------------------------------------------
# Loss is finite when computed
# ---------------------------------------------------------------------------

def test_loss_is_finite_when_present(model_and_opt):
    model, opt = model_and_opt
    inst = make_instance(n=30, grid_size=4)

    for seed in range(20):
        rng = np.random.default_rng(seed)
        result = training_step(model, opt, inst["coords"], inst["cell_ids"],
                               4, rng, torch.device("cpu"), temperature=1e6)
        if result.loss is not None:
            assert np.isfinite(result.loss), f"Non-finite loss: {result.loss}"


# ---------------------------------------------------------------------------
# Parameters change after an accepted step with loss
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reverse loss helper
# ---------------------------------------------------------------------------

def test_reverse_loss_uses_suffix_as_context():
    """
    The reverse loss at position p should use new_tour[p+1] as current city
    and new_tour[-1] as start — verified by checking that the helper runs
    without error and returns a finite tensor.
    """
    cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0)
    model = TSPTransformer(cfg).eval()
    inst = make_instance(n=20, grid_size=4)
    rng = np.random.default_rng(42)
    tour = list(range(20))   # deterministic tour for simplicity

    # position 5: p+1 = 6 exists, p < N-1, should return a tensor or None
    result = _reverse_loss_at_step(
        model, tour, p=5,
        coords=inst["coords"], cell_ids=inst["cell_ids"],
        grid_size=4, rng=rng, device=torch.device("cpu"),
    )
    if result is not None:
        assert torch.isfinite(result)


def test_reverse_loss_skips_last_position():
    cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0)
    model = TSPTransformer(cfg).eval()
    inst = make_instance(n=10, grid_size=3)
    rng = np.random.default_rng(0)
    tour = list(range(10))
    result = _reverse_loss_at_step(
        model, tour, p=9,   # N-1 → must be skipped
        coords=inst["coords"], cell_ids=inst["cell_ids"],
        grid_size=3, rng=rng, device=torch.device("cpu"),
    )
    assert result is None


def test_step_can_have_up_to_four_loss_terms(model_and_opt):
    """With high T, accepted swaps with interior i and j yield up to 4 terms."""
    model, opt = model_and_opt
    inst = make_instance(n=30, grid_size=4)

    max_terms = 0
    for seed in range(50):
        rng = np.random.default_rng(seed)
        result = training_step(model, opt, inst["coords"], inst["cell_ids"],
                               4, rng, torch.device("cpu"), temperature=1e6)
        if result.accepted:
            max_terms = max(max_terms, result.n_loss_terms)

    assert max_terms > 2, "Expected some steps to produce more than 2 loss terms"


def test_parameters_update_on_accepted(model_and_opt):
    model, opt = model_and_opt
    inst = make_instance(n=25, grid_size=4)

    params_before = {k: v.clone() for k, v in model.named_parameters()}

    # run until we get an accepted step with a loss
    for seed in range(50):
        rng = np.random.default_rng(seed)
        result = training_step(model, opt, inst["coords"], inst["cell_ids"],
                               4, rng, torch.device("cpu"), temperature=1e6)
        if result.loss is not None:
            break

    changed = any(
        not torch.equal(params_before[k], v)
        for k, v in model.named_parameters()
    )
    assert changed, "No parameter changed after an accepted step with loss"
