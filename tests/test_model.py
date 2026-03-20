import torch
import pytest

from neuraltsp.config import ModelConfig
from neuraltsp.model.model import TSPTransformer


@pytest.fixture
def cfg():
    return ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0)


@pytest.fixture
def model(cfg):
    return TSPTransformer(cfg).eval()


def make_batch(B, S, seed=0):
    """Random coords + is_start (only position 1 is True)."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(B, S, 2, generator=g)
    # translate so position 0 (P) is at origin
    coords = coords - coords[:, :1, :]
    is_start = torch.zeros(B, S, dtype=torch.bool)
    is_start[:, 1] = True
    return coords, is_start


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape(model):
    B, S = 3, 10
    coords, is_start = make_batch(B, S)
    logits = model(coords, is_start)
    assert logits.shape == (B, S)


# ---------------------------------------------------------------------------
# Padding mask: padded positions get -inf logit
# ---------------------------------------------------------------------------

def test_padded_positions_are_neginf(model):
    B, S = 2, 8
    coords, is_start = make_batch(B, S)
    pad_mask = torch.zeros(B, S, dtype=torch.bool)
    pad_mask[:, -2:] = True  # last 2 positions are padding

    logits = model(coords, is_start, pad_mask=pad_mask)
    assert torch.all(logits[:, -2:] == float("-inf"))
    assert torch.all(torch.isfinite(logits[:, :-2]))


# ---------------------------------------------------------------------------
# Start bias changes embeddings of position 1
# ---------------------------------------------------------------------------

def test_start_bias_has_effect(model):
    B, S = 1, 5
    coords, is_start = make_batch(B, S)

    # run with normal is_start (position 1 is A)
    logits_normal = model(coords, is_start)

    # run with is_start all-False (no start token)
    is_start_none = torch.zeros_like(is_start)
    logits_none = model(coords, is_start_none)

    # start_bias is initialised to zeros so they will be equal at init,
    # but the parameter must exist and be learnable
    assert hasattr(model, "start_bias")
    assert model.start_bias.requires_grad


def test_start_bias_effect_after_perturbation(model):
    """After randomising start_bias, logits at pos-1 should differ."""
    B, S = 1, 5
    coords, is_start = make_batch(B, S)

    with torch.no_grad():
        model.start_bias.normal_()  # perturb away from zero

    is_start_none = torch.zeros_like(is_start)
    logits_with = model(coords, is_start)
    logits_without = model(coords, is_start_none)

    assert not torch.allclose(logits_with, logits_without)


# ---------------------------------------------------------------------------
# Deterministic in eval mode (no dropout)
# ---------------------------------------------------------------------------

def test_deterministic_eval(model):
    B, S = 2, 6
    coords, is_start = make_batch(B, S)
    l1 = model(coords, is_start)
    l2 = model(coords, is_start)
    assert torch.allclose(l1, l2)


# ---------------------------------------------------------------------------
# build_inputs helper
# ---------------------------------------------------------------------------

def test_build_inputs_shape():
    import numpy as np
    coords_np = np.random.default_rng(0).random((20, 2)).astype("float32")
    candidates = [3, 7, 11, 15]

    coords_t, is_start_t = TSPTransformer.build_inputs(
        current_idx=0,
        start_idx=1,
        candidate_idxs=candidates,
        coords=coords_np,
        device=torch.device("cpu"),
    )
    S = 2 + len(candidates)  # current + start + candidates
    assert coords_t.shape == (1, S, 2)
    assert is_start_t.shape == (1, S)
    assert is_start_t[0, 1].item() is True
    # current point P should be at origin
    assert torch.allclose(coords_t[0, 0], torch.zeros(2))


def test_build_inputs_and_forward(model):
    import numpy as np
    coords_np = np.random.default_rng(1).random((30, 2)).astype("float32")
    candidates = list(range(2, 10))

    coords_t, is_start_t = TSPTransformer.build_inputs(
        current_idx=0,
        start_idx=1,
        candidate_idxs=candidates,
        coords=coords_np,
        device=torch.device("cpu"),
    )
    logits = model(coords_t, is_start_t)
    # mask P (pos 0), softmax over rest
    logits[0, 0] = float("-inf")
    probs = torch.softmax(logits[0], dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
