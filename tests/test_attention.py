import torch
import pytest

from neuraltsp.model.attention import TriangularAttentionContext


@pytest.fixture
def module():
    return TriangularAttentionContext(foreign_samples_per_cell=1)


def make_cell_ids(B, N, grid_size):
    """Random cell ids in a reproducible way."""
    rng = torch.Generator()
    rng.manual_seed(0)
    return torch.randint(0, grid_size ** 2, (B, N), generator=rng)


# ---------------------------------------------------------------------------
# build_context_indices
# ---------------------------------------------------------------------------

def test_output_shapes(module):
    B, N, G = 2, 16, 4
    cell_ids = make_cell_ids(B, N, G)
    ctx_idx, ctx_mask = module.build_context_indices(cell_ids)
    assert ctx_idx.shape[:2] == (B, N)
    assert ctx_mask.shape == ctx_idx.shape


def test_mask_coverage(module):
    """Mask should be True wherever ctx_idx != -1."""
    B, N, G = 2, 20, 3
    cell_ids = make_cell_ids(B, N, G)
    ctx_idx, ctx_mask = module.build_context_indices(cell_ids)
    assert (ctx_idx[ctx_mask] >= 0).all()
    assert (ctx_idx[~ctx_mask] == -1).all()


def test_same_cell_always_included(module):
    """Every point in the same cell must appear in the context of every other point in that cell."""
    B, N, G = 1, 12, 3
    # Force a simple layout: all points in cell 0
    cell_ids = torch.zeros(B, N, dtype=torch.long)
    ctx_idx, ctx_mask = module.build_context_indices(cell_ids)

    # For B=0, every query should see all N points
    for i in range(N):
        valid = ctx_idx[0, i][ctx_mask[0, i]]
        pt_set = set(valid.tolist())
        assert set(range(N)).issubset(pt_set), f"Point {i} missing some same-cell neighbours"


def test_context_size_lower_bound(module):
    """Each query must attend to at least its own cell (≥1) plus one per foreign cell."""
    B, N, G = 1, 50, 5
    cell_ids = make_cell_ids(B, N, G)
    ctx_idx, ctx_mask = module.build_context_indices(cell_ids)

    n_populated_cells = cell_ids[0].unique().numel()
    # minimum context = 1 (own cell has at least 1 pt) + (n_populated_cells-1) foreign samples
    min_ctx = 1 + (n_populated_cells - 1)
    for i in range(N):
        n_valid = ctx_mask[0, i].sum().item()
        assert n_valid >= min_ctx, f"Point {i}: context size {n_valid} < lower bound {min_ctx}"


def test_valid_indices_in_range(module):
    """All valid context indices must be in [0, N)."""
    B, N, G = 3, 30, 4
    cell_ids = make_cell_ids(B, N, G)
    ctx_idx, ctx_mask = module.build_context_indices(cell_ids)
    valid = ctx_idx[ctx_mask]
    assert (valid >= 0).all()
    assert (valid < N).all()


# ---------------------------------------------------------------------------
# forward (placeholder passthrough)
# ---------------------------------------------------------------------------

def test_forward_shape(module):
    B, N, D = 2, 16, 32
    G = 4
    x = torch.randn(B, N, D)
    cell_ids = make_cell_ids(B, N, G)
    out = module(x, cell_ids)
    assert out.shape == (B, N, D)
