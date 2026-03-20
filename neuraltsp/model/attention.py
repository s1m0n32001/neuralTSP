"""
Triangular attention context builder.

For each query point i, the set of key/value points is:
    - All points in the same grid cell as i  (local context)
    - One randomly sampled point from every *other* cell  (global sketch)

This yields a sparse, variable-size context per query.  The module provides
two things:

1. `TriangularAttentionContext.build_context_indices` – pure index computation
   (no learned parameters), useful both at dataset-prep time and during forward.

2. `TriangularAttentionContext` – a nn.Module placeholder that will house the
   full attention mechanism once the architecture is finalised.

Index layout returned by build_context_indices
-----------------------------------------------
For a batch of B instances, each with N points and G*G cells:

    ctx_indices : LongTensor (B, N, K_max)
        For query i, ctx_indices[b, i] contains the indices of the context
        points.  Padding entries are filled with -1 and should be masked.
    ctx_mask    : BoolTensor  (B, N, K_max)
        True where ctx_indices is a valid (non-padding) index.
    K_max       : int – maximum context size across the batch
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TriangularAttentionContext(nn.Module):
    """
    Builds triangular attention context indices for a batch of TSP instances.

    Parameters
    ----------
    foreign_samples_per_cell : int
        Number of random points to sample from each *foreign* cell.
        Defaults to 1 (one random representative per foreign cell).
    """

    def __init__(self, foreign_samples_per_cell: int = 1) -> None:
        super().__init__()
        self.foreign_samples_per_cell = foreign_samples_per_cell

    # ------------------------------------------------------------------
    # Core index builder
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_context_indices(
        self,
        cell_ids: Tensor,          # (B, N)  int
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute context point indices for every query in the batch.

        Returns
        -------
        ctx_indices : LongTensor (B, N, K_max)  – index into [0, N); -1 = pad
        ctx_mask    : BoolTensor  (B, N, K_max) – True = valid
        """
        B, N = cell_ids.shape
        device = cell_ids.device

        all_indices: list[list[list[int]]] = []

        for b in range(B):
            ids = cell_ids[b]  # (N,)
            unique_cells = ids.unique().tolist()

            # map cell_id -> list of point indices in that cell
            cell_to_pts: dict[int, list[int]] = {}
            for pt_idx in range(N):
                cid = int(ids[pt_idx].item())
                cell_to_pts.setdefault(cid, []).append(pt_idx)

            instance_ctx: list[list[int]] = []
            for pt_idx in range(N):
                my_cell = int(ids[pt_idx].item())
                ctx: list[int] = []

                # --- local context: all points in same cell ---
                ctx.extend(cell_to_pts[my_cell])

                # --- global sketch: one random point per foreign cell ---
                for cid in unique_cells:
                    if cid == my_cell:
                        continue
                    pts_in_cell = cell_to_pts[cid]
                    # sample without replacement up to foreign_samples_per_cell
                    k = min(self.foreign_samples_per_cell, len(pts_in_cell))
                    perm = torch.randperm(
                        len(pts_in_cell), generator=generator, device=device
                    )
                    for j in perm[:k].tolist():
                        ctx.append(pts_in_cell[j])

                instance_ctx.append(ctx)

            all_indices.append(instance_ctx)

        # ---- pad to K_max ----
        K_max = max(
            len(ctx)
            for instance_ctx in all_indices
            for ctx in instance_ctx
        )

        ctx_indices = torch.full((B, N, K_max), -1, dtype=torch.long, device=device)
        ctx_mask = torch.zeros((B, N, K_max), dtype=torch.bool, device=device)

        for b, instance_ctx in enumerate(all_indices):
            for i, ctx in enumerate(instance_ctx):
                k = len(ctx)
                ctx_indices[b, i, :k] = torch.tensor(ctx, dtype=torch.long)
                ctx_mask[b, i, :k] = True

        return ctx_indices, ctx_mask

    # ------------------------------------------------------------------
    # Forward – placeholder; full attention to be implemented later
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,           # (B, N, d_model) – point embeddings
        cell_ids: Tensor,    # (B, N) – grid cell assignment
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """
        Apply triangular attention.

        Currently returns *x* unchanged.  Replace with full attention once
        the architecture is decided.
        """
        ctx_indices, ctx_mask = self.build_context_indices(cell_ids, generator)
        # TODO: implement attention over (x, ctx_indices, ctx_mask)
        _ = ctx_indices, ctx_mask  # suppress unused warnings
        return x
