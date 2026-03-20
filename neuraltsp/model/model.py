"""
TSP Transformer model.

At each autoregressive step the model receives a token sequence:

    [P, A, Q1, Q2, ..., QK]

where:
    P  – current city (always at index 0 in the sequence)
    A  – starting city of the tour (always at index 1)
    Qi – candidate cities for the next hop

All coordinates are translated so that P sits at the origin.
A and the Qi may coincide only on the very last step (closing the tour).

The forward pass returns a logit for every position in the sequence.
The caller is responsible for masking P (position 0) out of the softmax —
it is never a valid next-city candidate.  A (position 1) should likewise
be masked unless it is the tour-closing step.

Architecture
------------
1. Linear embed: (x, y) → d_model
2. Start bias:  add a learned vector b to A's embedding only
3. TransformerEncoder: standard multi-head self-attention, batch_first=True
4. Output proj: d_model → 1  (scalar logit per token)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from neuraltsp.config import ModelConfig


class TSPTransformer(nn.Module):
    """
    Transformer that scores next-city candidates for TSP decoding.

    Parameters
    ----------
    cfg : ModelConfig
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # shared linear embedding for all tokens: 2D coord → d_model
        self.embed = nn.Linear(2, cfg.d_model)

        # learned additive bias for the start token A
        # acts like a type embedding distinguishing A from ordinary cities
        self.start_bias = nn.Parameter(torch.zeros(cfg.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,   # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # per-token scalar logit
        self.logit_proj = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        coords: Tensor,          # (B, S, 2)  relative to current city P
        is_start: Tensor,        # (B, S) bool — True only for the start token A
        pad_mask: Tensor | None = None,  # (B, S) bool — True = padding, ignored by attention
    ) -> Tensor:
        """
        Parameters
        ----------
        coords   : (B, S, 2) coordinates already translated so P is at (0, 0).
                   Sequence layout: [P, A, Q1, ..., QK, <pad...>]
        is_start : (B, S) bool mask — True at position of start token A.
        pad_mask : (B, S) bool — True for padding positions (default: no padding).

        Returns
        -------
        logits : (B, S) float  –∞ at padding positions, raw logit elsewhere.
                 Caller must additionally mask out P (index 0) before softmax.
        """
        # 1. embed all tokens
        x = self.embed(coords)                          # (B, S, d_model)

        # 2. add start bias to the A token(s)
        x = x + is_start.unsqueeze(-1).float() * self.start_bias

        # 3. transformer — pad_mask shape (B, S), True = ignore
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, S, d_model)

        # 4. scalar logit per position
        logits = self.logit_proj(x).squeeze(-1)         # (B, S)

        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask, float("-inf"))

        return logits

    # ------------------------------------------------------------------
    # Convenience: build input tensors from raw arrays (inference helper)
    # ------------------------------------------------------------------

    @staticmethod
    def build_inputs(
        current_idx: int,
        start_idx: int,
        candidate_idxs: list[int] | "np.ndarray",  # noqa: F821
        coords: "np.ndarray",                       # (N, 2) full coord array
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """
        Build (coords_rel, is_start) tensors for a single (unbatched) step.

        Sequence layout: [current, start, *candidates]

        Returns
        -------
        coords_rel : (1, S, 2)
        is_start   : (1, S) bool
        """
        import numpy as np

        seq_idxs = [current_idx, start_idx, *candidate_idxs]
        raw = coords[seq_idxs]                          # (S, 2)
        raw_rel = raw - raw[0]                          # translate: P → (0, 0)

        coords_rel = torch.tensor(raw_rel, dtype=torch.float32, device=device).unsqueeze(0)

        is_start = torch.zeros(1, len(seq_idxs), dtype=torch.bool, device=device)
        is_start[0, 1] = True                          # position 1 is always A

        return coords_rel, is_start
