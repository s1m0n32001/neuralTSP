"""
Per-instance tour cache.

Stores the current best tour for each dataset instance so that
`training_step` can skip the expensive O(N)-step autoregressive prediction
when a cached tour already exists.

Lifecycle
---------
- First time an instance is seen (or after a wipe): predict_tour runs.
- Subsequent steps on the same instance: cached tour is used directly.
- Every `reset_every` training epochs the cache is wiped so the model
  gets to re-predict all tours from its current weights.
- When a swap is accepted the caller should write `result.tour` back into
  the cache so the stored tour stays up-to-date with accepted SA moves.
"""

from __future__ import annotations


class PathCache:
    def __init__(self) -> None:
        self._cache: dict[int, list[int]] = {}

    def get(self, idx: int) -> list[int] | None:
        """Return the cached tour for instance *idx*, or None if not cached."""
        return self._cache.get(idx)

    def set(self, idx: int, tour: list[int]) -> None:
        """Store (or overwrite) the tour for instance *idx*."""
        self._cache[idx] = tour

    def wipe(self) -> None:
        """Clear all cached tours (triggers fresh prediction next epoch)."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"PathCache(n_cached={len(self._cache)})"
