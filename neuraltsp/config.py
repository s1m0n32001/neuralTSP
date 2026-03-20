from dataclasses import dataclass


@dataclass
class DataConfig:
    # number of cities per TSP instance
    n_points: int = 100
    # grid is grid_size x grid_size cells; coordinates in [0, 1)
    grid_size: int = 10
    # random seed for reproducibility (None = non-deterministic)
    seed: int | None = None


@dataclass
class ModelConfig:
    # embedding dimension
    d_model: int = 128
    # number of attention heads
    n_heads: int = 8
    # number of transformer layers
    n_layers: int = 6
    # feed-forward hidden dim
    d_ff: int = 512
    dropout: float = 0.1
