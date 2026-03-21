# NeuralTSP

A transformer-based solver for the **Traveling Salesman Problem (TSP)** trained with a simulated-annealing-style objective.

---

## Idea

Given a set of N cities placed randomly in the unit square, the goal is to find the shortest closed tour visiting every city exactly once.

### Grid structure

The unit square is partitioned into a **G × G grid** of equal square cells. Each city belongs to exactly one cell. This spatial structure is used to build a sparse, distance-aware attention context at every decoding step.

### Autoregressive decoding

The model constructs a tour city by city. At each step, starting from the current city P:

1. **Candidate selection** — a subset of the remaining unvisited cities is assembled using a distance-based thinning rule. At Chebyshev distance *d* from P's cell, `ceil(n / 2^d)` cities are sampled from that cell (at least 1 from every non-empty cell). This keeps nearby cities densely represented and distant cells sparsely sampled.

2. **Transformer forward pass** — the model receives the token sequence `[P, A, Q₁, …, Qₖ]` where A is the tour's start city and the Qᵢ are candidates. All coordinates are translated so P sits at the origin. A gets a learned additive bias added to its embedding (a type token distinguishing it as the start-of-tour anchor). Standard multi-head self-attention is applied across all tokens.

3. **City selection** — the per-token output vectors are projected to scalars. A softmax over the candidate positions gives a probability distribution; the next city is sampled (or taken greedily at evaluation time).

### Training: simulated annealing + cross-entropy

Training does not require ground-truth optimal tours. Instead, it uses a **simulated annealing (SA)**-style loop:

1. Predict a full tour for the current instance (or reuse a cached one).
2. Randomly swap two cities in the tour to get a candidate tour.
3. **Accept** the swap if it shortens the tour; otherwise accept with probability `exp(-ΔE / T)`, where ΔE is the length increase and T is the current temperature.
4. If accepted, push the model toward that decision via cross-entropy loss. For each of the two swapped positions, two loss terms are computed:
   - **Forward**: predict the new city at that position given the tour prefix.
   - **Reverse**: predict the new city at that position given the tour suffix read right-to-left.

   Each term is normalised by `log(K)` (K = number of candidates) to keep the scale comparable across steps with different context sizes. Terms are skipped if the correct city was not sampled into the candidate set.

The **temperature T decays linearly** from `T_start` to `T_end` over training, gradually shifting the objective from exploration toward pure improvement.

**Tour caching** — to avoid re-running the expensive O(N)-step prediction every iteration, the current tour is cached per instance and reused across epochs. Every `cache_reset_every` epochs the cache is wiped and tours are re-predicted from the updated model weights.

---

## Project structure

```
NeuralTSP/
├── neuraltsp/
│   ├── config.py               # DataConfig, ModelConfig dataclasses
│   ├── data/
│   │   ├── generator.py        # Instance generation and cell assignment
│   │   └── dataset.py          # TSPDataset (PyTorch Dataset over .npz files)
│   ├── model/
│   │   ├── candidates.py       # Spatial candidate selection
│   │   ├── model.py            # TSPTransformer (TransformerEncoder + logit head)
│   │   └── decode.py           # Autoregressive tour prediction
│   └── train/
│       ├── cache.py            # PathCache (per-instance tour cache)
│       ├── step.py             # Single SA training step
│       ├── eval.py             # Validation: greedy tour lengths over a dataset
│       └── plot.py             # Tour visualisation
├── scripts/
│   ├── generate_dataset.py     # Generate and save train/val/test .npz files
│   ├── train.py                # Training loop with validation and checkpointing
│   └── visualize.py            # Plot a single instance from a .npz file
└── tests/
    ├── test_generator.py
    ├── test_dataset.py
    ├── test_candidates.py
    ├── test_model.py
    ├── test_decode.py
    └── test_step.py
```

---

## Installation

Requires **Python 3.10+**.

```bash
git clone <repo-url>
cd NeuralTSP
pip install -e ".[dev]"
```

The core dependencies (`torch`, `numpy`, `tqdm`, `matplotlib`) are installed automatically. The `[dev]` extra adds `pytest` for running the test suite.

---

## Quickstart

### 1. Generate datasets

```bash
python scripts/generate_dataset.py \
    --out data/ \
    --n_points 100 \
    --grid_size 10 \
    --train 100000 \
    --val   10000  \
    --test  10000  \
    --seed  42
```

This writes `data/train.npz`, `data/val.npz`, and `data/test.npz`.

### 2. Train

```bash
python scripts/train.py \
    --train data/train.npz \
    --val   data/val.npz   \
    --epochs 200            \
    --T_start 1.0           \
    --T_end   0.0           \
    --cache_reset_every 10  \
    --val_every 10          \
    --n_val_plots 6         \
    --log_dir logs/         \
    --checkpoint_dir checkpoints/
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--T_start` | `1.0` | Initial SA temperature |
| `--T_end` | `0.0` | Final SA temperature (linear decay) |
| `--cache_reset_every` | `10` | Epochs between full tour re-predictions |
| `--val_every` | `10` | Epochs between validation runs |
| `--n_val_plots` | `4` | Val instances to plot per validation run |
| `--d_model` | `128` | Transformer embedding dimension |
| `--n_heads` | `8` | Attention heads |
| `--n_layers` | `6` | Transformer layers |

Training logs are written to `logs/`:
- `val_lengths.csv` — mean / std / min / max tour length per validation epoch
- `plots/epoch_NNNN/` — tour visualisations for the same fixed val instances each epoch

### 3. Visualise a dataset instance

```bash
python scripts/visualize.py data/val.npz --idx 0
```

Add `--out plot.png` to save instead of displaying interactively. Use `--highlight_cell N` to shade a specific grid cell.

### 4. Run tests

```bash
pytest
```
