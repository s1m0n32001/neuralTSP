"""
Training loop with validation.

Usage
-----
    python scripts/train.py \\
        --train data/train.npz \\
        --val   data/val.npz   \\
        --epochs 100

Every --val_every epochs the script:
  1. Computes greedy tour lengths for all validation instances and prints
     mean / std / min / max.  Results are appended to <log_dir>/val_lengths.csv.
  2. Plots --n_val_plots random validation instances and saves them to
     <log_dir>/plots/epoch_NNNN/.

The temperature decreases linearly from T_start to T_end over all epochs.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from neuraltsp.config import ModelConfig
from neuraltsp.data.dataset import TSPDataset
from neuraltsp.model.decode import predict_tour
from neuraltsp.model.model import TSPTransformer
from neuraltsp.train.cache import PathCache
from neuraltsp.train.eval import evaluate_dataset
from neuraltsp.train.plot import plot_tour
from neuraltsp.train.step import training_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def temperature_at(epoch: int, total_epochs: int, T_start: float, T_end: float) -> float:
    if total_epochs <= 1:
        return T_end
    frac = epoch / (total_epochs - 1)
    return T_start + frac * (T_end - T_start)


def run_validation(
    model: TSPTransformer,
    val_dataset: TSPDataset,
    grid_size: int,
    device: torch.device,
    epoch: int,
    log_dir: Path,
    n_plots: int,
    rng_plots: np.random.Generator,
) -> None:
    """Evaluate on val set, log lengths, and save tour plots."""

    # --- tour-length stats ---
    lengths = evaluate_dataset(model, val_dataset, grid_size, device, seed=0)
    print(
        f"  val  mean={lengths.mean():.4f}  std={lengths.std():.4f}"
        f"  min={lengths.min():.4f}  max={lengths.max():.4f}"
    )

    csv_path = log_dir / "val_lengths.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "mean", "std", "min", "max"])
        writer.writerow([epoch, lengths.mean(), lengths.std(), lengths.min(), lengths.max()])

    # --- tour plots ---
    if n_plots <= 0:
        return

    plot_dir = log_dir / "plots" / f"epoch_{epoch:04d}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    indices = rng_plots.choice(len(val_dataset), size=min(n_plots, len(val_dataset)), replace=False)
    rng_decode = np.random.default_rng(0)

    model.eval()
    with torch.no_grad():
        for rank, idx in enumerate(indices):
            item = val_dataset[int(idx)]
            coords = item["coords"].numpy()
            cell_ids = item["cell_ids"].numpy()

            tour = predict_tour(
                model, coords, cell_ids, grid_size,
                rng=rng_decode, device=device, greedy=True,
            )
            plot_tour(
                coords, cell_ids, tour, grid_size,
                title=f"epoch {epoch}  instance {idx}",
                path=plot_dir / f"instance_{idx:05d}.png",
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train", type=Path, required=True, help=".npz training dataset")
    p.add_argument("--val",   type=Path, default=None,  help=".npz validation dataset")
    # training
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--T_start",  type=float, default=1.0)
    p.add_argument("--T_end",    type=float, default=0.0)
    p.add_argument("--seed",     type=int,   default=0)
    # validation
    p.add_argument("--val_every",   type=int, default=10,
                   help="Run validation every N epochs (default: 10)")
    p.add_argument("--n_val_plots", type=int, default=4,
                   help="Number of val instances to plot per val run (default: 4)")
    p.add_argument("--cache_reset_every", type=int, default=10,
                   help="Wipe tour cache every N epochs, forcing fresh prediction (default: 10)")
    # logging / checkpointing
    p.add_argument("--log_dir",        type=Path, default=Path("logs"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every",     type=int,  default=10)
    # model
    p.add_argument("--d_model",  type=int,   default=128)
    p.add_argument("--n_heads",  type=int,   default=8)
    p.add_argument("--n_layers", type=int,   default=6)
    p.add_argument("--d_ff",     type=int,   default=512)
    p.add_argument("--dropout",  type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rng = np.random.default_rng(args.seed)
    # separate rng for plot-instance selection so it doesn't affect training
    rng_plots = np.random.default_rng(args.seed + 1)

    train_dataset = TSPDataset(args.train)
    print(f"Train: {train_dataset}")

    val_dataset = None
    if args.val is not None:
        val_dataset = TSPDataset(args.val)
        print(f"Val:   {val_dataset}")

    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = TSPTransformer(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    grid_size = train_dataset.grid_size
    cache = PathCache()

    for epoch in range(args.epochs):
        # wipe cache at epoch 0 and every cache_reset_every epochs thereafter
        if epoch % args.cache_reset_every == 0:
            cache.wipe()

        T = temperature_at(epoch, args.epochs, args.T_start, args.T_end)
        order = rng.permutation(len(train_dataset))
        epoch_losses, n_accepted, n_predicted = [], 0, 0

        for idx in tqdm(order, desc=f"Epoch {epoch+1}/{args.epochs}  T={T:.4f}", leave=False):
            idx = int(idx)
            item = train_dataset[idx]
            coords = item["coords"].numpy()
            cell_ids = item["cell_ids"].numpy()

            cached_tour = cache.get(idx)
            if cached_tour is None:
                n_predicted += 1

            result = training_step(
                model, optimizer, coords, cell_ids,
                grid_size, rng, device, temperature=T,
                cached_tour=cached_tour,
            )

            # write the current tour back to cache (accepted → new_tour, else unchanged)
            cache.set(idx, result.tour)

            if result.accepted:
                n_accepted += 1
            if result.loss is not None:
                epoch_losses.append(result.loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        accept_rate = n_accepted / len(train_dataset)
        print(
            f"Epoch {epoch+1:4d}  T={T:.4f}  "
            f"loss={avg_loss:.4f}  accept={accept_rate:.2%}  "
            f"predicted={n_predicted}/{len(train_dataset)}"
        )

        # --- validation ---
        if val_dataset is not None and (epoch + 1) % args.val_every == 0:
            run_validation(
                model, val_dataset, grid_size, device,
                epoch=epoch + 1,
                log_dir=args.log_dir,
                n_plots=args.n_val_plots,
                rng_plots=rng_plots,
            )

        # --- checkpoint ---
        if (epoch + 1) % args.save_every == 0:
            ckpt = args.checkpoint_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "cfg": cfg}, ckpt)
            print(f"  saved {ckpt}")

    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "cfg": cfg},
        args.checkpoint_dir / "final.pt",
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
