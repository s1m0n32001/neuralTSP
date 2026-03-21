"""
Training loop.

Usage
-----
    python scripts/train.py --data data/train.npz --epochs 100

The temperature decreases linearly from T_start to T_end over the full run.
At each iteration we pick a random instance from the dataset.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from neuraltsp.config import ModelConfig
from neuraltsp.data.dataset import TSPDataset
from neuraltsp.model.model import TSPTransformer
from neuraltsp.train.step import training_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help=".npz training dataset")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--T_start", type=float, default=1.0, help="Initial SA temperature")
    p.add_argument("--T_end", type=float, default=0.0, help="Final SA temperature")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    # model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def temperature_at(epoch: int, total_epochs: int, T_start: float, T_end: float) -> float:
    """Linear decay from T_start (epoch 0) to T_end (epoch total_epochs-1)."""
    if total_epochs <= 1:
        return T_end
    frac = epoch / (total_epochs - 1)
    return T_start + frac * (T_end - T_start)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rng = np.random.default_rng(args.seed)

    dataset = TSPDataset(args.data)
    print(dataset)

    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = TSPTransformer(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    grid_size = dataset.grid_size

    total_iters = args.epochs * len(dataset)
    global_step = 0

    for epoch in range(args.epochs):
        T = temperature_at(epoch, args.epochs, args.T_start, args.T_end)

        # shuffle instance order each epoch
        order = rng.permutation(len(dataset))

        epoch_losses, n_accepted = [], 0

        for idx in tqdm(order, desc=f"Epoch {epoch+1}/{args.epochs}  T={T:.4f}", leave=False):
            item = dataset[int(idx)]
            coords = item["coords"].numpy()
            cell_ids = item["cell_ids"].numpy()

            result = training_step(
                model, optimizer, coords, cell_ids,
                grid_size, rng, device, temperature=T,
            )

            if result.accepted:
                n_accepted += 1
            if result.loss is not None:
                epoch_losses.append(result.loss)

            global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        accept_rate = n_accepted / len(dataset)
        print(
            f"Epoch {epoch+1:4d}  T={T:.4f}  "
            f"loss={avg_loss:.4f}  accept={accept_rate:.2%}"
        )

        if (epoch + 1) % args.save_every == 0:
            ckpt = args.checkpoint_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "cfg": cfg}, ckpt)
            print(f"  saved {ckpt}")

    # final checkpoint
    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "cfg": cfg},
        args.checkpoint_dir / "final.pt",
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
