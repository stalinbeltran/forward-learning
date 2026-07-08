"""Config-driven training for the competitive Hebbian layer.

Trains a ``CompetitiveLayer`` on a line dataset, printing per-epoch metrics
and saving a resumable ``model.npz`` plus a ``metrics.csv`` under
``experiments/<run>/``. Learning rate anneals linearly from ``--lr0`` to
``--lr-min`` across epochs (set them equal for a constant lr).
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

# Support running both as a module (python -m hebbian.train) and as a script.
try:
    from .competitive_net import CompetitiveLayer
    from . import metrics as M
except ImportError:  # pragma: no cover - script execution fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from competitive_net import CompetitiveLayer
    import metrics as M


def load_dataset(path: str, key: str = "images") -> np.ndarray:
    """Load images from an .npz and flatten to float32 rows in [0, 1]."""
    d = np.load(path)
    imgs = d[key].astype(np.float32) / 255.0
    return imgs.reshape(len(imgs), -1)


def lr_schedule(epoch: int, total: int, lr0: float, lr_min: float) -> float:
    if total <= 1:
        return lr0
    t = epoch / (total - 1)
    return lr0 + (lr_min - lr0) * t


def build_layer(args) -> CompetitiveLayer:
    return CompetitiveLayer(
        n_in=args.n_in,
        n_out=args.grid * args.grid,
        rule=args.rule,
        reinforce_gain=args.reinforce_gain,
        grid_h=args.grid,
        grid_w=args.grid,
        inhib_on=args.inhib,
        inhib_spacing=args.inhib_spacing,
        inhib_radius=args.inhib_radius,
        inhib_metric=args.inhib_metric,
        fire_threshold=args.fire_threshold,
        inhib_K=args.inhib_K,
        inhib_gain=args.inhib_gain,
        inhib_mode=args.inhib_mode,
        seed=args.seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the competitive Hebbian layer")
    ap.add_argument("--dataset", default="data/processed/lines_hebbian/lines.npz")
    ap.add_argument("--key", default="images", help="npz key for the images")
    ap.add_argument("--run", default="experiments/run", help="output dir")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=0.1)
    ap.add_argument("--lr-min", type=float, default=0.1)
    # model / hyperparameters
    ap.add_argument("--n-in", type=int, default=784)
    ap.add_argument("--grid", type=int, default=50, help="output map is grid x grid")
    ap.add_argument("--rule", default="above_mean", choices=["above_mean", "softmax", "wta"])
    ap.add_argument("--reinforce-gain", type=float, default=1.0)
    ap.add_argument("--inhib", action="store_true", help="enable lateral inhibition")
    ap.add_argument("--inhib-spacing", type=int, default=5)
    ap.add_argument("--inhib-radius", type=int, default=8)
    ap.add_argument("--inhib-metric", default="cheby", choices=["cheby", "manhattan", "euclid"])
    ap.add_argument("--fire-threshold", type=float, default=0.40)
    ap.add_argument("--inhib-K", type=float, default=0.10)
    ap.add_argument("--inhib-gain", type=float, default=1.5)
    ap.add_argument("--inhib-mode", default="fraction", choices=["fraction", "hinge", "sigmoid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="resume from a model.npz")
    args = ap.parse_args()

    os.makedirs(args.run, exist_ok=True)
    X = load_dataset(args.dataset, key=args.key)
    print(f"dataset: {X.shape[0]} samples x {X.shape[1]} dims")

    if args.resume:
        layer = CompetitiveLayer.load(args.resume)
        print(f"resumed {layer} from {args.resume}")
    else:
        layer = build_layer(args)
        print(f"init {layer}")

    rng = np.random.default_rng(args.seed)
    csv_path = os.path.join(args.run, "metrics.csv")
    cols = ["epoch", "lr", "dead_units", "coverage", "unique_winners",
            "mean_winner_activation", "mean_fired"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for e in range(args.epochs):
            lr = lr_schedule(e, args.epochs, args.lr0, args.lr_min)
            layer.train_epoch(X, lr, rng=rng)
            m = M.epoch_metrics(layer, X)
            row = {"epoch": layer.epochs_trained, "lr": round(lr, 5), **m}
            writer.writerow({k: row[k] for k in cols})
            f.flush()
            print(
                f"epoch {layer.epochs_trained:3d} lr={lr:.4f} "
                f"dead={m['dead_units']:4d} cov={m['coverage']:.3f} "
                f"uniq={m['unique_winners']:4d} "
                f"sharp={m['mean_winner_activation']:.3f} "
                f"fired={m['mean_fired']:.1f}"
            )

    model_path = os.path.join(args.run, "model.npz")
    layer.save(model_path)
    print(f"saved {model_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
