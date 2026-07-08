"""Generate the persistence-trail sequence for ONE fixed image (condensado.md §9).

Trains the competitive layer epoch by epoch and, after each epoch, records the
activation of a single fixed image. The whole sequence (plus the image and the
metadata the viewer needs) is written to a fixed ``.npz`` so the web server can
pick it up dynamically, without either side knowing about the other.

    python hebbian/gen_evolution.py --dataset data/processed/hline/hline.npz \
        --image-index 0 --epochs 80 --lr 0.15 --inhib

Then serve it (independently) with ``webapp_evolution.py``; re-running this
script overwrites the same file and the running server shows it on Refresh.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from .competitive_net import CompetitiveLayer
    from .train import build_layer
except ImportError:  # pragma: no cover - script fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from competitive_net import CompetitiveLayer
    from train import build_layer


DEFAULT_OUT = "experiments/evolution/sequence.npz"


def build_sequence(layer, X, fixed, epochs, lr, seed):
    """Train epoch by epoch; record the fixed image's activation each time."""
    rng = np.random.default_rng(seed)
    seq = [layer.activation(fixed).astype(np.float32)]  # step 0 = before training
    for e in range(epochs):
        layer.train_epoch(X, lr, rng=rng)
        seq.append(layer.activation(fixed).astype(np.float32))
        print(f"epoch {e+1}/{epochs}  fired={int((seq[-1] >= layer.fire_threshold).sum())}")
    return np.stack(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a persistence-trail sequence file")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--key", default="images")
    ap.add_argument("--model", default=None, help="start from this model.npz (else fresh)")
    ap.add_argument("--image-index", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--out", default=DEFAULT_OUT, help="sequence file to (over)write")
    # fresh-layer hyperparameters (used only when --model is not given)
    ap.add_argument("--n-in", type=int, default=784)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--rule", default="above_mean", choices=["above_mean", "softmax", "wta"])
    ap.add_argument("--reinforce-gain", type=float, default=1.0)
    ap.add_argument("--inhib", action="store_true")
    ap.add_argument("--inhib-spacing", type=int, default=5)
    ap.add_argument("--inhib-radius", type=int, default=8)
    ap.add_argument("--inhib-metric", default="cheby", choices=["cheby", "manhattan", "euclid"])
    ap.add_argument("--fire-threshold", type=float, default=0.40)
    ap.add_argument("--inhib-K", type=float, default=0.10)
    ap.add_argument("--inhib-gain", type=float, default=1.5)
    ap.add_argument("--inhib-mode", default="fraction", choices=["fraction", "hinge", "sigmoid"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.dataset)
    X = d[args.key].astype(np.float32) / 255.0
    X = X.reshape(len(X), -1)
    fixed = X[args.image_index % len(X)]

    if args.model:
        layer = CompetitiveLayer.load(args.model)
        print(f"loaded {layer}")
    else:
        layer = build_layer(args)
        print(f"fresh {layer}")

    print(f"building {args.epochs}-epoch sequence for image {args.image_index}...")
    seq = build_sequence(layer, X, fixed, args.epochs, args.lr, args.seed)

    side = int(round(X.shape[1] ** 0.5))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        seq=np.round(seq, 4).astype(np.float32),
        image=(fixed * 255).astype(np.uint8),
        side=np.int64(side),
        map_h=np.int64(layer.grid_h),
        map_w=np.int64(layer.grid_w),
        steps=np.int64(args.epochs),
        image_index=np.int64(args.image_index),
        fire_threshold=np.float64(layer.fire_threshold),
    )
    print(f"wrote {args.out}  seq={seq.shape} (steps+1, n_out)")


if __name__ == "__main__":
    main()
