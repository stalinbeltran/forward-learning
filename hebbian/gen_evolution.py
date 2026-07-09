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
    from .evolution_io import write_sequence
except ImportError:  # pragma: no cover - script fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from competitive_net import CompetitiveLayer
    from train import build_layer
    from evolution_io import write_sequence


DEFAULT_OUT = "experiments/evolution/sequence.npz"
DEFAULT_RUNS_DIR = "experiments/evolution/runs"


def build_sequence(layer, X, fixed, epochs, lr, seed,
                   min_persistence=None, persist_patience=5):
    """Train epoch by epoch; record the fixed image's activation each time.

    Convergence / early-stop criterion (condensado.md §7) is *cumulative
    persistence*: the fraction of the currently-firing set that has been lit
    without interruption for at least ``persist_patience`` epochs. Training
    stops as soon as that fraction reaches ``min_persistence`` (if given).
    """
    rng = np.random.default_rng(seed)
    thr = layer.fire_threshold
    a0 = layer.activation(fixed).astype(np.float32)
    seq = [a0]  # step 0 = before training
    run = (a0 >= thr).astype(np.int64)  # per-neuron unbroken firing streak
    converged_at = None
    for e in range(epochs):
        layer.train_epoch(X, lr, rng=rng)
        a = layer.activation(fixed).astype(np.float32)
        seq.append(a)
        fired = a >= thr
        run = np.where(fired, run + 1, 0)
        n_fired = int(fired.sum())
        pers = (int((run >= persist_patience).sum()) / n_fired) if n_fired else 0.0
        print(f"epoch {e+1}/{epochs}  fired={n_fired}  persistence={pers:.3f}")
        if min_persistence is not None and pers >= min_persistence:
            converged_at = e + 1
            print(f"CONVERGED at epoch {converged_at}: persistence {pers:.3f} "
                  f">= {min_persistence} (>= {persist_patience} epochs lit)")
            break
    if min_persistence is not None and converged_at is None:
        print(f"did NOT reach persistence {min_persistence} within {epochs} epochs")
    return np.stack(seq), converged_at


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a persistence-trail sequence file")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--key", default="images")
    ap.add_argument("--model", default=None, help="start from this model.npz (else fresh)")
    ap.add_argument("--image-index", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80, help="max epochs (upper bound)")
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--min-persistence", type=float, default=None,
                    help="stop when cumulative persistence reaches this fraction (e.g. 0.7)")
    ap.add_argument("--persist-patience", type=int, default=5,
                    help="epochs a neuron must stay lit to count as persistent")
    ap.add_argument("--out", default=DEFAULT_OUT, help="sequence file to (over)write")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                    help="archive dir where each run is also saved (never overwritten); "
                         "'' to disable archiving")
    # fresh-layer hyperparameters (used only when --model is not given)
    ap.add_argument("--n-in", type=int, default=784)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--rule", default="above_mean", choices=["above_mean", "softmax", "wta"])
    ap.add_argument("--reinforce-gain", type=float, default=1.0)
    ap.add_argument("--learning-rule", default="gate", choices=["gate", "truth_table"],
                    help="'gate' (base continua) o 'truth_table' (regla por conexión)")
    ap.add_argument("--rule-n", type=float, default=1.1, help="truth_table: factor de aprendizaje disparado")
    ap.add_argument("--rule-m", type=float, default=0.3, help="truth_table: factor de desaprendizaje disparado")
    ap.add_argument("--rule-hr", type=float, default=0.1, help="truth_table: inhibition rate")
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

    print(f"building up-to-{args.epochs}-epoch sequence for image {args.image_index}...")
    seq, converged_at = build_sequence(
        layer, X, fixed, args.epochs, args.lr, args.seed,
        min_persistence=args.min_persistence,
        persist_patience=args.persist_patience,
    )

    side = int(round(X.shape[1] ** 0.5))
    steps = len(seq) - 1
    src = os.path.basename(args.model) if args.model else "fresh"
    label = (f"gen · {os.path.basename(args.dataset)} · img{args.image_index} · "
             f"{steps}ép · lr{args.lr:g} · {layer.learning_rule} · {src}")
    meta = {
        "label": label,
        "script": "gen_evolution",
        "dataset": args.dataset.replace("\\", "/"),
        "model_source": (args.model.replace("\\", "/") if args.model else "fresh"),
        "learning_rule": layer.learning_rule,
        "lr": args.lr,
        "epochs": steps,
        "nn_epochs": int(layer.epochs_trained),
    }
    out_path, archive = write_sequence(
        args.out, args.runs_dir,
        seq=seq, image=(fixed * 255).astype(np.uint8), side=side,
        map_h=layer.grid_h, map_w=layer.grid_w,
        image_index=args.image_index, fire_threshold=layer.fire_threshold,
        converged_at=converged_at, meta=meta,
    )
    print(f"wrote {out_path}  seq={seq.shape} (steps+1, n_out)")
    if archive:
        print(f"archived run -> {archive}")


if __name__ == "__main__":
    main()
