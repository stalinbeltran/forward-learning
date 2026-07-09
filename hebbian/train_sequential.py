"""Train one competitive layer on a dataset ONE image at a time, sequentially.

Unlike ``train.py`` (which shuffles the whole set every epoch), this presents a
single fixed image over and over until it meets the convergence criterion, then
moves on to the next image, reusing the SAME (persistent) network. The criterion
is the cumulative-persistence rule of ``condensado.md`` §7, the same one
``gen_evolution.py`` uses: stop when the fraction of the currently-firing set
that has stayed lit (a >= fire_threshold) without interruption for at least
``--persist-patience`` epochs reaches ``--min-persistence``.

    python hebbian/train_sequential.py \
        --dataset data/processed/hlines_set/hlines.npz \
        --run experiments/hlines_seq --min-persistence 0.7 --lr 0.15 --inhib

Saves a resumable ``model.npz`` plus a ``sequential.csv`` (one row per image with
the epoch it converged at) under ``--run``.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

# Support running as a module (python -m hebbian.train_sequential) or as a script.
try:
    from .competitive_net import CompetitiveLayer
    from .train import build_layer, load_dataset
    from .evolution_io import write_sequence
except ImportError:  # pragma: no cover - script execution fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from competitive_net import CompetitiveLayer
    from train import build_layer, load_dataset
    from evolution_io import write_sequence

DEFAULT_RUNS_DIR = "experiments/evolution/runs"


def train_image(layer, x, lr, max_epochs, min_persistence, persist_patience):
    """Present ``x`` repeatedly until convergence.

    Convergence = cumulative persistence (condensado.md §7): fraction of the
    currently-firing neurons that have been lit without interruption for at
    least ``persist_patience`` epochs reaches ``min_persistence``.

    Returns ``(summary, acts)`` where ``acts`` is the list of this image's
    activation vectors, one per epoch trained, for the evolution sequence.
    """
    thr = layer.fire_threshold
    run = np.zeros(layer.n_out, dtype=np.int64)  # per-neuron unbroken firing streak
    converged_at = None
    n_fired = 0
    pers = 0.0
    acts: list[np.ndarray] = []
    for e in range(max_epochs):
        layer.learn_sample(x, lr)           # one presentation of this single image
        layer.epochs_trained += 1
        a = layer.activation(x).astype(np.float32)
        acts.append(a)
        fired = a >= thr
        run = np.where(fired, run + 1, 0)
        n_fired = int(fired.sum())
        pers = (int((run >= persist_patience).sum()) / n_fired) if n_fired else 0.0
        if pers >= min_persistence:
            converged_at = e + 1
            break
    summary = {
        "epochs_used": converged_at if converged_at is not None else max_epochs,
        "converged": converged_at is not None,
        "n_fired": n_fired,
        "persistence": round(pers, 4),
        "winner": int(a.argmax()),
        "winner_activation": round(float(a.max()), 4),
    }
    return summary, acts


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a competitive layer one image at a time")
    ap.add_argument("--dataset", default="data/processed/hlines_set/hlines.npz")
    ap.add_argument("--key", default="images", help="npz key for the images")
    ap.add_argument("--run", default="experiments/hlines_seq", help="output dir")
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--max-epochs", type=int, default=200,
                    help="per-image cap if convergence is never reached")
    ap.add_argument("--min-persistence", type=float, default=0.7,
                    help="stop an image when cumulative persistence reaches this")
    ap.add_argument("--persist-patience", type=int, default=5,
                    help="epochs a neuron must stay lit to count as persistent")
    ap.add_argument("--sequence", default="experiments/evolution/sequence.npz",
                    help="evolution sequence file to (over)write for the webapp viewer")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                    help="archive dir where each run is also saved (never overwritten); "
                         "'' to disable archiving")
    ap.add_argument("--resume", default=None, help="resume from a model.npz")
    # fresh-layer hyperparameters (used only when --resume is not given)
    ap.add_argument("--n-in", type=int, default=784)
    ap.add_argument("--grid", type=int, default=50, help="output map is grid x grid")
    ap.add_argument("--rule", default="above_mean", choices=["above_mean", "softmax", "wta"])
    ap.add_argument("--reinforce-gain", type=float, default=1.0)
    ap.add_argument("--learning-rule", default="gate", choices=["gate", "truth_table"],
                    help="'gate' (base continua) o 'truth_table' (regla por conexión)")
    ap.add_argument("--rule-n", type=float, default=1.1, help="truth_table: factor de aprendizaje disparado")
    ap.add_argument("--rule-m", type=float, default=0.3, help="truth_table: factor de desaprendizaje disparado")
    ap.add_argument("--rule-hr", type=float, default=0.1, help="truth_table: inhibition rate")
    ap.add_argument("--inhib", action="store_true", help="enable lateral inhibition")
    ap.add_argument("--inhib-spacing", type=int, default=5)
    ap.add_argument("--inhib-radius", type=int, default=8)
    ap.add_argument("--inhib-metric", default="cheby", choices=["cheby", "manhattan", "euclid"])
    ap.add_argument("--fire-threshold", type=float, default=0.40)
    ap.add_argument("--inhib-K", type=float, default=0.10)
    ap.add_argument("--inhib-gain", type=float, default=1.5)
    ap.add_argument("--inhib-mode", default="fraction", choices=["fraction", "hinge", "sigmoid"])
    ap.add_argument("--seed", type=int, default=0)
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

    side = int(round(X.shape[1] ** 0.5))
    # Evolution sequence: step 0 = state before training (on image 0), then one
    # step per epoch. ``imgseq`` tracks which image each step was trained on so
    # the viewer's "Fixed image" panel follows the sequential run.
    seq = [layer.activation(X[0]).astype(np.float32)]
    imgseq = [(X[0] * 255).astype(np.uint8)]

    csv_path = os.path.join(args.run, "sequential.csv")
    cols = ["image", "epochs_used", "converged", "n_fired", "persistence",
            "winner", "winner_activation", "total_epochs"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for i in range(len(X)):
            info, acts = train_image(
                layer, X[i], args.lr, args.max_epochs,
                args.min_persistence, args.persist_patience,
            )
            img_u8 = (X[i] * 255).astype(np.uint8)
            for a in acts:
                seq.append(a)
                imgseq.append(img_u8)
            row = {"image": i, "total_epochs": layer.epochs_trained, **info}
            writer.writerow({k: row[k] for k in cols})
            f.flush()
            status = f"converged@{info['epochs_used']}" if info["converged"] \
                else f"NO-CONV (capped {args.max_epochs})"
            print(
                f"image {i:2d}  {status:>22s}  fired={info['n_fired']:4d} "
                f"pers={info['persistence']:.3f} winner={info['winner']:4d} "
                f"act={info['winner_activation']:.3f}"
            )

    model_path = os.path.join(args.run, "model.npz")
    layer.save(model_path)
    print(f"saved {model_path}")
    print(f"saved {csv_path}")

    seq_arr = np.stack(seq)
    steps = len(seq) - 1
    src = os.path.basename(args.resume) if args.resume else "fresh"
    label = (f"seq · {os.path.basename(args.dataset)} · {len(X)}img · "
             f"{steps}ép · lr{args.lr:g} · {layer.learning_rule} · {src}")
    meta = {
        "label": label,
        "script": "train_sequential",
        "dataset": args.dataset.replace("\\", "/"),
        "model_source": (args.resume.replace("\\", "/") if args.resume else "fresh"),
        "learning_rule": layer.learning_rule,
        "lr": args.lr,
        "epochs": steps,
        "nn_epochs": int(layer.epochs_trained),
    }
    out_path, archive = write_sequence(
        args.sequence, args.runs_dir,
        seq=seq_arr, image=imgseq[0], imgseq=np.stack(imgseq), side=side,
        map_h=layer.grid_h, map_w=layer.grid_w,
        image_index=-1, fire_threshold=layer.fire_threshold,
        converged_at=None, meta=meta,
    )
    print(f"saved {out_path}  seq={seq_arr.shape} (steps+1, n_out)")
    if archive:
        print(f"archived run -> {archive}")


if __name__ == "__main__":
    main()
