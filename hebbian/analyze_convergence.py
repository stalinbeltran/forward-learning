"""Convergence analysis based on firing-set *persistence* (condensado.md §7).

Reads the persistence-trail sequence written by ``gen_evolution.py``
(``experiments/evolution/sequence.npz``) and measures convergence as *how many
neurons stay firing* as training progresses, following two complementary views:

  (a) Step-to-step RETENTION  |S_t & S_{t-1}| / |S_{t-1}|
      Fraction of the firing set that survives from one epoch to the next.
      All neurons rotate -> ~0 (low persistence). Many stay lit -> ->1.
      (Jaccard |S_t & S_{t-1}| / |S_t | S_{t-1}| is reported alongside: same
      idea but also penalizes newcomers.)

  (b) Cumulative PERSISTENCE  |{i: lit >= P epochs in a row}| / |S_t|
      Fraction of the current firing set that has been lit without interruption
      for at least ``patience`` epochs -- i.e. the same neurons endure, not a
      rotating set of constant size (the webapp "persistence trail").

Convergence epoch: first epoch where retention >= ``ret_thr`` for ``patience``
consecutive epochs.

    python hebbian/analyze_convergence.py \
        --file experiments/evolution/sequence.npz --out experiments/evolution/convergence.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def firing_sets(seq: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean matrix (steps, n_out): True where a neuron fires that epoch."""
    return seq >= threshold


def retention(fired: np.ndarray) -> np.ndarray:
    """|S_t & S_{t-1}| / |S_{t-1}| for t>=1 (retention rate). Length steps-1."""
    out = []
    for t in range(1, len(fired)):
        prev, cur = fired[t - 1], fired[t]
        inter = int((prev & cur).sum())
        denom = int(prev.sum())
        out.append(inter / denom if denom else np.nan)
    return np.array(out)


def jaccard(fired: np.ndarray) -> np.ndarray:
    """|S_t & S_{t-1}| / |S_t | S_{t-1}| for t>=1. Length steps-1."""
    out = []
    for t in range(1, len(fired)):
        prev, cur = fired[t - 1], fired[t]
        inter = int((prev & cur).sum())
        union = int((prev | cur).sum())
        out.append(inter / union if union else np.nan)
    return np.array(out)


def cumulative_persistence(fired: np.ndarray, patience: int) -> np.ndarray:
    """Fraction of S_t lit for >= ``patience`` consecutive epochs. Length steps."""
    steps, n_out = fired.shape
    run = np.zeros(n_out, dtype=np.int64)  # current unbroken firing streak
    frac = np.zeros(steps, dtype=np.float64)
    for t in range(steps):
        run = np.where(fired[t], run + 1, 0)
        cur = int(fired[t].sum())
        frac[t] = int((run >= patience).sum()) / cur if cur else np.nan
    return frac


def convergence_epoch(pers: np.ndarray, min_persistence: float) -> int | None:
    """First epoch where cumulative persistence reaches ``min_persistence``.

    ``pers`` is indexed from epoch 0, so the returned index is the epoch itself.
    This mirrors the early-stop rule in ``gen_evolution.py``.
    """
    for t, p in enumerate(pers):
        if not np.isnan(p) and p >= min_persistence:
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Firing-set persistence convergence analysis")
    ap.add_argument("--file", default="experiments/evolution/sequence.npz")
    ap.add_argument("--out", default="experiments/evolution/convergence.png")
    ap.add_argument("--csv", default="experiments/evolution/convergence.csv")
    ap.add_argument("--patience", type=int, default=5, help="consecutive epochs a neuron must stay lit to count as persistent")
    ap.add_argument("--min-persistence", type=float, default=0.7,
                    help="cumulative-persistence fraction that marks convergence (matches gen_evolution.py)")
    args = ap.parse_args()

    d = np.load(args.file)
    seq = d["seq"].astype(np.float32)
    thr = float(d["fire_threshold"]) if "fire_threshold" in d.files else 0.40
    steps = len(seq)

    fired = firing_sets(seq, thr)
    count = fired.sum(axis=1)                 # |S_t|            (steps,)
    ret = retention(fired)                    # retention        (steps-1,)
    jac = jaccard(fired)                      # jaccard          (steps-1,)
    pers = cumulative_persistence(fired, args.patience)  # (steps,)

    conv = convergence_epoch(pers, args.min_persistence)

    # ---- report -----------------------------------------------------------
    print(f"file={args.file}  steps={steps}  n_out={seq.shape[1]}  fire_threshold={thr}")
    print(f"|S_t| firing count: first={count[0]}  min={count.min()}  max={count.max()}  last={count[-1]}")
    print(f"retention  last={ret[-1]:.3f}  mean(last5)={np.nanmean(ret[-5:]):.3f}")
    print(f"jaccard    last={jac[-1]:.3f}")
    print(f"persistence(>= {args.patience} ep) last={pers[-1]:.3f}")
    if conv is not None:
        print(f"CONVERGED at epoch {conv} (cumulative persistence >= {args.min_persistence})")
    else:
        print(f"NOT converged (persistence never reached {args.min_persistence})")

    # ---- csv --------------------------------------------------------------
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", encoding="utf-8") as f:
        f.write("epoch,fired_count,retention,jaccard,persistence\n")
        for t in range(steps):
            r = "" if t == 0 else f"{ret[t-1]:.4f}"
            j = "" if t == 0 else f"{jac[t-1]:.4f}"
            f.write(f"{t},{count[t]},{r},{j},{pers[t]:.4f}\n")
    print(f"wrote {args.csv}")

    # ---- plot -------------------------------------------------------------
    ep = np.arange(steps)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(ep[1:], ret, color="#1f77b4", lw=2, label="Retención  |S_t & S_{t-1}| / |S_{t-1}|")
    ax1.plot(ep[1:], jac, color="#2ca02c", lw=1.5, ls="--", label="Jaccard  |S_t & S_{t-1}| / |S_t | S_{t-1}|")
    ax1.plot(ep, pers, color="#d62728", lw=2.5, label=f"Persistencia acumulada (>= {args.patience} ep)  [criterio]")
    ax1.axhline(args.min_persistence, color="gray", lw=1, ls=":", label=f"umbral persistencia = {args.min_persistence}")
    if conv is not None:
        ax1.axvline(conv, color="black", lw=1.5, alpha=0.6)
        ax1.text(conv, 0.02, f" convergió\n época {conv}", fontsize=9, va="bottom")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel("persistencia / estabilidad")
    ax1.set_title(f"Convergencia por persistencia del conjunto que dispara (umbral disparo = {thr})")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(ep, count, color="#9467bd", lw=2)
    ax2.set_ylabel("|S_t|  neuronas encendidas")
    ax2.set_xlabel("época")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
