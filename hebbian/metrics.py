"""Shared metric helpers (condensado.md §7).

All functions take activations or weights already computed, so they can be
reused by the training loop and the offline analysis scripts.
"""

from __future__ import annotations

import numpy as np


def activations(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cosine activations of every neuron for every input.

    ``W`` rows are assumed unit-norm; ``X`` rows are normalized here.
    Returns ``(n_samples, n_out)``.
    """
    Xu = X.astype(np.float32)
    Xu = Xu / np.maximum(np.linalg.norm(Xu, axis=1, keepdims=True), 1e-8)
    return Xu @ W.T


def winners(A: np.ndarray) -> np.ndarray:
    """Argmax neuron per input."""
    return A.argmax(axis=1)


def dead_units(win_count: np.ndarray) -> int:
    """Neurons that have never won any input."""
    return int((win_count == 0).sum())


def coverage(A: np.ndarray, n_out: int) -> float:
    """Fraction of neurons that win at least one input this epoch."""
    return len(np.unique(winners(A))) / n_out


def unique_winners(A: np.ndarray) -> int:
    return int(len(np.unique(winners(A))))


def mean_winner_activation(A: np.ndarray) -> float:
    """Sharpness: mean cosine activation of the winning neuron."""
    return float(A.max(axis=1).mean())


def mean_fired(A: np.ndarray, threshold: float) -> float:
    """Mean number of neurons firing (a >= threshold) per input."""
    return float((A >= threshold).sum(axis=1).mean())


def dW_rel(W_old: np.ndarray, W_new: np.ndarray) -> float:
    """Relative weight change ||dW|| / ||W|| between two snapshots."""
    denom = np.linalg.norm(W_old)
    return float(np.linalg.norm(W_new - W_old) / max(denom, 1e-8))


def act_cos(A_old: np.ndarray, A_new: np.ndarray) -> float:
    """Mean per-input cosine between the two activation vectors."""
    num = (A_old * A_new).sum(axis=1)
    den = np.linalg.norm(A_old, axis=1) * np.linalg.norm(A_new, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def top_k_jaccard(A_old: np.ndarray, A_new: np.ndarray, k: int = 10) -> float:
    """Mean Jaccard overlap of the top-k active neurons per input."""
    top_old = np.argsort(-A_old, axis=1)[:, :k]
    top_new = np.argsort(-A_new, axis=1)[:, :k]
    out = []
    for o, n in zip(top_old, top_new):
        so, sn = set(o.tolist()), set(n.tolist())
        out.append(len(so & sn) / len(so | sn))
    return float(np.mean(out))


def win_match(A_old: np.ndarray, A_new: np.ndarray) -> float:
    """Fraction of inputs whose argmax winner is identical across snapshots."""
    return float((winners(A_old) == winners(A_new)).mean())


def epoch_metrics(layer, X: np.ndarray) -> dict:
    """Convenience bundle of single-snapshot metrics for a training epoch."""
    A = activations(layer.W, X)
    return {
        "dead_units": dead_units(layer.win_count),
        "coverage": coverage(A, layer.n_out),
        "unique_winners": unique_winners(A),
        "mean_winner_activation": mean_winner_activation(A),
        "mean_fired": mean_fired(A, layer.fire_threshold),
    }
