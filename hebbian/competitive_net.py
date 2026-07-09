"""Competitive Hebbian layer ("forward learning") — pure numpy.

Single-layer unsupervised network that learns by competitive Hebbian
reinforcement plus lateral inhibition, following the spec in ``condensado.md``.

- Weights ``W`` are ``(n_out, n_in)``; every row is kept at unit norm.
- Each input ``x`` is normalized to unit norm, so activation ``a = W @ xu``
  is the cosine similarity in ``[-1, 1]`` and comparable across neurons.
- The base rule ONLY reinforces (gate >= 0). Every weight *reduction* comes
  from an overlapping mesh of lateral inhibitors, which is the only homeostatic
  force keeping the competition from collapsing onto a few hoarding neurons.
"""

from __future__ import annotations

import numpy as np

try:
    from .learning_rules import TruthTableRule
except ImportError:  # pragma: no cover - script execution fallback
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from learning_rules import TruthTableRule


def _normalize_rows(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return ``M`` with each row scaled to unit L2 norm."""
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), eps)


class CompetitiveLayer:
    """A single competitive Hebbian layer with lateral inhibition.

    Parameters mirror the calibrated defaults from ``condensado.md`` §4.
    """

    def __init__(
        self,
        n_in: int = 784,
        n_out: int = 2500,
        *,
        rule: str = "above_mean",
        reinforce_gain: float = 1.0,
        # --- learning rule: "gate" (base) o "truth_table" (condensado §regla) ---
        learning_rule: str = "gate",
        rule_n: float = 1.1,
        rule_m: float = 0.3,
        rule_hr: float = 0.1,
        # --- inhibition mesh (laid out over the grid_h x grid_w output map) ---
        grid_h: int = 50,
        grid_w: int = 50,
        inhib_on: bool = True,
        inhib_spacing: int = 5,
        inhib_offset: int | None = None,
        inhib_radius: int = 8,
        inhib_metric: str = "cheby",
        fire_threshold: float = 0.40,
        inhib_K: float = 0.10,
        inhib_gain: float = 1.5,
        inhib_mode: str = "fraction",
        seed: int = 0,
    ) -> None:
        if grid_h * grid_w != n_out:
            raise ValueError(
                f"grid_h*grid_w ({grid_h*grid_w}) must equal n_out ({n_out})"
            )
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.rule = rule
        self.reinforce_gain = float(reinforce_gain)

        self.learning_rule = learning_rule
        self.rule_n = float(rule_n)
        self.rule_m = float(rule_m)
        self.rule_hr = float(rule_hr)
        self._rule = TruthTableRule(n=self.rule_n, m=self.rule_m, hr=self.rule_hr)

        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.inhib_on = bool(inhib_on)
        self.inhib_spacing = int(inhib_spacing)
        self.inhib_offset = (
            self.inhib_spacing // 2 if inhib_offset is None else int(inhib_offset)
        )
        self.inhib_radius = int(inhib_radius)
        self.inhib_metric = inhib_metric
        self.fire_threshold = float(fire_threshold)
        self.inhib_K = float(inhib_K)
        self.inhib_gain = float(inhib_gain)
        self.inhib_mode = inhib_mode
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.W = _normalize_rows(
            rng.standard_normal((self.n_out, self.n_in)).astype(np.float32)
        )
        self.win_count = np.zeros(self.n_out, dtype=np.int64)
        self.epochs_trained = 0

        self._build_inhib_regions()

    # ------------------------------------------------------------------ setup
    def _build_inhib_regions(self) -> None:
        """Precompute, for each inhibitor center, the neuron indices it covers."""
        spacing, radius = self.inhib_spacing, self.inhib_radius
        offset = self.inhib_offset
        centers = [
            (r, c)
            for r in range(offset, self.grid_h, spacing)
            for c in range(offset, self.grid_w, spacing)
        ]
        idx = np.arange(self.n_out)
        nr, nc = idx // self.grid_w, idx % self.grid_w  # row/col of each neuron
        regions: list[np.ndarray] = []
        for rc, cc in centers:
            dr = np.abs(nr - rc)
            dc = np.abs(nc - cc)
            if self.inhib_metric == "cheby":  # square region
                mask = (dr <= radius) & (dc <= radius)
            elif self.inhib_metric == "manhattan":  # diamond
                mask = (dr + dc) <= radius
            elif self.inhib_metric == "euclid":  # disk
                mask = (dr * dr + dc * dc) <= radius * radius
            else:
                raise ValueError(f"unknown inhib_metric: {self.inhib_metric!r}")
            regions.append(np.nonzero(mask)[0])
        self._inhib_regions = regions

    def rebuild_derived(self) -> None:
        """Rebuild everything derived from the (possibly edited) hyperparameters.

        Used when an existing NN's hyperparameters are changed in place (see
        ``training_manager.update_nn``): re-creates the truth-table rule object
        from ``rule_n/m/hr`` and recomputes the inhibitor regions from the
        current spacing/radius/metric. Does NOT touch ``W`` — structural fields
        (``n_in``, ``n_out``, ``grid_*``) are never edited.
        """
        self._rule = TruthTableRule(n=self.rule_n, m=self.rule_m, hr=self.rule_hr)
        if self.inhib_offset is None:
            self.inhib_offset = self.inhib_spacing // 2
        self._build_inhib_regions()

    # -------------------------------------------------------------- internals
    @staticmethod
    def _normalize_vec(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        x = x.astype(np.float32).ravel()
        return x / max(float(np.linalg.norm(x)), eps)

    def _gate(self, a: np.ndarray) -> np.ndarray:
        """Reinforcement gate >= 0. Only neurons above the mean reinforce."""
        if self.rule == "above_mean":
            return np.tanh(np.maximum(a - a.mean(), 0.0) / (a.std() + 1e-8))
        if self.rule == "softmax":
            z = a - a.max()
            e = np.exp(z)
            return e / (e.sum() + 1e-8)
        if self.rule == "wta":  # winner-take-all
            g = np.zeros_like(a)
            g[int(a.argmax())] = 1.0
            return g
        raise ValueError(f"unknown rule: {self.rule!r}")

    def _inhibition_coeffs(self, a: np.ndarray) -> np.ndarray:
        """Nonneg penalty ``s_i`` to subtract, applied only to fired neurons."""
        fired = a >= self.fire_threshold
        s = np.zeros(self.n_out, dtype=np.float32)
        for idx in self._inhib_regions:
            fr = fired[idx]
            nf = int(fr.sum())
            if nf == 0:
                continue
            frac = nf / idx.size
            if self.inhib_mode == "fraction":
                e = frac - self.inhib_K
            elif self.inhib_mode == "hinge":  # absolute count over threshold
                e = nf - self.inhib_K
            elif self.inhib_mode == "sigmoid":  # softplus of the excess fraction
                e = np.log1p(np.exp(frac - self.inhib_K))
            else:
                raise ValueError(f"unknown inhib_mode: {self.inhib_mode!r}")
            if e > 0:
                s[idx[fr]] += self.inhib_gain * e
        return s

    def _inhibitors_fired(self, a: np.ndarray) -> np.ndarray:
        """Bool ``(n_out,)``: True si algún inhibidor que cubre la neurona dispara.

        Un inhibidor "dispara" cuando la fracción de neuronas encendidas en su
        región supera ``inhib_K`` (mismo umbral que la penalización continua).
        Marca a TODA su región como inhibida, según pide la tabla de verdad.
        """
        fired = a >= self.fire_threshold
        out = np.zeros(self.n_out, dtype=bool)
        for idx in self._inhib_regions:
            nf = int(fired[idx].sum())
            if nf == 0:
                continue
            if (nf / idx.size) - self.inhib_K > 0:
                out[idx] = True
        return out

    # ---------------------------------------------------------------- forward
    def activation(self, x: np.ndarray) -> np.ndarray:
        """Cosine-similarity activation of every neuron for input ``x``."""
        return self.W @ self._normalize_vec(x)

    def learn_sample(self, x: np.ndarray, lr: float) -> np.ndarray:
        """Present one sample and update the weights in place. Returns ``a``."""
        xu = self._normalize_vec(x)
        a = self.W @ xu
        self.win_count[int(a.argmax())] += 1

        if self.learning_rule == "truth_table":
            return self._learn_truth_table(xu, a, lr)

        coef = lr * self.reinforce_gain * self._gate(a)
        if self.inhib_on:
            coef = coef - self._inhibition_coeffs(a)
        idx = np.nonzero(coef)[0]
        if idx.size:
            self.W[idx] += coef[idx][:, None] * xu[None, :]
            self.W[idx] = _normalize_rows(self.W[idx])
        return a

    def _learn_truth_table(self, xu: np.ndarray, a: np.ndarray, lr: float) -> np.ndarray:
        """Per-connection update following ``TruthTableRule``.

        The three binary signals are: input active (``|xu| > eps``), neuron
        fired (``a >= fire_threshold``) and its covering inhibitor firing.
        Rows are renormalized to keep the unit-norm invariant.
        """
        x_active = np.abs(xu) > 1e-8
        fired = a >= self.fire_threshold
        if self.inhib_on:
            inhib_fired = self._inhibitors_fired(a)
        else:
            inhib_fired = np.zeros(self.n_out, dtype=bool)
        dW = self._rule.delta(x_active, fired, inhib_fired, lr)
        rows = np.nonzero(np.any(dW != 0.0, axis=1))[0]
        if rows.size:
            self.W[rows] += dW[rows]
            self.W[rows] = _normalize_rows(self.W[rows])
        return a

    def train_epoch(self, X: np.ndarray, lr: float, rng: np.random.Generator | None = None) -> None:
        """One pass over ``X`` (rows = samples) in shuffled order."""
        order = np.arange(len(X))
        if rng is None:
            rng = np.random.default_rng(self.seed + self.epochs_trained)
        rng.shuffle(order)
        for i in order:
            self.learn_sample(X[i], lr)
        self.epochs_trained += 1

    # ---------------------------------------------------------- persistence
    _HPARAMS = (
        "n_in", "n_out", "rule", "reinforce_gain",
        "learning_rule", "rule_n", "rule_m", "rule_hr",
        "grid_h", "grid_w",
        "inhib_on", "inhib_spacing", "inhib_offset", "inhib_radius",
        "inhib_metric", "fire_threshold", "inhib_K", "inhib_gain",
        "inhib_mode", "seed",
    )

    def save(self, path: str) -> None:
        """Persist weights, win_count, epochs and every hyperparameter."""
        payload = {
            "W": self.W,
            "win_count": self.win_count,
            "epochs_trained": np.int64(self.epochs_trained),
        }
        for name in self._HPARAMS:
            payload[name] = getattr(self, name)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "CompetitiveLayer":
        """Reconstruct a layer from a ``.npz`` (backward compatible defaults)."""
        d = np.load(path, allow_pickle=True)

        def get(key, default):
            return d[key].item() if key in d.files else default

        layer = cls(
            n_in=int(get("n_in", 784)),
            n_out=int(get("n_out", 2500)),
            rule=str(get("rule", "above_mean")),
            reinforce_gain=float(get("reinforce_gain", 1.0)),
            learning_rule=str(get("learning_rule", "gate")),
            rule_n=float(get("rule_n", 1.1)),
            rule_m=float(get("rule_m", 0.3)),
            rule_hr=float(get("rule_hr", 0.1)),
            grid_h=int(get("grid_h", 50)),
            grid_w=int(get("grid_w", 50)),
            inhib_on=bool(get("inhib_on", True)),
            inhib_spacing=int(get("inhib_spacing", 5)),
            inhib_offset=int(get("inhib_offset", 2)),
            inhib_radius=int(get("inhib_radius", 8)),
            inhib_metric=str(get("inhib_metric", "cheby")),
            fire_threshold=float(get("fire_threshold", 0.40)),
            inhib_K=float(get("inhib_K", 0.10)),
            inhib_gain=float(get("inhib_gain", 1.5)),
            inhib_mode=str(get("inhib_mode", "fraction")),
            seed=int(get("seed", 0)),
        )
        layer.W = d["W"].astype(np.float32)
        layer.win_count = d["win_count"].astype(np.int64)
        layer.epochs_trained = int(get("epochs_trained", 0))
        return layer

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:
        return (
            f"CompetitiveLayer(n_in={self.n_in}, n_out={self.n_out}, "
            f"rule={self.rule!r}, learning_rule={self.learning_rule!r}, "
            f"inhib_on={self.inhib_on}, inhib_gain={self.inhib_gain}, "
            f"epochs={self.epochs_trained})"
        )
