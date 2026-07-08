"""Reglas de actualización de peso por conexión — tabla de verdad.

Cada peso ``W[j, i]`` conecta la entrada ``i`` con la neurona ``j``. Su cambio
depende de tres señales **binarias** y de cuatro parámetros.

Señales (por conexión):

- ``entrada activa``      → ``x_active[i]``
- ``neurona disparada``   → ``fired[j]``
- ``inhibidora disparada``→ ``inhib_fired[j]``

Parámetros:

- ``lr`` — learning rate (paso base), pasado por época (permite annealing).
- ``n``  — factor de aprendizaje disparado.
- ``m``  — factor de desaprendizaje disparado.
- ``hr`` — inhibition rate (penalización lateral, fija, NO escalada por ``lr``).

Tabla completa ``(entrada, neurona, inhibidora) -> Δpeso``::

    0 0 0 ->  0
    1 0 0 -> +lr
    1 1 0 -> +n*lr
    0 1 0 -> -m*lr
    0 0 1 ->  0
    0 1 1 ->  0
    1 1 1 -> -hr
    1 0 1 ->  0     (no especificado en la tabla; se asume sin cambio)

Idea: con la inhibidora apagada la regla refuerza (``+lr`` / ``+n*lr``) donde hay
entrada y castiga (``-m*lr``) a la neurona que dispara sin entrada. Con la
inhibidora encendida se congela todo salvo el disparo con entrada, que recibe la
penalización lateral ``-hr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TruthTableRule:
    """Regla de la tabla de verdad, vectorizada sobre toda la matriz de pesos."""

    n: float = 1.1   # factor de aprendizaje disparado
    m: float = 0.3   # factor de desaprendizaje disparado
    hr: float = 0.1  # inhibition rate

    def delta(
        self,
        x_active: np.ndarray,
        fired: np.ndarray,
        inhib_fired: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """Devuelve ``ΔW`` de forma ``(n_out, n_in)`` según la tabla.

        Las tres entradas son máscaras booleanas: ``x_active`` de tamaño
        ``n_in``; ``fired`` e ``inhib_fired`` de tamaño ``n_out``.
        """
        X = np.asarray(x_active, dtype=np.float32).ravel()[None, :]    # (1, n_in)
        F = np.asarray(fired, dtype=np.float32).ravel()[:, None]       # (n_out, 1)
        H = np.asarray(inhib_fired, dtype=np.float32).ravel()[:, None]  # (n_out, 1)
        free = 1.0 - H  # sin inhibición

        dW = (X * (1.0 - F) * free) * lr                    # 1 0 0 -> +lr
        dW = dW + (X * F * free) * (self.n * lr)            # 1 1 0 -> +n*lr
        dW = dW + ((1.0 - X) * F * free) * (-self.m * lr)   # 0 1 0 -> -m*lr
        dW = dW + (X * F * H) * (-self.hr)                  # 1 1 1 -> -hr
        return dW.astype(np.float32)
