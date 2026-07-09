"""Shared I/O for the evolution viewer sequences.

A single place so ``gen_evolution.py`` and ``train_sequential.py`` write the
sequence the same way and never diverge. Every run does TWO writes:

1. the **fixed** file (``experiments/evolution/sequence.npz`` by default) — kept
   for backward compatibility (``analyze_convergence.py``, the ``--file`` default
   of the server) and as "the latest run";
2. an **archived, timestamped copy** under ``runs_dir``
   (``experiments/evolution/runs/run_YYYYMMDD-HHMMSS.npz``) that is **never
   overwritten**, so every training stays on disk and can be reviewed later.

The archived copy carries extra descriptive metadata (``label``, ``created``,
``script``, ``dataset``, ``model_source`` …) so the web viewer can list runs with
the most recent first and say which NN each one trained.
"""

from __future__ import annotations

import os
import time

import numpy as np


def _unique_run_path(runs_dir: str) -> str:
    """A fresh, collision-free ``run_<timestamp>.npz`` path inside ``runs_dir``."""
    os.makedirs(runs_dir, exist_ok=True)
    base = time.strftime("run_%Y%m%d-%H%M%S")
    path = os.path.join(runs_dir, base + ".npz")
    i = 2
    while os.path.exists(path):
        path = os.path.join(runs_dir, f"{base}_{i}.npz")
        i += 1
    return path


def write_sequence(
    out_path: str,
    runs_dir: str | None,
    *,
    seq: np.ndarray,
    image: np.ndarray,
    side: int,
    map_h: int,
    map_w: int,
    image_index: int,
    fire_threshold: float,
    converged_at: int | None,
    imgseq: np.ndarray | None = None,
    meta: dict | None = None,
) -> tuple[str, str | None]:
    """Write the fixed sequence file and an archived, timestamped copy.

    Returns ``(out_path, archive_path)`` (``archive_path`` is ``None`` when
    ``runs_dir`` is falsy, i.e. archiving disabled).

    ``seq`` is ``(steps+1, n_out)``; ``image`` is the fixed image (uint8);
    ``imgseq`` (optional, ``(steps+1, n_in)`` uint8) is the per-step image for
    sequential runs. ``meta`` holds string/scalar descriptors for the runs list.
    """
    payload = {
        "seq": np.round(seq, 4).astype(np.float32),
        "image": image.astype(np.uint8),
        "side": np.int64(side),
        "map_h": np.int64(map_h),
        "map_w": np.int64(map_w),
        "steps": np.int64(len(seq) - 1),
        "image_index": np.int64(image_index),
        "fire_threshold": np.float64(fire_threshold),
        "converged_at": np.int64(-1 if converged_at is None else converged_at),
    }
    if imgseq is not None:
        payload["imgseq"] = imgseq.astype(np.uint8)

    # Fixed file: latest run (kept for the --file default / analyze_convergence).
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **payload)

    if not runs_dir:
        return out_path, None

    # Archived copy: same data + descriptive metadata, never overwritten. The
    # metadata (all stored as tiny numpy scalars/strings, no pickle needed) lets
    # the viewer build a "most recent first" runs list.
    meta = dict(meta or {})
    meta.setdefault("created", time.time())
    archive = dict(payload)
    for k, v in meta.items():
        if isinstance(v, str):
            archive[k] = np.str_(v)
        elif isinstance(v, bool):
            archive[k] = np.bool_(v)
        elif isinstance(v, int):
            archive[k] = np.int64(v)
        elif isinstance(v, float):
            archive[k] = np.float64(v)
        else:
            archive[k] = np.str_(str(v))
    archive_path = _unique_run_path(runs_dir)
    np.savez(archive_path, **archive)
    return out_path, archive_path
