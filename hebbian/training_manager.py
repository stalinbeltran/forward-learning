"""In-app NN store + background training manager (for ``webapp_evolution.py``).

Adds "train from the web app" on top of the pure viewer:

- A **store of NNs** under ``experiments/nns/<name>/`` where each NN keeps its
  weights (``model.npz``, which already persists every model hyperparameter via
  ``CompetitiveLayer._HPARAMS``) plus ``train_params.json`` with the *last-used*
  training parameters (lr, epochs, …). Those become the defaults next time, and
  copying an NN copies them too.
- Helpers to **create a fresh NN** (new random weights from chosen
  hyperparameters) or **copy** an existing one (any ``model.npz`` on disk),
  so more elaborate experiments can branch off a previous network.
- A single-job ``TrainingManager`` that trains a store NN over a dataset in a
  **background thread**, tracks live status, and can be **stopped** mid-run. It
  validates dataset/NN compatibility (``dim == n_in``) before starting, writes
  the evolution ``sequence.npz`` (so the viewer shows the new run on Refresh),
  saves the NN back to the store, and refreshes ``lastexperiment/`` so the
  ``/test`` page reflects the just-trained network.

Kept separate from the web server so the training logic stays testable on its
own (see the ``__main__`` self-check at the bottom).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time

import numpy as np

try:
    from .competitive_net import CompetitiveLayer
    from . import metrics as M
    from .evolution_io import write_sequence
except ImportError:  # pragma: no cover - script fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from competitive_net import CompetitiveLayer
    import metrics as M
    from evolution_io import write_sequence


STORE_DIR = "experiments/nns"
SEQUENCE_OUT = "experiments/evolution/sequence.npz"
RUNS_DIR = "experiments/evolution/runs"
LASTEXPERIMENT_DIR = "lastexperiment"
DATA_ROOT = "data"
# Also offer these (non-store) models as copy sources, so prior work is reusable.
EXTRA_MODEL_GLOBS = ("experiments/**/model.npz", "lastexperiment/model.npz")


# Model hyperparameters exposed in the UI, with their defaults. ``grid`` drives
# both grid_h/grid_w (square map) and n_out = grid*grid. These mirror
# CompetitiveLayer.__init__ / train.build_layer.
DEFAULT_MODEL_PARAMS = {
    "n_in": 784,
    "grid": 50,
    "rule": "above_mean",
    "reinforce_gain": 1.0,
    "learning_rule": "gate",
    "rule_n": 1.1,
    "rule_m": 0.3,
    "rule_hr": 0.1,
    "inhib_on": True,
    "inhib_spacing": 5,
    "inhib_radius": 8,
    "inhib_metric": "cheby",
    "fire_threshold": 0.40,
    "inhib_K": 0.10,
    "inhib_gain": 1.5,
    "inhib_mode": "fraction",
    "seed": 0,
}

# Per-training parameters (last-used values are remembered per NN).
DEFAULT_TRAIN_PARAMS = {
    "lr": 0.15,
    "epochs": 80,
    "min_persistence": None,   # None = no early stop
    "persist_patience": 5,
    "image_index": 0,          # which dataset image drives the persistence trail
    "key": "images",
}

_SCALAR_KEYS = (  # scalar/string hyperparameters readable without loading W
    "n_in", "n_out", "rule", "reinforce_gain",
    "learning_rule", "rule_n", "rule_m", "rule_hr",
    "grid_h", "grid_w", "inhib_on", "inhib_spacing", "inhib_offset",
    "inhib_radius", "inhib_metric", "fire_threshold", "inhib_K",
    "inhib_gain", "inhib_mode", "seed", "epochs_trained",
)


# --------------------------------------------------------------- NN building ---
def build_layer_from_params(p: dict) -> CompetitiveLayer:
    """Fresh ``CompetitiveLayer`` from a UI param dict (``grid`` -> square map)."""
    q = dict(DEFAULT_MODEL_PARAMS, **{k: v for k, v in p.items() if v is not None})
    grid = int(q["grid"])
    return CompetitiveLayer(
        n_in=int(q["n_in"]),
        n_out=grid * grid,
        rule=str(q["rule"]),
        reinforce_gain=float(q["reinforce_gain"]),
        learning_rule=str(q["learning_rule"]),
        rule_n=float(q["rule_n"]),
        rule_m=float(q["rule_m"]),
        rule_hr=float(q["rule_hr"]),
        grid_h=grid,
        grid_w=grid,
        inhib_on=bool(q["inhib_on"]),
        inhib_spacing=int(q["inhib_spacing"]),
        inhib_radius=int(q["inhib_radius"]),
        inhib_metric=str(q["inhib_metric"]),
        fire_threshold=float(q["fire_threshold"]),
        inhib_K=float(q["inhib_K"]),
        inhib_gain=float(q["inhib_gain"]),
        inhib_mode=str(q["inhib_mode"]),
        seed=int(q["seed"]),
    )


def _safe_name(name: str) -> str:
    """Sanitize a user NN name to a safe single directory component."""
    name = (name or "").strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    return name or time.strftime("nn_%Y%m%d-%H%M%S")


def _read_scalars(model_path: str) -> dict:
    """Read only the scalar hyperparameters from a ``model.npz`` (skips ``W``)."""
    d = np.load(model_path, allow_pickle=True)

    def g(k, default=None):
        return d[k].item() if k in d.files else default

    n_in = int(g("n_in", 784))
    n_out = int(g("n_out", 2500))
    grid_h = int(g("grid_h", 50))
    grid_w = int(g("grid_w", 50))
    return {
        "n_in": n_in,
        "n_out": n_out,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "side": int(round(n_in ** 0.5)),
        "rule": str(g("rule", "above_mean")),
        "reinforce_gain": float(g("reinforce_gain", 1.0)),
        "learning_rule": str(g("learning_rule", "gate")),
        "rule_n": float(g("rule_n", 1.1)),
        "rule_m": float(g("rule_m", 0.3)),
        "rule_hr": float(g("rule_hr", 0.1)),
        "inhib_on": bool(g("inhib_on", True)),
        "inhib_spacing": int(g("inhib_spacing", 5)),
        "inhib_radius": int(g("inhib_radius", 8)),
        "inhib_metric": str(g("inhib_metric", "cheby")),
        "fire_threshold": float(g("fire_threshold", 0.40)),
        "inhib_K": float(g("inhib_K", 0.10)),
        "inhib_gain": float(g("inhib_gain", 1.5)),
        "inhib_mode": str(g("inhib_mode", "fraction")),
        "seed": int(g("seed", 0)),
        "epochs_trained": int(g("epochs_trained", 0)),
    }


def _train_params_path(nn_dir: str) -> str:
    return os.path.join(nn_dir, "train_params.json")


def read_train_params(nn_dir: str) -> dict:
    """Last-used training params for an NN (defaults if none saved yet)."""
    path = _train_params_path(nn_dir)
    params = dict(DEFAULT_TRAIN_PARAMS)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                params.update(json.load(f))
        except Exception:
            pass
    return params


def write_train_params(nn_dir: str, params: dict) -> None:
    keep = {k: params.get(k, DEFAULT_TRAIN_PARAMS[k]) for k in DEFAULT_TRAIN_PARAMS}
    with open(_train_params_path(nn_dir), "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=2)


def _model_entry(model_path: str, name: str, in_store: bool, nn_dir=None) -> dict:
    info = _read_scalars(model_path)
    entry = {
        "id": model_path.replace("\\", "/"),
        "name": name,
        "in_store": in_store,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(os.path.getmtime(model_path))),
        **info,
    }
    if in_store and nn_dir:
        entry["train_params"] = read_train_params(nn_dir)
    return entry


def list_store_nns(store_dir: str = STORE_DIR) -> list:
    """Trainable NNs in the store (most recently modified first)."""
    out = []
    if os.path.isdir(store_dir):
        for name in sorted(os.listdir(store_dir)):
            nn_dir = os.path.join(store_dir, name)
            model = os.path.join(nn_dir, "model.npz")
            if os.path.isfile(model):
                try:
                    out.append(_model_entry(model, name, True, nn_dir))
                except Exception:
                    continue
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def list_copy_sources(store_dir: str = STORE_DIR) -> list:
    """Every ``model.npz`` usable as a copy source (store + prior experiments)."""
    import glob
    seen, out = set(), []
    for e in list_store_nns(store_dir):
        seen.add(os.path.realpath(e["id"]))
        out.append(e)
    for pat in EXTRA_MODEL_GLOBS:
        for path in sorted(glob.glob(pat, recursive=True)):
            rp = os.path.realpath(path)
            if rp in seen or not os.path.isfile(path):
                continue
            seen.add(rp)
            name = "/".join(os.path.normpath(path).replace("\\", "/").split("/")[-2:])
            try:
                out.append(_model_entry(path, name, False))
            except Exception:
                continue
    return out


def create_new_nn(store_dir: str, name: str, model_params: dict,
                  train_params: dict | None = None) -> dict:
    """Build a fresh NN from ``model_params`` and register it in the store."""
    name = _unique_store_name(store_dir, _safe_name(name))
    nn_dir = os.path.join(store_dir, name)
    os.makedirs(nn_dir, exist_ok=True)
    layer = build_layer_from_params(model_params)
    model_path = os.path.join(nn_dir, "model.npz")
    layer.save(model_path)
    tp = dict(DEFAULT_TRAIN_PARAMS, **(train_params or {}))
    write_train_params(nn_dir, tp)
    return _model_entry(model_path, name, True, nn_dir)


def copy_nn(store_dir: str, name: str, source_path: str) -> dict:
    """Duplicate an existing model (weights + hyperparameters + train params)."""
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"no existe el modelo fuente: {source_path}")
    name = _unique_store_name(store_dir, _safe_name(name))
    nn_dir = os.path.join(store_dir, name)
    os.makedirs(nn_dir, exist_ok=True)
    model_path = os.path.join(nn_dir, "model.npz")
    shutil.copyfile(source_path, model_path)
    # Copy the source's last-used train params if it had any, else defaults.
    src_dir = os.path.dirname(source_path)
    src_tp = read_train_params(src_dir)
    write_train_params(nn_dir, src_tp)
    return _model_entry(model_path, name, True, nn_dir)


def _unique_store_name(store_dir: str, name: str) -> str:
    """Avoid clobbering an existing NN: append -2, -3, … if the name is taken."""
    if not os.path.exists(os.path.join(store_dir, name)):
        return name
    i = 2
    while os.path.exists(os.path.join(store_dir, f"{name}-{i}")):
        i += 1
    return f"{name}-{i}"


def load_dataset(path: str, key: str = "images"):
    """Load an ``.npz`` of images -> float32 rows in [0, 1]; returns (X, key)."""
    d = np.load(path, allow_pickle=False)
    if key not in d.files:  # fall back to the first 2D+ array
        key = next((k for k in d.files if d[k].ndim >= 2), None)
        if key is None:
            raise ValueError(f"{path}: no encuentro un array de imágenes")
    arr = d[key]
    X = arr.reshape(arr.shape[0], -1).astype(np.float32)
    if float(X.max(initial=0.0)) > 1.0:
        X = X / 255.0
    return X, key


# ------------------------------------------------------- background training ---
class TrainingManager:
    """Single-job trainer: runs in a daemon thread and is stoppable."""

    def __init__(self, store_dir=STORE_DIR, sequence_out=SEQUENCE_OUT,
                 runs_dir=RUNS_DIR, lastexperiment_dir=LASTEXPERIMENT_DIR):
        self.store_dir = store_dir
        self.sequence_out = sequence_out
        self.runs_dir = runs_dir
        self.lastexperiment_dir = lastexperiment_dir
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._status = {"state": "idle"}

    # -- status ---------------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    # -- control --------------------------------------------------------------
    def start(self, nn_path, dataset, train_params) -> dict:
        """Validate and launch a training run. Returns {ok, ...}."""
        if self.is_running():
            return {"ok": False, "error": "ya hay un entrenamiento en curso"}
        if not os.path.isfile(nn_path):
            return {"ok": False, "error": f"no existe la NN: {nn_path}"}
        tp = dict(DEFAULT_TRAIN_PARAMS, **(train_params or {}))
        key = tp.get("key", "images")
        try:
            X, key = load_dataset(dataset, key)
        except Exception as e:
            return {"ok": False, "error": f"no pude leer el dataset: {e}"}
        try:
            scalars = _read_scalars(nn_path)
        except Exception as e:
            return {"ok": False, "error": f"no pude leer la NN: {e}"}
        if X.shape[1] != scalars["n_in"]:
            return {"ok": False, "error": (
                f"incompatible: dataset dim {X.shape[1]} != n_in {scalars['n_in']} "
                "de la NN")}

        self._stop.clear()
        self._set(state="running", nn=nn_path.replace("\\", "/"),
                  dataset=dataset.replace("\\", "/"), epoch=0,
                  total=int(tp["epochs"]), log=[], converged_at=None,
                  error=None, started=time.strftime("%H:%M:%S"))
        self._thread = threading.Thread(
            target=self._run, args=(nn_path, dataset, X, key, tp), daemon=True)
        self._thread.start()
        return {"ok": True, "status": self.status()}

    def stop(self) -> dict:
        if not self.is_running():
            return {"ok": False, "error": "no hay entrenamiento en curso"}
        self._stop.set()
        return {"ok": True}

    # -- worker ---------------------------------------------------------------
    def _log(self, line: str):
        with self._lock:
            self._status.setdefault("log", []).append(line)
            self._status["log"] = self._status["log"][-200:]  # cap history

    def _run(self, nn_path, dataset, X, key, tp):
        try:
            layer = CompetitiveLayer.load(nn_path)
            epochs = int(tp["epochs"])
            lr = float(tp["lr"])
            min_pers = tp.get("min_persistence")
            min_pers = None if min_pers in (None, "", "None") else float(min_pers)
            patience = int(tp["persist_patience"])
            img_idx = int(tp["image_index"]) % len(X)
            fixed = X[img_idx]
            thr = layer.fire_threshold
            rng = np.random.default_rng(layer.seed + layer.epochs_trained)

            a0 = layer.activation(fixed).astype(np.float32)
            seq = [a0]
            run = (a0 >= thr).astype(np.int64)
            converged_at = None

            for e in range(epochs):
                if self._stop.is_set():
                    self._log(f"detenido por el usuario en la época {e}")
                    break
                layer.train_epoch(X, lr, rng=rng)
                a = layer.activation(fixed).astype(np.float32)
                seq.append(a)
                fired = a >= thr
                run = np.where(fired, run + 1, 0)
                n_fired = int(fired.sum())
                pers = (int((run >= patience).sum()) / n_fired) if n_fired else 0.0
                m = M.epoch_metrics(layer, X)
                self._set(epoch=e + 1, persistence=round(pers, 3),
                          n_fired=n_fired, coverage=round(m["coverage"], 3),
                          unique_winners=m["unique_winners"])
                self._log(
                    f"época {e+1}/{epochs}  fired={n_fired}  "
                    f"persist={pers:.3f}  cov={m['coverage']:.3f}  "
                    f"uniq={m['unique_winners']}")
                if min_pers is not None and pers >= min_pers:
                    converged_at = e + 1
                    self._log(f"CONVERGIÓ en la época {converged_at} "
                              f"(persistencia {pers:.3f} >= {min_pers})")
                    break

            steps = len(seq) - 1
            self._finalize(layer, nn_path, dataset, key, np.stack(seq), fixed,
                           img_idx, lr, steps, converged_at, tp)
        except Exception as e:  # surface the failure in the status
            self._set(state="error", error=str(e))
            self._log(f"ERROR: {e}")

    def _finalize(self, layer, nn_path, dataset, key, seq, fixed, img_idx,
                  lr, steps, converged_at, tp):
        # 1) save the (continued) NN back to its store slot
        layer.save(nn_path)
        nn_dir = os.path.dirname(nn_path)
        write_train_params(nn_dir, tp)

        # 2) write the evolution sequence (fixed file + archived run) so the
        #    viewer shows this run on Refresh (see CLAUDE.md: always emit seq).
        side = int(round(layer.n_in ** 0.5))
        nn_name = os.path.basename(nn_dir)
        stopped = self._stop.is_set()
        label = (f"app · {nn_name} · {os.path.basename(dataset)} · img{img_idx} · "
                 f"{steps}ép · lr{lr:g} · {layer.learning_rule}"
                 + (" · STOP" if stopped else ""))
        meta = {
            "label": label,
            "script": "webapp_train",
            "dataset": dataset.replace("\\", "/"),
            "model_source": nn_path.replace("\\", "/"),
            "learning_rule": layer.learning_rule,
            "lr": lr,
            "epochs": steps,
            "nn_epochs": int(layer.epochs_trained),
        }
        out_path, archive = write_sequence(
            self.sequence_out, self.runs_dir,
            seq=seq, image=(fixed * 255).astype(np.uint8), side=side,
            map_h=layer.grid_h, map_w=layer.grid_w,
            image_index=img_idx, fire_threshold=layer.fire_threshold,
            converged_at=converged_at, meta=meta,
        )

        # 3) refresh lastexperiment/ so /test uses the just-trained NN
        self._refresh_lastexperiment(nn_path, nn_name, dataset, meta, steps)

        self._set(state=("stopped" if stopped else "done"),
                  converged_at=converged_at, sequence=out_path.replace("\\", "/"),
                  archive=(archive.replace("\\", "/") if archive else None),
                  epochs_trained=int(layer.epochs_trained))
        self._log(f"guardado modelo -> {nn_path.replace(os.sep, '/')}")
        if archive:
            self._log(f"secuencia archivada -> {archive.replace(os.sep, '/')}")

    def _refresh_lastexperiment(self, nn_path, nn_name, dataset, meta, steps):
        try:
            os.makedirs(self.lastexperiment_dir, exist_ok=True)
            shutil.copyfile(nn_path, os.path.join(self.lastexperiment_dir, "model.npz"))
            with open(os.path.join(self.lastexperiment_dir, "META.txt"),
                      "w", encoding="utf-8") as f:
                f.write(f"NN: {nn_name}\n")
                f.write(f"origen: {nn_path.replace(os.sep, '/')}\n")
                f.write(f"dataset: {dataset.replace(os.sep, '/')}\n")
                f.write(f"regla: {meta['learning_rule']}  lr: {meta['lr']}  "
                        f"épocas de esta corrida: {steps}\n")
                f.write(f"épocas totales de la NN: {meta['nn_epochs']}\n")
                f.write(f"entrenado desde la app: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            self._log(f"aviso: no pude refrescar lastexperiment/: {e}")


if __name__ == "__main__":  # tiny self-check on a synthetic dataset
    import tempfile
    tmp = tempfile.mkdtemp(prefix="nnstore_")
    store = os.path.join(tmp, "nns")
    # small dataset: 8 random 4x4 images
    ds = os.path.join(tmp, "toy.npz")
    rng = np.random.default_rng(0)
    imgs = (rng.random((8, 4, 4)) * 255).astype(np.uint8)
    np.savez(ds, images=imgs)

    e = create_new_nn(store, "toy", {"n_in": 16, "grid": 6, "inhib_on": True,
                                     "inhib_spacing": 3, "inhib_radius": 2})
    print("created:", e["name"], "n_in", e["n_in"], "n_out", e["n_out"])
    mgr = TrainingManager(store_dir=store,
                          sequence_out=os.path.join(tmp, "seq.npz"),
                          runs_dir=os.path.join(tmp, "runs"),
                          lastexperiment_dir=os.path.join(tmp, "last"))
    r = mgr.start(e["id"], ds, {"epochs": 5, "lr": 0.2})
    print("start:", r["ok"])
    while mgr.is_running():
        time.sleep(0.05)
    print("final status state:", mgr.status()["state"],
          "epoch:", mgr.status().get("epoch"))
    cp = copy_nn(store, "toy-copy", e["id"])
    print("copied:", cp["name"], "epochs_trained", cp["epochs_trained"])
    print("store nns:", [x["name"] for x in list_store_nns(store)])
    print("OK")
