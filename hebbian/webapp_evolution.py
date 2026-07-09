"""Persistence-trail viewer — pure server, independent of training.

Reads a sequence file produced by ``gen_evolution.py`` and serves it. The file
is re-read from disk whenever its modification time changes, so re-running the
generator and clicking "Refrescar" in the page shows the new training WITHOUT
restarting this server.

    # 1) generate (or regenerate) the sequence
    python hebbian/gen_evolution.py --dataset data/processed/hline/hline.npz \
        --image-index 0 --epochs 80 --lr 0.15 --inhib
    # 2) serve it (leave running; just Refresh after each regenerate)
    python hebbian/webapp_evolution.py

The page replays, epoch by epoch, how the fixed image's firing pattern changes
as the network keeps learning, plus a per-neuron leaky "persistence trail".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from urllib.parse import urlparse, parse_qs

try:
    from .competitive_net import CompetitiveLayer
    from . import metrics as M
except ImportError:  # pragma: no cover - script fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from competitive_net import CompetitiveLayer
    import metrics as M


DEFAULT_FILE = "experiments/evolution/sequence.npz"
# Cada entrenamiento se archiva aqui (gen_evolution.py / train_sequential.py) con
# un nombre unico por timestamp, sin sobrescribir. El visor lista estos runs con
# el mas reciente arriba y sirve el que elijas, sin reiniciar el servidor.
DEFAULT_RUNS_DIR = "experiments/evolution/runs"
# "NN actual": el experimento mas reciente vive siempre en lastexperiment/ (ver
# CLAUDE.md). No se fija: se recarga por mtime, asi que si el modelo cambia en
# disco la pagina de pruebas usa el nuevo sin reiniciar el servidor.
DEFAULT_MODEL = "lastexperiment/model.npz"
DATA_ROOT = "data"


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Persistence trail</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0d12; color:#e6e9ef; font:14px/1.5 system-ui, sans-serif; }
  header { padding:14px 18px; border-bottom:1px solid #222836;
           display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .ctl { display:flex; gap:8px; align-items:center; }
  label { color:#9aa4b6; font-size:12px; }
  input[type=range] { width:130px; accent-color:#6ea8fe; }
  button, a.btn, select { background:#161b26; color:#e6e9ef; border:1px solid #2a3242;
           border-radius:6px; padding:5px 10px; font:inherit; }
  button:hover, a.btn:hover, select:hover { border-color:#3a4256; }
  a.btn { text-decoration:none; display:inline-block; }
  select { max-width:420px; }
  #testlink { border-color:#2f6feb; color:#9ec1ff; }
  #applylink { border-color:#2f8f5f; color:#8fe0b0; }
  #refresh { border-color:#2f6feb; color:#9ec1ff; }
  #status { color:#6b7488; font-size:12px; font-variant-numeric:tabular-nums; }
  #apnote { color:#f0b849; font-size:12px; font-weight:600; }
  main { display:flex; gap:36px; justify-content:center; align-items:flex-start;
         padding:30px 18px; flex-wrap:wrap; }
  .col { text-align:center; }
  .col h2 { font-size:12px; color:#9aa4b6; font-weight:600; margin:0 0 10px;
            text-transform:uppercase; letter-spacing:.08em; }
  canvas { image-rendering:pixelated; background:#000; border:1px solid #222836; border-radius:4px; }
  .val { font-variant-numeric:tabular-nums; color:#6ea8fe; }
  .fired { color:#9aa4b6; font-size:12px; margin-top:8px; }
</style></head>
<body>
<header>
  <h1>persistence trail &middot; epoch <span class="val" id="ep">0</span>/<span id="steps">?</span></h1>
  <div class="ctl"><button id="play">Pause</button><button id="skip">Saltar &#9197;</button>
    <button id="reset">Reset trail</button><button id="refresh">Refrescar</button>
    <a class="btn" id="testlink" href="/test">Probar NN &#128300;</a>
    <a class="btn" id="applylink" href="/apply">Aplicar set &#127919;</a></div>
  <div class="ctl"><label>entrenamiento</label><select id="run" title="mas reciente arriba"></select></div>
  <div class="ctl"><label>ms/step <span class="val" id="msv">40</span></label>
    <input type="range" id="ms" min="30" max="1500" step="10" value="40"></div>
  <div class="ctl"><label>trail speed <span class="val" id="rtv">0.30</span></label>
    <input type="range" id="rt" min="0.02" max="1" step="0.02" value="0.30"></div>
  <div class="ctl"><label>&theta; <span class="val" id="thv">0.40</span></label>
    <input type="range" id="th" min="0" max="1" step="0.01" value="0.40"></div>
  <div class="ctl"><label><input type="checkbox" id="autopause" checked>
    pausar antes de cambiar de imagen</label></div>
  <span id="apnote"></span>
  <span id="status"></span>
</header>
<main>
  <div class="col"><h2>Fixed image</h2>
    <canvas id="img" width="28" height="28" style="width:168px;height:168px"></canvas></div>
  <div class="col"><h2>Firing (this epoch)</h2>
    <canvas id="now" width="50" height="50" style="width:300px;height:300px"></canvas>
    <div class="fired">fired: <span id="fired">0</span></div></div>
  <div class="col"><h2>Persistence trail</h2>
    <canvas id="trail" width="50" height="50" style="width:300px;height:300px"></canvas>
    <div class="fired">bright = persistently active</div></div>
</main>
<script>
let META=null, SEQ=null, IMGSEQ=null, IMG=null, step=0, trail=null, timer=null, playing=true, thTouched=false;
let blockEnd=null, heldAtBoundary=false;  // auto-pause on the last frame of each trained image
let RUNS=[];  // archived trainings, most recent first (populated from /api/runs)
const el = id => document.getElementById(id);

function computeBlockEnds(){
  // blockEnd[s] = true when step s is the final frame before the input image changes
  // (or the very last step). Only meaningful when imgseq (per-step image) is present.
  const n = SEQ.length;
  blockEnd = new Array(n).fill(false);
  if(!IMGSEQ) return;
  for(let s=0; s<n; s++){
    if(s === n-1){ blockEnd[s]=true; continue; }
    const a=IMGSEQ[s], b=IMGSEQ[s+1];
    for(let k=0; k<a.length; k++){ if(a[k]!==b[k]){ blockEnd[s]=true; break; } }
  }
}

function drawImg(canvas, arr, side){
  const ctx=canvas.getContext('2d'), im=ctx.createImageData(side,side);
  for(let i=0;i<arr.length;i++){const v=arr[i];im.data[i*4]=v;im.data[i*4+1]=v;im.data[i*4+2]=v;im.data[i*4+3]=255;}
  ctx.putImageData(im,0,0);
}
function drawNow(act, th){
  const ctx=el('now').getContext('2d'), n=act.length, im=ctx.createImageData(META.map_w,META.map_h);
  let fired=0;
  for(let i=0;i<n;i++){ const on=act[i]>=th; if(on)fired++;
    const v=on?255:0; im.data[i*4]=v; im.data[i*4+1]=v; im.data[i*4+2]=on?255:0; im.data[i*4+3]=255; }
  ctx.putImageData(im,0,0); return fired;
}
function updateTrail(act, th, rate){
  const n=act.length;
  for(let i=0;i<n;i++){ const on=act[i]>=th;
    trail[i] += on ? rate*(1-trail[i]) : -rate*trail[i]; }
  const ctx=el('trail').getContext('2d'), im=ctx.createImageData(META.map_w,META.map_h);
  for(let i=0;i<n;i++){ const v=Math.round(trail[i]*255);
    im.data[i*4]=v; im.data[i*4+1]=v; im.data[i*4+2]=Math.min(255,v+20); im.data[i*4+3]=255; }
  ctx.putImageData(im,0,0);
}
function resetTrail(){ trail=new Float32Array(META.map_w*META.map_h); }
function integTrail(s, th, rate){  // advance the trail integrator over step s WITHOUT drawing
  const act=SEQ[s];
  for(let i=0;i<act.length;i++){ const on=act[i]>=th;
    trail[i] += on ? rate*(1-trail[i]) : -rate*trail[i]; }
}
function nextInputTarget(){
  // first step of the NEXT trained image; without imgseq, jump to the final state
  if(!IMGSEQ || !blockEnd) return SEQ.length-1;
  let e=step; while(e<SEQ.length-1 && !blockEnd[e]) e++;  // end of the current block
  return (e+1) % SEQ.length;
}
function jumpTo(target){
  // Jump to `target` but keep the persistence trail faithful by replaying the
  // integrator over the skipped frames (render() integrates the target itself).
  const rate=parseFloat(el('rt').value), th=parseFloat(el('th').value);
  if(target<=step){ resetTrail(); for(let s=1;s<target;s++) integTrail(s,th,rate); }
  else { for(let s=step+1;s<target;s++) integTrail(s,th,rate); }
  step=target; heldAtBoundary=false; el('apnote').textContent='';
  render();
}
function render(){
  const th=parseFloat(el('th').value), rate=parseFloat(el('rt').value);
  el('ep').textContent=step;
  if(IMGSEQ) drawImg(el('img'), IMGSEQ[step], META.side);  // fixed panel follows the trained image
  el('fired').textContent=drawNow(SEQ[step], th);
  updateTrail(SEQ[step], th, rate);
}
function advance(){ step=(step+1)%SEQ.length; if(step===0) resetTrail(); render(); }
function setPlaying(p){ playing=p; el('play').textContent=p?'Pause':'Play'; }
function tick(){
  if(playing){
    // At a block end with auto-pause on, stop on this final frame until the user
    // clicks Play; heldAtBoundary lets that click cross into the next image.
    if(el('autopause').checked && blockEnd && blockEnd[step] && !heldAtBoundary){
      setPlaying(false);
      heldAtBoundary=true;
      el('apnote').textContent='⏸ estado final · Play para la siguiente imagen';
    } else {
      advance();
      heldAtBoundary=false;
      el('apnote').textContent='';
    }
  }
  timer=setTimeout(tick, parseInt(el('ms').value));
}
el('play').onclick=()=>{
  const next=!playing;
  // Resuming while parked on a boundary: allow this Play to cross it once.
  if(next && el('autopause').checked && blockEnd && blockEnd[step]) heldAtBoundary=true;
  if(next) el('apnote').textContent='';
  setPlaying(next);
};
el('skip').onclick=()=>jumpTo(nextInputTarget());
el('reset').onclick=()=>{resetTrail(); render();};
el('refresh').onclick=()=>refresh();
el('run').onchange=()=>load(true);
el('ms').oninput=()=>el('msv').textContent=el('ms').value;
el('rt').oninput=()=>el('rtv').textContent=parseFloat(el('rt').value).toFixed(2);
el('th').oninput=()=>{thTouched=true; el('thv').textContent=parseFloat(el('th').value).toFixed(2);};

async function loadRuns(){
  // Refresh the runs list (most recent first), keeping the current selection if
  // it still exists; otherwise select the newest.
  const prev = el('run').value;
  try{ RUNS = (await (await fetch('/api/runs')).json()).runs || []; }
  catch(e){ RUNS=[]; }
  const sel = el('run'); sel.innerHTML='';
  RUNS.forEach((r,i)=>{
    const o=document.createElement('option'); o.value=r.file||r.path;
    o.textContent=(i===0?'★ ':'')+r.label+'  ['+r.mtime+']';
    sel.appendChild(o);
  });
  if(RUNS.length){
    const keep = RUNS.find(r=>(r.file||r.path)===prev);
    sel.value = keep ? prev : (RUNS[0].file||RUNS[0].path);
  }
}
function selectedFile(){ return el('run').value || ''; }
async function refresh(){ await loadRuns(); await load(true); }

async function load(manual){
  try{
    const f = selectedFile();
    const q = f ? ('?file='+encodeURIComponent(f)) : '';
    const meta=await (await fetch('/api/meta'+q)).json();
    if(meta.error){ el('status').textContent='sin datos: '+meta.error; return; }
    const seq=(await (await fetch('/api/seq'+q)).json()).seq;
    const img=(await (await fetch('/api/image'+q)).json()).image;
    const imgseq=meta.has_imgseq ? (await (await fetch('/api/imgseq'+q)).json()).imgseq : null;
    META=meta; SEQ=seq; IMG=img; IMGSEQ=imgseq; step=0;
    computeBlockEnds(); heldAtBoundary=false; el('apnote').textContent='';
    el('steps').textContent=SEQ.length-1;
    if(!thTouched){ el('th').value=META.fire_threshold; el('thv').textContent=META.fire_threshold.toFixed(2); }
    drawImg(el('img'), IMGSEQ?IMGSEQ[0]:img, META.side); resetTrail(); render();
    const t=new Date().toLocaleTimeString();
    el('status').textContent=(manual?'refrescado ':'cargado ')+t+' · '+META.mtime;
  }catch(e){ el('status').textContent='error al cargar: '+e; }
}
(async()=>{ await loadRuns(); await load(false); tick(); })();
</script></body></html>"""


def _load_file(path):
    """Read the sequence file; return (meta, seq, image, imgseq).

    ``imgseq`` (one image per step) is optional: sequential runs
    (``train_sequential.py``) write it so the fixed-image panel follows the
    image being trained; single-image runs (``gen_evolution.py``) omit it.
    """
    d = np.load(path, allow_pickle=False)
    seq = d["seq"].astype(np.float32)
    image = d["image"].astype(np.uint8)
    imgseq = d["imgseq"].astype(np.uint8) if "imgseq" in d.files else None
    meta = {
        "steps": int(d["steps"]),
        "side": int(d["side"]),
        "map_h": int(d["map_h"]),
        "map_w": int(d["map_w"]),
        "fire_threshold": float(d["fire_threshold"]),
        "image_index": int(d["image_index"]),
        "has_imgseq": imgseq is not None,
        "mtime": __import__("time").strftime(
            "%H:%M:%S", __import__("time").localtime(os.path.getmtime(path))
        ),
    }
    return meta, seq, image, imgseq


def _run_entry(path):
    """Lightweight descriptor of one sequence file for the runs list.

    ``np.load`` is lazy, so reading only the small scalar/string fields does not
    load the big ``seq`` array. Older files (pre-archive) miss the descriptive
    metadata; a label is synthesised from whatever is present.
    """
    d = np.load(path, allow_pickle=False)

    def g(k, default=None):
        return d[k].item() if k in d.files else default

    mtime = os.path.getmtime(path)
    label = g("label")
    if not label:
        ii = int(g("image_index", 0))
        steps = int(g("steps", 0))
        who = "secuencial" if ii == -1 else f"img{ii}"
        label = f"{who} · {steps}ép"
    return {
        "path": path.replace("\\", "/"),
        "label": str(label),
        "created": float(g("created", mtime)),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
        "script": str(g("script", "")),
        "dataset": str(g("dataset", "")),
        "model_source": str(g("model_source", "")),
        "learning_rule": str(g("learning_rule", "")),
        "steps": int(g("steps", 0)),
    }


def scan_runs(runs_dir, default_file):
    """List archived runs (most recent first).

    Sorted by the embedded ``created`` timestamp descending, so the last
    training — and thus the last-trained NN — is at the top. The legacy fixed
    file (``default_file``) is a mere copy of the latest run, so it is listed
    only as a fallback when there are no archived runs at all.
    """
    runs = []
    for path in glob.glob(os.path.join(runs_dir, "*.npz")):
        try:
            runs.append(_run_entry(path))
        except Exception:
            continue
    if not runs and os.path.exists(default_file):
        try:
            runs.append(_run_entry(default_file))
        except Exception:
            pass
    runs.sort(key=lambda r: r["created"], reverse=True)  # most recent first
    return runs


# --------------------------------------------------------------- test page ---
# The "Probar NN" page loads the CURRENT model (default lastexperiment/model.npz),
# lists every dataset under data/, validates each against the network's input
# dimension, and reports analysis of the model's response over the chosen sets.

_model_cache = {"key": None, "layer": None}


def load_current_model(model_path):
    """(Re)load the current model, refreshing when the file changes on disk.

    Returns ``(layer, info)``; ``layer`` is ``None`` when the file is missing.
    The model is never fixed: keyed by ``mtime`` so a retrain is picked up on
    the next request without restarting the server.
    """
    if not os.path.exists(model_path):
        return None, {"error": f"no existe {model_path} (corre un entrenamiento primero)"}
    mtime = os.path.getmtime(model_path)
    key = (model_path, mtime)
    if _model_cache["key"] != key:
        _model_cache["layer"] = CompetitiveLayer.load(model_path)
        _model_cache["key"] = key
    layer = _model_cache["layer"]
    info = {
        "path": model_path.replace("\\", "/"),
        "n_in": layer.n_in,
        "n_out": layer.n_out,
        "grid_h": layer.grid_h,
        "grid_w": layer.grid_w,
        "side": int(round(layer.n_in ** 0.5)),
        "fire_threshold": float(layer.fire_threshold),
        "learning_rule": layer.learning_rule,
        "epochs_trained": int(layer.epochs_trained),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
    }
    return layer, info


def _dataset_key(d):
    """Pick the image array inside an ``.npz``: prefer ``images``, else first 2D+."""
    if "images" in d.files:
        return "images"
    for k in d.files:
        if d[k].ndim >= 2:
            return k
    return None


def scan_datasets(root=DATA_ROOT, n_in=None):
    """List every ``.npz`` under ``root`` with its shape and (if ``n_in`` given)
    whether it is compatible with the network input dimension."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
        try:
            d = np.load(path, allow_pickle=False)
        except Exception:
            continue
        key = _dataset_key(d)
        if key is None:
            continue
        arr = d[key]
        if arr.ndim < 2:
            continue
        dim = int(np.prod(arr.shape[1:]))
        item = {
            "path": path.replace("\\", "/"),
            "key": key,
            "n": int(arr.shape[0]),
            "dim": dim,
            "shape": [int(s) for s in arr.shape],
        }
        if n_in is not None:
            item["compatible"] = dim == int(n_in)
        out.append(item)
    return out


def _load_images(path):
    d = np.load(path, allow_pickle=False)
    key = _dataset_key(d)
    arr = d[key]
    X = arr.reshape(arr.shape[0], -1).astype(np.float32)
    return X, key


def evaluate(layer, paths, threshold):
    """Run the current model over the selected datasets and bundle the analysis.

    Every dataset is validated against ``layer.n_in`` first; incompatible ones
    are reported and excluded (never fed to the network).
    """
    datasets_info = []
    incompatible = []
    Xs, src = [], []
    for i, path in enumerate(paths):
        try:
            X, key = _load_images(path)
        except Exception as e:
            datasets_info.append({"path": path, "error": str(e), "compatible": False})
            continue
        compatible = X.shape[1] == layer.n_in
        info = {"path": path.replace("\\", "/"), "key": key,
                "n": int(X.shape[0]), "dim": int(X.shape[1]), "compatible": compatible}
        if not compatible:
            incompatible.append(info["path"])
            datasets_info.append(info)
            continue
        A = M.activations(layer.W, X)
        w = A.argmax(1)
        wa = A.max(1)
        nf = (A >= threshold).sum(1)
        info.update({
            "unique_winners": int(len(np.unique(w))),
            "mean_winner_act": float(wa.mean()),
            "mean_fired": float(nf.mean()),
        })
        datasets_info.append(info)
        Xs.append(X)
        src.extend([i] * X.shape[0])

    if not Xs:
        return {"ok": False, "threshold": threshold, "datasets": datasets_info,
                "incompatible": incompatible,
                "error": "ningun dataset compatible seleccionado"}

    Xall = np.concatenate(Xs, 0)
    A = M.activations(layer.W, Xall)
    w = A.argmax(1)
    wa = A.max(1)
    fired = A >= threshold
    nf = fired.sum(1)
    win_count = np.bincount(w, minlength=layer.n_out)
    fire_fraction = fired.mean(0)
    combined = {
        "n_inputs": int(Xall.shape[0]),
        "unique_winners": int(len(np.unique(w))),
        "coverage": float(len(np.unique(w)) / layer.n_out),
        "mean_winner_act": float(wa.mean()),
        "mean_fired": float(nf.mean()),
        "dead_selected": int((win_count == 0).sum()),
        "win_count": win_count.astype(int).tolist(),
        "fire_fraction": np.round(fire_fraction, 4).tolist(),
        "per_input": {
            "winner": w.astype(int).tolist(),
            "winner_act": np.round(wa, 4).tolist(),
            "n_fired": nf.astype(int).tolist(),
            "src": [int(s) for s in src],
        },
    }
    return {"ok": True, "threshold": threshold, "datasets": datasets_info,
            "incompatible": incompatible, "combined": combined,
            "map_h": layer.grid_h, "map_w": layer.grid_w}


def apply_dataset(layer, path):
    """Run the frozen model over every input of one dataset, for the /apply
    replay. Validates compatibility first; returns per-input activation maps
    (``seq``) and the input images (``imgseq``) in the same shape the viewer
    already knows how to play back. Threshold is applied client-side."""
    try:
        X, key = _load_images(path)
    except Exception as e:
        return {"ok": False, "error": f"no se pudo leer {path}: {e}"}
    if X.shape[1] != layer.n_in:
        return {"ok": False,
                "error": f"incompatible: dim {X.shape[1]} != n_in {layer.n_in}"}
    A = M.activations(layer.W, X)
    side = int(round(layer.n_in ** 0.5))
    disp = X * 255.0 if float(X.max(initial=0.0)) <= 1.0 else X
    imgseq = np.clip(disp, 0, 255).astype(np.uint8)
    return {
        "ok": True,
        "path": path.replace("\\", "/"),
        "key": key,
        "n": int(X.shape[0]),
        "side": side,
        "map_h": layer.grid_h,
        "map_w": layer.grid_w,
        "fire_threshold": float(layer.fire_threshold),
        "seq": np.round(A, 4).tolist(),
        "imgseq": imgseq.tolist(),
    }


TEST_PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Probar NN actual</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0d12; color:#e6e9ef; font:14px/1.5 system-ui, sans-serif; }
  header { padding:14px 18px; border-bottom:1px solid #222836;
           display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  h2 { font-size:13px; color:#9aa4b6; font-weight:600; margin:0 0 10px;
       text-transform:uppercase; letter-spacing:.06em; }
  h3 { font-size:12px; color:#9aa4b6; font-weight:600; margin:0 0 8px; }
  button, a.btn { background:#161b26; color:#e6e9ef; border:1px solid #2a3242;
           border-radius:6px; padding:5px 10px; font:inherit; cursor:pointer; }
  button:hover, a.btn:hover { border-color:#3a4256; }
  a.btn { text-decoration:none; display:inline-block; }
  button.primary { border-color:#2f6feb; color:#9ec1ff; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .dim { color:#6b7488; font-size:12px; }
  .val { font-variant-numeric:tabular-nums; color:#6ea8fe; }
  main { padding:22px 18px; display:flex; flex-direction:column; gap:22px; max-width:1000px; }
  .panel { border:1px solid #222836; border-radius:8px; padding:18px; background:#0f131b; }
  .ctl { display:flex; gap:8px; align-items:center; margin:14px 0; }
  input[type=range] { width:150px; accent-color:#6ea8fe; }
  .dsrow { display:flex; gap:10px; align-items:center; padding:6px 8px; border-radius:6px; }
  .dsrow:hover { background:#141a24; }
  .dsrow label { cursor:pointer; }
  .badge { font-size:11px; padding:2px 7px; border-radius:10px; font-weight:600; }
  .ok { background:#12331f; color:#5fd08a; }
  .bad { background:#3a1520; color:#f08497; }
  .tiles { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }
  .tile { border:1px solid #222836; border-radius:8px; padding:10px 14px; min-width:120px; }
  .tile .k { font-size:11px; color:#9aa4b6; text-transform:uppercase; letter-spacing:.05em; }
  .tile .v { font-size:20px; font-weight:600; color:#e6e9ef; font-variant-numeric:tabular-nums; }
  .maps { display:flex; gap:28px; flex-wrap:wrap; margin-bottom:16px; }
  canvas { image-rendering:pixelated; background:#000; border:1px solid #222836;
           border-radius:4px; width:260px; height:260px; }
  table { border-collapse:collapse; width:100%; font-size:12px; }
  th, td { text-align:left; padding:5px 8px; border-bottom:1px solid #1c2431; }
  th { color:#9aa4b6; font-weight:600; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  details { margin-top:12px; }
  summary { cursor:pointer; color:#9ec1ff; }
  .scroll { max-height:320px; overflow:auto; margin-top:8px; }
</style></head>
<body>
<header>
  <h1>probar NN actual</h1>
  <a class="btn" href="/">&larr; volver al visor</a>
  <button id="reload">Recargar NN</button>
  <span id="minfo" class="dim"></span>
</header>
<main>
  <section class="panel">
    <h2>1 &middot; Datos de entrada disponibles</h2>
    <p class="dim">Marca los sets a probar (incluyen los usados en entrenamiento).
      Los incompatibles con la entrada de la NN actual quedan bloqueados y no se aplican.</p>
    <div id="datasets"></div>
    <div class="ctl"><label>&theta; disparo <span class="val" id="thv">0.40</span></label>
      <input type="range" id="th" min="0" max="1" step="0.01" value="0.40"></div>
    <button id="run" class="primary">Evaluar</button>
    <span id="runstatus" class="dim"></span>
  </section>
  <section class="panel" id="results" style="display:none">
    <h2>2 &middot; Análisis de resultados</h2>
    <div id="tiles" class="tiles"></div>
    <div class="maps">
      <div><h3>Neuronas ganadoras (conteo)</h3>
        <canvas id="winmap" width="50" height="50"></canvas>
        <div class="dim">brillo = nº de entradas que gana esa neurona</div></div>
      <div><h3>Fracción de disparo (&theta;)</h3>
        <canvas id="firemap" width="50" height="50"></canvas>
        <div class="dim">brillo = fracción de entradas en que dispara</div></div>
    </div>
    <div id="perdataset"></div>
    <details><summary>Ver respuesta por entrada</summary>
      <div class="scroll" id="perinput"></div></details>
  </section>
</main>
<script>
let MODEL=null, DATASETS=[];
const el = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

async function loadModel(){
  const m = await (await fetch('/api/model')).json();
  if(m.error){ el('minfo').textContent = m.error; MODEL=null;
    el('datasets').innerHTML=''; el('run').disabled=true; return; }
  MODEL=m; el('run').disabled=false;
  el('minfo').innerHTML = 'NN actual: <b>'+esc(m.path)+'</b> · '+m.n_in+'&rarr;'+m.n_out
    +' · mapa '+m.grid_h+'&times;'+m.grid_w+' · regla '+esc(m.learning_rule)
    +' · '+m.epochs_trained+' épocas · guardada '+esc(m.mtime);
  el('th').value=m.fire_threshold; el('thv').textContent=m.fire_threshold.toFixed(2);
  await loadDatasets();
}
async function loadDatasets(){
  DATASETS = (await (await fetch('/api/datasets')).json()).datasets;
  const box = el('datasets'); box.innerHTML='';
  DATASETS.forEach((d,i)=>{
    const row=document.createElement('div'); row.className='dsrow';
    const badge = d.compatible
      ? '<span class="badge ok">compatible</span>'
      : '<span class="badge bad">incompatible · dim '+d.dim+' &ne; '+MODEL.n_in+'</span>';
    row.innerHTML =
      '<input type="checkbox" id="ds'+i+'" '+(d.compatible?'':'disabled')+
        (d.compatible?' checked':'')+'>'+
      '<label for="ds'+i+'"><b>'+esc(d.path)+'</b></label>'+
      '<span class="dim">'+d.n+' imgs · '+d.shape.slice(1).join('&times;')+'</span>'+badge;
    box.appendChild(row);
  });
}
function selectedPaths(){
  return DATASETS.filter((d,i)=> d.compatible && el('ds'+i) && el('ds'+i).checked)
                 .map(d=>d.path);
}
function drawMap(canvas, arr, mw, mh, norm){
  const ctx=canvas.getContext('2d'), im=ctx.createImageData(mw,mh);
  let mx=norm; if(mx===undefined){ mx=1e-9; for(const v of arr) if(v>mx) mx=v; }
  for(let i=0;i<arr.length;i++){ const v=Math.round(Math.max(0,arr[i])/mx*255);
    im.data[i*4]=v; im.data[i*4+1]=v; im.data[i*4+2]=Math.min(255,v+20); im.data[i*4+3]=255; }
  ctx.putImageData(im,0,0);
}
function tile(k,v){ return '<div class="tile"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>'; }
function render(res){
  el('results').style.display='';
  const c=res.combined;
  el('tiles').innerHTML =
    tile('entradas', c.n_inputs) +
    tile('neuronas ganadoras', c.unique_winners) +
    tile('cobertura', (c.coverage*100).toFixed(2)+'%') +
    tile('act. ganador (μ)', c.mean_winner_act.toFixed(3)) +
    tile('disparan / entrada (μ)', c.mean_fired.toFixed(1)) +
    tile('muertas (del set)', c.dead_selected);
  drawMap(el('winmap'), c.win_count, res.map_w, res.map_h);
  drawMap(el('firemap'), c.fire_fraction, res.map_w, res.map_h, 1.0);
  // per-dataset summary
  let t='<table><tr><th>dataset</th><th class="num">imgs</th><th class="num">dim</th>'+
        '<th>estado</th><th class="num">ganadoras</th><th class="num">act μ</th>'+
        '<th class="num">disparan μ</th></tr>';
  res.datasets.forEach(d=>{
    const st = d.compatible ? '<span class="badge ok">ok</span>'
                            : '<span class="badge bad">excluido</span>';
    t+='<tr><td>'+esc(d.path)+'</td><td class="num">'+(d.n??'')+'</td>'+
       '<td class="num">'+(d.dim??'')+'</td><td>'+st+'</td>'+
       '<td class="num">'+(d.unique_winners??'—')+'</td>'+
       '<td class="num">'+(d.mean_winner_act!=null?d.mean_winner_act.toFixed(3):'—')+'</td>'+
       '<td class="num">'+(d.mean_fired!=null?d.mean_fired.toFixed(1):'—')+'</td></tr>';
  });
  t+='</table>'; el('perdataset').innerHTML=t;
  // per-input
  const p=c.per_input, names=res.datasets.map(d=>d.path);
  let r='<table><tr><th class="num">#</th><th>dataset</th><th>ganador (fila,col)</th>'+
        '<th class="num">act</th><th class="num">disparan</th></tr>';
  for(let i=0;i<p.winner.length;i++){
    const w=p.winner[i], row=Math.floor(w/res.map_w), col=w%res.map_w;
    r+='<tr><td class="num">'+i+'</td><td>'+esc(names[p.src[i]]||'')+'</td>'+
       '<td>#'+w+' ('+row+','+col+')</td>'+
       '<td class="num">'+p.winner_act[i].toFixed(3)+'</td>'+
       '<td class="num">'+p.n_fired[i]+'</td></tr>';
  }
  r+='</table>'; el('perinput').innerHTML=r;
}
async function run(){
  const paths=selectedPaths();
  if(!paths.length){ el('runstatus').textContent='marca al menos un set compatible'; return; }
  el('run').disabled=true; el('runstatus').textContent='evaluando…';
  try{
    const res=await (await fetch('/api/evaluate', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({paths, threshold:parseFloat(el('th').value)})})).json();
    if(!res.ok){ el('runstatus').textContent='error: '+(res.error||'?'); return; }
    el('runstatus').textContent='listo · '+res.combined.n_inputs+' entradas'+
      (res.incompatible.length?(' · '+res.incompatible.length+' excluidas por incompatibles'):'');
    render(res);
  }catch(e){ el('runstatus').textContent='error: '+e; }
  finally{ el('run').disabled=false; }
}
el('reload').onclick=loadModel;
el('run').onclick=run;
el('th').oninput=()=>el('thv').textContent=parseFloat(el('th').value).toFixed(2);
loadModel();
</script></body></html>"""


APPLY_PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aplicar set a la NN actual</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0d12; color:#e6e9ef; font:14px/1.5 system-ui, sans-serif; }
  header { padding:14px 18px; border-bottom:1px solid #222836;
           display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .ctl { display:flex; gap:8px; align-items:center; }
  label { color:#9aa4b6; font-size:12px; }
  input[type=range] { width:130px; accent-color:#6ea8fe; }
  button, a.btn, select { background:#161b26; color:#e6e9ef; border:1px solid #2a3242;
           border-radius:6px; padding:5px 10px; font:inherit; cursor:pointer; }
  button:hover, a.btn:hover, select:hover { border-color:#3a4256; }
  a.btn { text-decoration:none; display:inline-block; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  #status, #minfo { color:#6b7488; font-size:12px; font-variant-numeric:tabular-nums; }
  .val { font-variant-numeric:tabular-nums; color:#6ea8fe; }
  main { display:flex; gap:36px; justify-content:center; align-items:flex-start;
         padding:30px 18px; flex-wrap:wrap; }
  .col { text-align:center; }
  .col h2 { font-size:12px; color:#9aa4b6; font-weight:600; margin:0 0 10px;
            text-transform:uppercase; letter-spacing:.08em; }
  canvas { image-rendering:pixelated; background:#000; border:1px solid #222836; border-radius:4px; }
  .fired { color:#9aa4b6; font-size:12px; margin-top:8px; }
</style></head>
<body>
<header>
  <h1>aplicar set &middot; entrada <span class="val" id="idx">0</span>/<span id="n">?</span></h1>
  <a class="btn" href="/">&larr; visor entrenamiento</a>
  <a class="btn" href="/test">análisis /test</a>
  <div class="ctl"><label>dataset</label><select id="dataset"></select></div>
  <div class="ctl"><button id="play">Pause</button>
    <button id="prev">&larr;</button><button id="next">&rarr;</button>
    <button id="reset">Reset trail</button><button id="reload">Recargar NN</button></div>
  <div class="ctl"><label>ms/entrada <span class="val" id="msv">120</span></label>
    <input type="range" id="ms" min="30" max="1500" step="10" value="120"></div>
  <div class="ctl"><label>trail speed <span class="val" id="rtv">0.30</span></label>
    <input type="range" id="rt" min="0.02" max="1" step="0.02" value="0.30"></div>
  <div class="ctl"><label>&theta; <span class="val" id="thv">0.40</span></label>
    <input type="range" id="th" min="0" max="1" step="0.01" value="0.40"></div>
  <span id="minfo"></span><span id="status"></span>
</header>
<main>
  <div class="col"><h2>Input (this one)</h2>
    <canvas id="img" width="28" height="28" style="width:168px;height:168px"></canvas></div>
  <div class="col"><h2>Firing (this input)</h2>
    <canvas id="now" width="50" height="50" style="width:300px;height:300px"></canvas>
    <div class="fired">fired: <span id="fired">0</span> · winner #<span id="win">-</span></div></div>
  <div class="col"><h2>Uso acumulado</h2>
    <canvas id="trail" width="50" height="50" style="width:300px;height:300px"></canvas>
    <div class="fired">brillo = neuronas usadas a lo largo del set</div></div>
</main>
<script>
let MODEL=null, DATASETS=[], META=null, SEQ=null, IMGSEQ=null;
let step=0, trail=null, timer=null, playing=true, thTouched=false;
const el = id => document.getElementById(id);

function drawImg(canvas, arr, side){
  const ctx=canvas.getContext('2d'), im=ctx.createImageData(side,side);
  for(let i=0;i<arr.length;i++){const v=arr[i];im.data[i*4]=v;im.data[i*4+1]=v;im.data[i*4+2]=v;im.data[i*4+3]=255;}
  ctx.putImageData(im,0,0);
}
function drawNow(act, th){
  const ctx=el('now').getContext('2d'), n=act.length, im=ctx.createImageData(META.map_w,META.map_h);
  let fired=0, mx=-1e9, arg=0;
  for(let i=0;i<n;i++){ if(act[i]>mx){mx=act[i];arg=i;} }
  for(let i=0;i<n;i++){ const on=act[i]>=th; if(on)fired++;
    const v=on?255:0; im.data[i*4]=v; im.data[i*4+1]=v; im.data[i*4+2]=on?255:0; im.data[i*4+3]=255; }
  ctx.putImageData(im,0,0); el('win').textContent=arg; return fired;
}
function accumTrail(act, th, rate){
  // "uso acumulado": monotonic — a neuron brightens when it fires and never
  // fades (no decay term), so the panel builds up the set's usage map. Called
  // once per input advance, NOT on every re-draw (would over-count).
  for(let i=0;i<act.length;i++){ if(act[i]>=th) trail[i] += rate*(1-trail[i]); }
}
function drawTrail(){
  const ctx=el('trail').getContext('2d'), im=ctx.createImageData(META.map_w,META.map_h);
  for(let i=0;i<trail.length;i++){ const v=Math.round(trail[i]*255);
    im.data[i*4]=v; im.data[i*4+1]=v; im.data[i*4+2]=Math.min(255,v+20); im.data[i*4+3]=255; }
  ctx.putImageData(im,0,0);
}
function resetTrail(){ trail=new Float32Array(META.map_w*META.map_h); drawTrail(); }
function render(){  // advance to a new input: accumulate the trail once, then draw
  const th=parseFloat(el('th').value), rate=parseFloat(el('rt').value);
  el('idx').textContent=step;
  drawImg(el('img'), IMGSEQ[step], META.side);
  el('fired').textContent=drawNow(SEQ[step], th);
  accumTrail(SEQ[step], th, rate); drawTrail();
}
function redraw(){  // re-draw current step WITHOUT re-accumulating (e.g. on θ drag)
  el('fired').textContent=drawNow(SEQ[step], parseFloat(el('th').value)); drawTrail();
}
function advance(){ step=(step+1)%SEQ.length; if(step===0) resetTrail(); render(); }
function setPlaying(p){ playing=p; el('play').textContent=p?'Pause':'Play'; }
function step_(d){ setPlaying(false); step=(step+d+SEQ.length)%SEQ.length; if(step===0) resetTrail(); render(); }
function tick(){ if(playing && SEQ) advance(); timer=setTimeout(tick, parseInt(el('ms').value)); }

el('play').onclick=()=>setPlaying(!playing);
el('prev').onclick=()=>step_(-1);
el('next').onclick=()=>step_(1);
el('reset').onclick=()=>{resetTrail(); render();};
el('reload').onclick=()=>init();
el('ms').oninput=()=>el('msv').textContent=el('ms').value;
el('rt').oninput=()=>el('rtv').textContent=parseFloat(el('rt').value).toFixed(2);
el('th').oninput=()=>{thTouched=true; el('thv').textContent=parseFloat(el('th').value).toFixed(2); if(SEQ) redraw();};
el('dataset').onchange=()=>loadApply(el('dataset').value);

async function loadApply(path){
  if(!path){ el('status').textContent='no hay datasets compatibles'; return; }
  el('status').textContent='aplicando…';
  try{
    const r=await (await fetch('/api/apply?path='+encodeURIComponent(path))).json();
    if(!r.ok){ el('status').textContent='error: '+r.error; return; }
    META=r; SEQ=r.seq; IMGSEQ=r.imgseq; step=0;
    el('n').textContent=SEQ.length-1;
    if(!thTouched){ el('th').value=META.fire_threshold; el('thv').textContent=META.fire_threshold.toFixed(2); }
    resetTrail(); render();
    el('status').textContent='listo · '+r.n+' entradas · '+r.path;
  }catch(e){ el('status').textContent='error: '+e; }
}
async function init(){
  const m=await (await fetch('/api/model')).json();
  if(m.error){ el('minfo').textContent=m.error; return; }
  MODEL=m;
  el('minfo').textContent='NN: '+m.n_in+'→'+m.n_out+' · '+m.learning_rule+' · '+m.epochs_trained+' épocas';
  DATASETS=(await (await fetch('/api/datasets')).json()).datasets;
  const sel=el('dataset'); sel.innerHTML='';
  DATASETS.forEach(d=>{
    const o=document.createElement('option'); o.value=d.path;
    o.textContent=d.path+' ('+d.n+')'+(d.compatible?'':' — incompatible');
    o.disabled=!d.compatible; sel.appendChild(o);
  });
  const first=DATASETS.find(d=>d.compatible);
  if(first){ sel.value=first.path; await loadApply(first.path); }
  else el('status').textContent='no hay datasets compatibles con esta NN';
}
(async()=>{ await init(); tick(); })();
</script></body></html>"""


def make_handler(default_file, model_path=DEFAULT_MODEL, runs_dir=DEFAULT_RUNS_DIR):
    # Payloads are cached per file, keyed by mtime; rebuilt only when a file
    # changes on disk. A request may select any archived run via ?file=, so we
    # whitelist paths (the default file or anything under runs_dir) to avoid
    # serving arbitrary files.
    cache = {}  # resolved path -> {"mtime": ..., "payloads": (...)}
    runs_real = os.path.realpath(runs_dir)
    default_real = os.path.realpath(default_file)

    def resolve(file_param):
        """Map a ?file= value to an allowed absolute path, or None if rejected."""
        if not file_param:
            return default_file if os.path.exists(default_file) else None
        real = os.path.realpath(file_param)
        if real == default_real:
            return file_param
        if real.startswith(runs_real + os.sep) and real.endswith(".npz"):
            return file_param
        return None

    def payloads(file_param):
        path = resolve(file_param)
        if path is None or not os.path.exists(path):
            miss = file_param or default_file
            err = json.dumps({"error": f"no existe/invalido {miss} (corre gen_evolution.py)"}).encode()
            return err, err, err, err
        mtime = os.path.getmtime(path)
        entry = cache.get(path)
        if entry is None or entry["mtime"] != mtime:
            meta, seq, image, imgseq = _load_file(path)
            meta["file"] = path.replace("\\", "/")
            imgseq_p = (
                json.dumps({"imgseq": imgseq.tolist()}).encode()
                if imgseq is not None else json.dumps({"imgseq": None}).encode()
            )
            entry = {"mtime": mtime, "payloads": (
                json.dumps(meta).encode(),
                json.dumps({"seq": np.round(seq, 4).tolist()}).encode(),
                json.dumps({"image": image.tolist()}).encode(),
                imgseq_p,
            )}
            cache[path] = entry
        return entry["payloads"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
                return
            if self.path == "/test" or self.path.startswith("/test?"):
                self._send(TEST_PAGE.encode(), "text/html; charset=utf-8")
                return
            if self.path == "/apply" or self.path.startswith("/apply?"):
                self._send(APPLY_PAGE.encode(), "text/html; charset=utf-8")
                return
            if self.path.startswith("/api/apply"):
                layer, info = load_current_model(model_path)
                if layer is None:
                    self._json({"ok": False, "error": info.get("error", "sin modelo")})
                    return
                qs = parse_qs(urlparse(self.path).query)
                ds = qs.get("path", [None])[0]
                if not ds:
                    self._json({"ok": False, "error": "falta ?path=<dataset>"})
                    return
                self._json(apply_dataset(layer, ds))
                return
            if self.path == "/api/model":
                _, info = load_current_model(model_path)
                self._json(info)
                return
            if self.path == "/api/datasets":
                layer, _ = load_current_model(model_path)
                n_in = layer.n_in if layer is not None else None
                self._json({"datasets": scan_datasets(n_in=n_in)})
                return
            if self.path == "/api/runs":
                self._json({"runs": scan_runs(runs_dir, default_file)})
                return
            file_param = parse_qs(urlparse(self.path).query).get("file", [None])[0]
            meta_p, seq_p, image_p, imgseq_p = payloads(file_param)
            base = urlparse(self.path).path
            if base == "/api/meta":
                self._send(meta_p, "application/json")
            elif base == "/api/seq":
                self._send(seq_p, "application/json")
            elif base == "/api/image":
                self._send(image_p, "application/json")
            elif base == "/api/imgseq":
                self._send(imgseq_p, "application/json")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/evaluate":
                self.send_error(404)
                return
            layer, info = load_current_model(model_path)
            if layer is None:
                self._json({"ok": False, "error": info.get("error", "sin modelo")})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as e:
                self._json({"ok": False, "error": f"body invalido: {e}"})
                return
            paths = body.get("paths", [])
            threshold = float(body.get("threshold", layer.fire_threshold))
            self._json(evaluate(layer, paths, threshold))

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistence-trail viewer (pure server)")
    ap.add_argument("--file", default=DEFAULT_FILE,
                    help="fallback sequence .npz (legacy 'latest' file)")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                    help="dir con los runs archivados a listar (mas reciente arriba)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="modelo de la 'NN actual' para la pagina /test "
                         "(por defecto el ultimo experimento en lastexperiment/)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not os.path.exists(args.file) and not glob.glob(os.path.join(args.runs_dir, "*.npz")):
        print(f"warning: no hay runs en {args.runs_dir} ni {args.file}; "
              f"corre gen_evolution.py y pulsa Refrescar en la pagina")

    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(args.file, args.model, args.runs_dir))
    print(f"serving runs from {args.runs_dir} at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    print(f"pagina de pruebas: http://127.0.0.1:{args.port}/test  (NN actual: {args.model})")
    print("re-run gen_evolution.py anytime, then click Refrescar (no restart needed)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
