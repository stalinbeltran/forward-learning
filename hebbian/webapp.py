"""Local web viewer for a trained model (condensado.md §9).

Walks through the dataset inputs; for each one it shows the ORIGINAL and its
NEGATIVE side by side, each with the 50x50 map of active neurons. The neuron
map has three modes (computed live in the browser from the raw activation):

  * digital fire   -> pixel on when a >= theta
  * only winner    -> only the argmax neuron
  * full activation-> grayscale of the (clipped, scaled) activation

Editable in the page: milliseconds per input (10-1000), theta, mode, play/pause.
Pure stdlib http.server + numpy. No external web dependencies.

    python hebbian/webapp.py --model experiments/run/model.npz \
        --dataset data/processed/lines_hebbian/lines.npz
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:
    from .competitive_net import CompetitiveLayer
    from . import metrics as M
except ImportError:  # pragma: no cover - script fallback
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from competitive_net import CompetitiveLayer
    import metrics as M


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forward-learning viewer</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0d12; color:#e6e9ef;
         font:14px/1.5 system-ui, sans-serif; }
  header { padding:14px 18px; border-bottom:1px solid #222836;
           display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.02em; }
  .ctl { display:flex; gap:8px; align-items:center; }
  label { color:#9aa4b6; font-size:12px; }
  input[type=range] { width:130px; accent-color:#6ea8fe; }
  select, button { background:#161b26; color:#e6e9ef; border:1px solid #2a3242;
                   border-radius:6px; padding:5px 10px; font:inherit; }
  button:hover, select:hover { border-color:#3a4256; }
  main { display:flex; gap:32px; justify-content:center; padding:28px 18px; flex-wrap:wrap; }
  .col { text-align:center; }
  .col h2 { font-size:12px; color:#9aa4b6; font-weight:600; margin:0 0 8px;
            text-transform:uppercase; letter-spacing:.08em; }
  canvas { image-rendering:pixelated; background:#000; border:1px solid #222836;
           border-radius:4px; }
  .pair { display:flex; gap:14px; margin-top:12px; }
  .val { font-variant-numeric:tabular-nums; color:#6ea8fe; }
  .fired { color:#9aa4b6; font-size:12px; margin-top:6px; }
</style></head>
<body>
<header>
  <h1>forward-learning &middot; input <span class="val" id="idx">0</span>/<span id="n">?</span></h1>
  <div class="ctl"><button id="play">Pause</button>
    <button id="prev">&larr;</button><button id="next">&rarr;</button></div>
  <div class="ctl"><label>ms/input <span class="val" id="msv">120</span></label>
    <input type="range" id="ms" min="10" max="1000" step="10" value="120"></div>
  <div class="ctl"><label>mode</label>
    <select id="mode">
      <option value="fire">digital fire</option>
      <option value="winner">only winner</option>
      <option value="full">full activation</option>
    </select></div>
  <div class="ctl"><label>&theta; <span class="val" id="thv">0.40</span></label>
    <input type="range" id="th" min="0" max="1" step="0.01" value="0.40"></div>
</header>
<main>
  <div class="col"><h2>Original</h2>
    <div class="pair"><canvas id="oimg" width="28" height="28" style="width:112px;height:112px"></canvas>
      <canvas id="omap" width="50" height="50" style="width:200px;height:200px"></canvas></div>
    <div class="fired">fired: <span id="ofired">0</span></div></div>
  <div class="col"><h2>Negative</h2>
    <div class="pair"><canvas id="nimg" width="28" height="28" style="width:112px;height:112px"></canvas>
      <canvas id="nmap" width="50" height="50" style="width:200px;height:200px"></canvas></div>
    <div class="fired">fired: <span id="nfired">0</span></div></div>
</main>
<script>
let META=null, cur=0, timer=null, playing=true, cache={};
const el = id => document.getElementById(id);

function drawImg(canvas, arr, side){
  const ctx=canvas.getContext('2d'), im=ctx.createImageData(side,side);
  for(let i=0;i<arr.length;i++){const v=arr[i]; im.data[i*4]=v;im.data[i*4+1]=v;im.data[i*4+2]=v;im.data[i*4+3]=255;}
  ctx.putImageData(im,0,0);
}
function drawMap(canvas, act, mode, th){
  const ctx=canvas.getContext('2d'), n=act.length, im=ctx.createImageData(META.map_w,META.map_h);
  let fired=0, mx=1e-6, arg=0;
  for(let i=0;i<n;i++){ if(act[i]>mx){mx=act[i];arg=i;} }
  for(let i=0;i<n;i++){
    let v=0;
    if(mode==='fire'){ v = act[i]>=th?255:0; if(act[i]>=th)fired++; }
    else if(mode==='winner'){ v = (i===arg)?255:0; }
    else { v = Math.max(0,act[i])/mx*255; if(act[i]>=th)fired++; }
    im.data[i*4]=v; im.data[i*4+1]=(mode==='winner'&&i===arg)?180:v;
    im.data[i*4+2]=v>0?Math.min(255,v+30):0; im.data[i*4+3]=255;
  }
  ctx.putImageData(im,0,0);
  return fired;
}
async function frame(i){
  if(!cache[i]){ const r=await fetch('/api/frame/'+i); cache[i]=await r.json(); }
  return cache[i];
}
async function render(){
  const f=await frame(cur), mode=el('mode').value, th=parseFloat(el('th').value);
  el('idx').textContent=cur;
  drawImg(el('oimg'), f.orig, META.side); drawImg(el('nimg'), f.neg, META.side);
  el('ofired').textContent=drawMap(el('omap'), f.act, mode, th);
  el('nfired').textContent=drawMap(el('nmap'), f.act_neg, mode, th);
}
function step(d){ cur=(cur+d+META.n)%META.n; render(); }
function loop(){ if(playing) step(1); timer=setTimeout(loop, parseInt(el('ms').value)); }
el('play').onclick=()=>{playing=!playing; el('play').textContent=playing?'Pause':'Play';};
el('prev').onclick=()=>{playing=false;el('play').textContent='Play';step(-1);};
el('next').onclick=()=>{playing=false;el('play').textContent='Play';step(1);};
el('ms').oninput=()=>el('msv').textContent=el('ms').value;
el('th').oninput=()=>{el('thv').textContent=parseFloat(el('th').value).toFixed(2);render();};
el('mode').onchange=render;
(async()=>{ META=await (await fetch('/api/meta')).json();
  el('n').textContent=META.n; el('th').value=META.fire_threshold;
  el('thv').textContent=META.fire_threshold.toFixed(2);
  render(); loop(); })();
</script></body></html>"""


class Viewer:
    """Holds precomputed per-input images and activations."""

    def __init__(self, layer: CompetitiveLayer, X: np.ndarray, Xneg: np.ndarray):
        self.layer = layer
        self.imgs = (X * 255).astype(np.uint8)
        self.imgs_neg = (Xneg * 255).astype(np.uint8)
        self.A = M.activations(layer.W, X).astype(np.float32)
        self.A_neg = M.activations(layer.W, Xneg).astype(np.float32)
        self.side = int(round(X.shape[1] ** 0.5))

    def meta(self) -> dict:
        return {
            "n": len(self.imgs), "side": self.side,
            "map_h": self.layer.grid_h, "map_w": self.layer.grid_w,
            "fire_threshold": self.layer.fire_threshold,
        }

    def frame(self, i: int) -> dict:
        return {
            "orig": self.imgs[i].tolist(),
            "neg": self.imgs_neg[i].tolist(),
            "act": np.round(self.A[i], 4).tolist(),
            "act_neg": np.round(self.A_neg[i], 4).tolist(),
        }


def make_handler(viewer: Viewer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/meta":
                self._json(viewer.meta())
            elif self.path.startswith("/api/frame/"):
                i = int(self.path.rsplit("/", 1)[1])
                self._json(viewer.frame(i % viewer.meta()["n"]))
            else:
                self.send_error(404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Local web viewer for a trained model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--key", default="images")
    ap.add_argument("--neg-key", default=None,
                    help="npz key for negatives; default computes 255-image")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    layer = CompetitiveLayer.load(args.model)
    d = np.load(args.dataset)
    X = d[args.key].astype(np.float32) / 255.0
    X = X.reshape(len(X), -1)
    if args.neg_key and args.neg_key in d.files:
        Xneg = d[args.neg_key].astype(np.float32) / 255.0
        Xneg = Xneg.reshape(len(Xneg), -1)
    else:
        Xneg = 1.0 - X

    viewer = Viewer(layer, X, Xneg)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(viewer))
    print(f"serving {len(X)} inputs at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    print(f"model: {layer}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
