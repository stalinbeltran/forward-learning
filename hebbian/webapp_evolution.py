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
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


DEFAULT_FILE = "experiments/evolution/sequence.npz"


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
  button { background:#161b26; color:#e6e9ef; border:1px solid #2a3242;
           border-radius:6px; padding:5px 10px; font:inherit; }
  button:hover { border-color:#3a4256; }
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
    <button id="reset">Reset trail</button><button id="refresh">Refrescar</button></div>
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
el('refresh').onclick=()=>load(true);
el('ms').oninput=()=>el('msv').textContent=el('ms').value;
el('rt').oninput=()=>el('rtv').textContent=parseFloat(el('rt').value).toFixed(2);
el('th').oninput=()=>{thTouched=true; el('thv').textContent=parseFloat(el('th').value).toFixed(2);};

async function load(manual){
  try{
    const meta=await (await fetch('/api/meta')).json();
    if(meta.error){ el('status').textContent='sin datos: '+meta.error; return; }
    const seq=(await (await fetch('/api/seq')).json()).seq;
    const img=(await (await fetch('/api/image')).json()).image;
    const imgseq=meta.has_imgseq ? (await (await fetch('/api/imgseq')).json()).imgseq : null;
    META=meta; SEQ=seq; IMG=img; IMGSEQ=imgseq; step=0;
    computeBlockEnds(); heldAtBoundary=false; el('apnote').textContent='';
    el('steps').textContent=SEQ.length-1;
    if(!thTouched){ el('th').value=META.fire_threshold; el('thv').textContent=META.fire_threshold.toFixed(2); }
    drawImg(el('img'), IMGSEQ?IMGSEQ[0]:img, META.side); resetTrail(); render();
    const t=new Date().toLocaleTimeString();
    el('status').textContent=(manual?'refrescado ':'cargado ')+t+' · '+META.mtime;
  }catch(e){ el('status').textContent='error al cargar: '+e; }
}
(async()=>{ await load(false); tick(); })();
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


def make_handler(path):
    # Cache payloads keyed by the file's mtime; rebuild only when the file changes.
    cache = {"mtime": None, "payloads": None}

    def payloads():
        if not os.path.exists(path):
            err = json.dumps({"error": f"no existe {path} (corre gen_evolution.py)"}).encode()
            return err, err, err, err
        mtime = os.path.getmtime(path)
        if cache["mtime"] != mtime:
            meta, seq, image, imgseq = _load_file(path)
            cache["mtime"] = mtime
            imgseq_p = (
                json.dumps({"imgseq": imgseq.tolist()}).encode()
                if imgseq is not None else json.dumps({"imgseq": None}).encode()
            )
            cache["payloads"] = (
                json.dumps(meta).encode(),
                json.dumps({"seq": np.round(seq, 4).tolist()}).encode(),
                json.dumps({"image": image.tolist()}).encode(),
                imgseq_p,
            )
        return cache["payloads"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
                return
            meta_p, seq_p, image_p, imgseq_p = payloads()
            if self.path == "/api/meta":
                self._send(meta_p, "application/json")
            elif self.path == "/api/seq":
                self._send(seq_p, "application/json")
            elif self.path == "/api/image":
                self._send(image_p, "application/json")
            elif self.path == "/api/imgseq":
                self._send(imgseq_p, "application/json")
            else:
                self.send_error(404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistence-trail viewer (pure server)")
    ap.add_argument("--file", default=DEFAULT_FILE, help="sequence .npz to serve")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"warning: {args.file} no existe todavia; "
              f"corre gen_evolution.py y pulsa Refrescar en la pagina")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.file))
    print(f"serving {args.file} at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    print("re-run gen_evolution.py anytime, then click Refrescar (no restart needed)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
