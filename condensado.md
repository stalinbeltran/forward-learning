# Condensado — Red neuronal de *forward learning* (competitiva Hebbiana + inhibición lateral)

Documento autocontenido para **re-crear esta funcionalidad desde cero en otro proyecto**. Reúne el
modelo, el algoritmo exacto, los valores de entrenamiento calibrados, las herramientas de análisis y
los hallazgos. Todo vive hoy en la carpeta `hebbian/` de este repo y es **numpy puro** (sin PyTorch).

---

## 1. Objetivo y concepto

Red de **una sola capa** que aprende **sin supervisión ni backprop**, por refuerzo competitivo tipo
Hebbiano ("*forward learning*"):

1. Se crea la capa con **pesos aleatorios**.
2. Se presenta una entrada. Con pesos aleatorios, unas neuronas quedan **cerca de dispararse**
   (activación alta) y otras lejos.
3. **Regla base (solo refuerza):** a las neuronas cerca de dispararse se les acercan los pesos a la
   entrada presente → la próxima vez es más probable que disparen ante entradas parecidas.
4. **Reducción de pesos = neuronas inhibidoras** (única vía): una malla de inhibidores superpuestos
   regula el exceso de disparos locales debilitando conexiones. Sin ellas, la competición colapsa a
   pocas neuronas acaparadoras.

Entrada de ejemplo: imágenes 28×28 de **rectas** (líneas) en distintas inclinaciones (estilo NIST).
Salida: mapa de **50×50 = 2500 neuronas**. No hay clasificación; es aprendizaje de representación.

---

## 2. Núcleo matemático (preciso)

- Pesos `W` de forma `(n_out, n_in)` = `(2500, 784)`. **Cada fila se normaliza a norma 1**.
- Cada entrada `x` se normaliza a norma 1 → `xu`. Así **activación = similitud coseno**:
  `a = W @ xu ∈ [-1, 1]` (comparable entre neuronas). "Disparar" ≈ `a` alto.

**Refuerzo (gate ≥ 0, solo incrementa):**
`gate_i = tanh( relu(a_i − media(a)) / (std(a) + 1e-8) )`  (regla `above_mean`, la usada).
Las que están por encima de la media refuerzan; las de abajo quedan a 0 (no se tocan).

**Inhibición lateral (única reducción):** malla de inhibidores en el mapa 50×50. Cada inhibidor `I`
tiene una **región** = neuronas dentro de `radius` (métrica *cheby* = cuadrado). Mira qué fracción de
su región dispara (`a ≥ fire_threshold`); si supera `K`, aplica un castigo proporcional al exceso a
las neuronas **disparadas** de su región:
`exceso_I = max(fired_I / |region_I| − K, 0)` ; `s_i += inhib_gain · exceso_I` (por cada `I` que cubre a `i`).
La ganancia inhibidora es **independiente del lr**.

**Actualización combinada por muestra** (refuerzo e inhibición van sobre el mismo eje `xu`):
```
coef_i = lr · reinforce_gain · gate_i  −  s_i
W[i]  += coef_i · xu          (solo filas con coef_i ≠ 0)
W[i]  ← W[i] / ‖W[i]‖         (renormalizar a norma 1)
```
La renormalización mantiene la escala coseno: la inhibición no "encoge" el vector, lo **rota
alejándolo** de la entrada (reduce el disparo futuro).

---

## 3. Código núcleo (lo esencial para reimplementar)

Clase `CompetitiveLayer` (numpy). Métodos clave:

```python
def _normalize_rows(M, eps=1e-8):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), eps)

def _gate(self, a):                      # refuerzo >= 0 (rule 'above_mean')
    return np.tanh(np.maximum(a - a.mean(), 0.0) / (a.std() + 1e-8))

def _inhibition_coeffs(self, a):         # s_i >= 0 a restar (solo a disparadas)
    fired = a >= self.fire_threshold
    s = np.zeros(self.n_out, np.float32)
    for idx in self._inhib_regions:      # regiones precalculadas (una por inhibidor)
        fr = fired[idx]; nf = int(fr.sum())
        if nf == 0: continue
        e = nf/idx.size - self.inhib_K   # 'fraction'; si >0 castiga
        if e > 0: s[idx[fr]] += self.inhib_gain * e
    return s

def learn_sample(self, x, lr):
    xu = self._normalize_vec(x)
    a = self.W @ xu
    self.win_count[int(a.argmax())] += 1
    coef = lr * self.reinforce_gain * self._gate(a)
    if self.inhib_on:
        coef = coef - self._inhibition_coeffs(a)
    idx = np.nonzero(coef)[0]
    if idx.size:
        self.W[idx] += coef[idx][:, None] * xu[None, :]
        self.W[idx] = self._normalize_rows(self.W[idx])
```

**Construcción de las regiones de inhibición** (malla cada `spacing`, radio `radius`, cuadrado):
```python
centers = [(r,c) for r in range(offset, grid_h, spacing)
                 for c in range(offset, grid_w, spacing)]   # offset = spacing//2
nr, nc = np.arange(n_out)//grid_w, np.arange(n_out)%grid_w  # fila/col de cada neurona
for rc, cc in centers:
    mask = (np.abs(nr-rc) <= radius) & (np.abs(nc-cc) <= radius)  # 'cheby'
    regions.append(np.nonzero(mask)[0])
```
Init: `W = rng.standard_normal((n_out, n_in)); W = _normalize_rows(W)`.

**Persistencia** (`save`/`load`): guardar `W`, `win_count`, `epochs_trained`, y TODOS los
hiperparámetros (rule, reinforce_gain, grid_h/w, inhib_on/spacing/offset/radius/metric,
fire_threshold, inhib_K, inhib_gain, inhib_mode) en un `.npz`. Reconstruir la capa desde ellos y
recomputar las regiones de inhibición. Con `load` compatible hacia atrás (defaults si falta clave).
Esto permite **reanudar** el entrenamiento, incluso con otro set de entradas.

---

## 4. Hiperparámetros y valores recomendados (calibrados)

| Parámetro | Valor recomendado | Notas / cómo se calibró |
|---|---|---|
| `n_in`, `n_out` | 784, 2500 (mapa 50×50) | entrada 28×28, salida 50×50 |
| `rule` | `above_mean` | gate = tanh(relu(a−media)/std). Alternativas: `softmax`, `wta` |
| `reinforce_gain` (ganancia de activación) | **1.0** | *sweep*: U en neuronas muertas; óptimo 1–2. <0.5 no aprende, >4 colapsa |
| `inhib_gain` (ganancia inhibidora, indep. del lr) | **1.5** | *sweep*: `gain>0` recorta disparos; codo agudo ~0.5–2; saturación >8 |
| `fire_threshold` θ | **0.40** | umbral de "disparo" (coseno). Sube a ~0.44–0.46 para código muy esparso |
| `inhib_K` (fracción) | **0.10** | bajo K no inhibe; usa fracción (comparable centro/borde) |
| `inhib_spacing` | 5 | inhibidores cada 5 → 10×10 = 100 inhibidores |
| `inhib_radius` | 8 | región cuadrada (`cheby`); interior 289 neuronas, borde menos |
| `inhib_metric` | `cheby` (cuadrado) | también `euclid`, `manhattan` |
| `inhib_mode` | `fraction` | también `hinge` (conteo absoluto), `sigmoid` (softplus) |
| `lr` | 0.02–0.1 | **constante alto (0.1) = churn líquido**; 0.02 calma pesos. Ver §8 |
| `seed` | 0 | reproducible (init de pesos y shuffle) |

Regla de oro observada: **el algoritmo base solo incrementa pesos**; toda reducción la hacen los
inhibidores. La `anti` (anti-Hebbiano por-neurona) es legado y ya **no se usa**.

---

## 5. Datos

- **Rectas 28×28** (`generate_lines.py`): N imágenes con una línea que cruza el lienzo a un ángulo
  aleatorio 0–180°, con jitter de posición ~4 px. Dibujadas a 4× y reducidas (antialiasing, PIL).
  Se guardan como `.npz` con `images` uint8 `(N,28,28)`. Por defecto **1000** imágenes.
- **Negativo fotográfico**: `neg = 255 − imagen`. Un set **pos+neg** = concatenar originales +
  negativas (fondo blanco, línea negra). Se usó para entrenar con ambos y comparar respuesta.
- Sub-sets pequeños (10, 20, 1 imagen) para estudiar dinámica. Todo en `data/processed/lines_hebbian/`.

Nota: bajo similitud coseno, las **negativas disparan ~4–10× más** neuronas que las positivas (son
imágenes "densas": casi todo el fondo brillante correlaciona con muchas neuronas). Entrenar con ambas
lo atenúa pero no lo iguala (es estructural del coseno + estadística de la entrada).

---

## 6. Estructura de archivos (módulo `hebbian/`)

```
competitive_net.py     # NÚCLEO: CompetitiveLayer (init/gate/inhibición/learn/save/load)
generate_lines.py      # dataset de rectas 28x28 -> .npz (+ preview)
train.py               # entrenamiento config-driven con animaciones (GIF campos receptivos/ganadoras)
train_series.py        # entrena y guarda snapshot cada N épocas; soporta --resume
train_rounds.py        # aprendizaje CONTINUAL por rondas (10 pos + sus 10 neg por ronda)
train_watch.py         # entrena registrando parámetros de estabilidad EN CADA época (CSV + plot)
train_one.py           # entrena con UNA imagen; graba frames solo cuando cambia el disparo
show_winners.py        # mapa 50x50 de solo las GANADORAS (resto en negro) + panel
compare_models.py      # montaje comparando el disparo de varios snapshots por entrada
analyze_series.py      # estabilidad (Jaccard/winner_match) + disparos por snapshot
analyze_rounds.py      # retención/OLVIDO: matriz snapshot × set-de-ronda + heatmap
diagnose_stability.py  # diagnóstico fino: dW_rel, act_cos, top10_jac, win_match
sweep_inhib_gain.py    # barrido de la ganancia inhibidora (mean_fired vs gain)
sweep_reinforce_gain.py# barrido de la ganancia de activación (muertas/cobertura/nitidez vs gain)
webapp.py              # visor web local: por entrada, ORIGINAL vs NEGATIVO, neuronas activas
webapp_evolution.py    # visor web: 1 imagen fija, RASTRO DE PERSISTENCIA del disparo por época
```

Salidas en `experiments/<run>/` (gitignored): `model.npz` (reanudable), snapshots, GIFs, PNG, CSV.

---

## 7. Métricas (y qué significan)

- **`dead_units`**: neuronas que nunca han ganado (argmax) ninguna entrada. Menos = mejor uso del mapa.
- **`coverage`**: fracción de neuronas que ganan algo en la época.
- **`unique_winners`**: nº de ganadoras distintas.
- **`mean_winner_activation`** (nitidez): activación media del ganador (coseno). Alto = patrones nítidos.
- **`mean_fired`**: media de neuronas que disparan por entrada (a ≥ θ). "No queremos que sean pocas."
- **`dW_rel`**: cambio relativo de pesos entre dos fotos, `‖ΔW‖/‖W‖`. 0 = pesos quietos.
- **`act_cos`**: coseno medio, por entrada, entre el vector de activación de las 2500 neuronas en dos
  fotos. Alto = la **función** (respuesta) se conserva aunque cambien las neuronas.
- **`top10_jac`**: solape Jaccard de las 10 neuronas más activas por entrada (robusto a empates).
- **`win_match`**: fracción de entradas cuya **ganadora** (argmax) es la misma entre dos fotos (lo más estricto).
- **Retención** (continual): activación del ganador de un snapshot sobre el set de una ronda antigua
  (mide olvido).

---

## 8. Hallazgos clave (lo aprendido)

1. **La competición pura colapsa**: sin inhibición, ~7 neuronas acaparan todo (miles muertas). Con
   `above_mean` puro también. La **inhibición lateral** (o una "conciencia") es necesaria para repartir.
2. **Base solo-refuerza + inhibidores = regulador homeostático**: entrenar con la inhibición **desde
   el init** (no añadida tarde) reduce muchísimo las muertas (p. ej. 383 vs ~1400) y acota el disparo.
3. **Presión de datos ⇒ inestabilidad ("líquido")**: con muchas imágenes (~2000) ↔ muchas neuronas,
   la representación se **reorganiza sin parar**: `dW_rel` alto, `top10_jac`/`win_match` ≈ 0, y **no
   converge** ni con annealing del lr. Con **pocas imágenes** (10–20) la **función cristaliza**
   (`act_cos` ~0.9–0.98) y converge, aunque el sustrato sigue con algo de churn (redundancia).
4. **Función estable ≠ sustrato estable**: casi siempre `act_cos` alto (la salida no cambia) pero las
   neuronas concretas que la producen rotan mucho. Analogía: "misma atención, cajero distinto".
5. **Bajar el lr calma el sustrato pero no fija el argmax**: a `lr=0.02`, `dW_rel` baja a ~0.12–0.18 y
   `act_cos`→0.98, pero `win_match` sigue oscilando (empates cercanos que se rompen por micro-ruido).
6. **Cuántas neuronas cambian por época**: ~**44% (~1090 de 2500)** cambian **apreciablemente** cada
   época (bimodal: se mueven mucho o nada). No es solo la ganadora.
7. **Aprendizaje continual SIN OLVIDO**: entrenando por rondas con sets nuevos (10 pos+10 neg, 5
   épocas c/u), retiene **88–105%** de la respuesta a sets antiguos. Motivo: **asignación esparsa**
   desde una reserva enorme de neuronas muertas (cada set toma neuronas frescas sin pisar las viejas).
8. **Repetir el mismo dato consolida la función** (respuestas más altas y planas) **pero no cristaliza
   el sustrato** (`dW_rel` sigue ~0.65 con lr alto). La cristalización necesita **contexto estable**
   (foco continuo en un set), no rotación de contexto.
9. Incluso con **1 imagen**, el conjunto de disparo cambia **casi cada época**.

---

## 9. Visualizadores (web local, `http.server` + numpy, sin dependencias)

- **`webapp.py`**: recorre las entradas; por cada una muestra **Original vs Negativo** lado a lado,
  con su mapa 50×50 de neuronas activas. Modos: *disparo digital* (umbral θ), *solo ganadora*,
  *activación completa*. Parámetro editable: **ms por entrada** (10–1000). Carga cualquier `model.npz`.
- **`webapp_evolution.py`**: para **1 imagen fija**, reproduce paso a paso cómo cambia el disparo con
  un **rastro de persistencia** (integrador con fuga por neurona): se aclara hacia blanco mientras
  dispara, se oscurece hacia negro mientras sigue apagada → resaltan las **persistentes**. Editables:
  ms por paso, velocidad del rastro, θ.
- Exposición pública temporal sin hosting: **cloudflared** quick tunnel
  (`cloudflared tunnel --url http://localhost:8000 --protocol http2`; el `--protocol http2` evita el
  fallo si la red bloquea QUIC/UDP).

---

## 10. Entorno y cómo ejecutar

- **venv Python 3.12**, paquetes: `numpy`, `matplotlib`, `Pillow` (PIL). **No requiere PyTorch**
  (todo el modelo es numpy). Imágenes/GIF con PIL; gráficas con matplotlib (Agg).
- Ejemplos:
```bash
python hebbian/generate_lines.py --n 1000
python hebbian/train.py --inhib --inhib-gain 1.5 --reinforce-gain 1.0 --epochs 10 --lr0 0.1 --lr-min 0.1
python hebbian/train_series.py --total 50 --snapshot-every 10 --inhib-gain 1.5   # (train_series activa inhib)
python hebbian/analyze_series.py --dir experiments/<serie> --dataset <npz>
python hebbian/webapp.py --model experiments/<run>/model.npz --dataset <npz>
```

---

## 11. Preguntas abiertas / próximos pasos

- ¿Cristaliza el sustrato con lr→0 muy agresivo o hace falta **learning rate por-neurona** que decae
  con las victorias (compromiso)?
- Barrer la relación **datos/neuronas** para ubicar la transición cristalino↔líquido.
- Forzar **olvido real** reduciendo neuronas (`n_out` pequeño) para que los sets compitan por sustrato.
- Igualar la respuesta **pos/neg** (centrar la entrada antes de normalizar, o θ por-entrada).
- Escalar a más clases/dígitos reales (MNIST) y evaluar la representación con un clasificador lineal.

---

## 12. Qué más convendría guardar aquí (sugerencias)

- **Snapshots-hito** (`model.npz`) de los experimentos representativos + su `metrics.csv`, para no
  reentrenar al comparar (los grandes van gitignored; guardar al menos las semillas y comandos exactos).
- **Comandos exactos reproducibles** de cada figura/experimento (semilla, dataset, flags) — una tabla
  "experimento → comando → salida".
- **Rangos de los *sweeps*** ya hechos y sus tablas (inhib_gain, reinforce_gain) para no repetirlos.
- **Especificación de formatos** de los `.npz` (claves de `model.npz`, de los datasets, de `frames.npz`).
- **Definiciones formales de las métricas** (fórmulas) y umbrales por defecto (θ=0.40, etc.).
- **Notas de calibración negativas** (qué NO funcionó: annealing no estabiliza; `anti` descartado;
  conscience propuesta y frenada) para no re-litigar.
- **Decisiones de diseño con su porqué** (solo-refuerzo + inhibidores; inhib independiente del lr;
  fracción vs conteo; superpuestos vs ocupar celdas).
- **Glosario** de términos (líquido/cristalino, sustrato vs función, churn, persistencia).
- **Capturas/GIF** de referencia (campos receptivos, winners_map, rastro de persistencia) como "esto
  es lo que debería verse".
- **Requisitos y versiones** exactas (`requirements` mínimos: numpy, matplotlib, pillow).
