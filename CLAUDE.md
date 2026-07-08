# forward-learning

Red competitiva Hebbiana de una sola capa ("forward learning") reconstruida en
**numpy puro** (sin PyTorch) a partir de la especificación autocontenida en
[condensado.md](condensado.md).

## Entorno

- `.venv` (Python 3.14; numpy 2.5.1, matplotlib 3.11, Pillow 12.3).
- Intérprete en Windows: `.venv\Scripts\python.exe`
- `experiments/` y `data/` están en `.gitignore`: son artefactos **regenerables**
  (datasets, modelos, secuencias). No se versionan; se recrean con los comandos
  de abajo.

## La red

- Capa competitiva **784 → 2500** (mapa de salida 50×50).
- Aprende por reforzamiento Hebbiano competitivo + inhibición lateral. Los pesos
  `W` son `(n_out, n_in)` con filas de norma unitaria; la activación es la
  similitud coseno en `[-1, 1]`.
- Hay **dos reglas de aprendizaje** seleccionables (`learning_rule`):
  - `"gate"` (por defecto) — regla base continua: solo **refuerza** (gate ≥ 0) y
    toda reducción de peso viene de la malla de inhibidores laterales.
  - `"truth_table"` — regla **por conexión** de [hebbian/learning_rules.py](hebbian/learning_rules.py)
    (`TruthTableRule`), definida por la tabla de verdad de abajo.
- Núcleo en [hebbian/competitive_net.py](hebbian/competitive_net.py)
  (`CompetitiveLayer`: gate / inhibición / learn / save / load).

### Regla de la tabla de verdad (`hebbian/learning_rules.py`)

Cada peso `W[j, i]` cambia según tres señales **binarias** —entrada activa
(`|x_i|>0`), neurona `j` disparada (`a_j ≥ fire_threshold`) e inhibidora que la
cubre disparada— y cuatro parámetros: `lr` (paso base, por época), `n` (factor de
aprendizaje disparado), `m` (factor de desaprendizaje disparado) y `hr`
(inhibition rate, penalización **fija**, no escalada por `lr`).

| Entrada | Neurona disparada | Inhibidora disparada | Δpeso |
|:--:|:--:|:--:|:--:|
| 0 | 0 | 0 | `0` |
| 1 | 0 | 0 | `+lr` |
| 1 | 1 | 0 | `+n·lr` |
| 0 | 1 | 0 | `−m·lr` |
| 0 | 0 | 1 | `0` |
| 0 | 1 | 1 | `0` |
| 1 | 1 | 1 | `−hr` |

(`1 0 1` no aparece en la tabla → se asume `0`.) Con la inhibidora apagada
refuerza donde hay entrada y castiga el disparo sin entrada; con la inhibidora
encendida congela todo salvo el disparo-con-entrada, que recibe `−hr`. Defaults:
`n=1.1`, `m=0.3`, `hr=0.1` (flags `--learning-rule truth_table --rule-n/--rule-m/--rule-hr`).

## Módulos (`hebbian/`)

- `competitive_net.py` — núcleo de la capa.
- `generate_lines.py` — dataset de N líneas 28×28 con ángulo/posición aleatorios.
- `single_line.py` — dataset de **una sola** línea centrada a un ángulo fijo
  (`--angle 0` = horizontal, `90` = vertical). Útil para observar el aprendizaje
  sobre una entrada única.
- `train.py` — entrenamiento config-driven; guarda `model.npz` + `metrics.csv`.
- `metrics.py` — métricas compartidas (§7).
- `webapp.py` — visor Original vs Negativo.
- `gen_evolution.py` + `webapp_evolution.py` — visor de "persistence trail"
  (ver abajo).

## Visor de evolución del entrenamiento (desacoplado)

Arquitectura en **tres piezas independientes** para ver, paso a paso, cómo las
neuronas aprenden una entrada fija (una época = un paso de aprendizaje sobre esa
imagen):

1. **`single_line.py`** define la imagen de entrada.
2. **`gen_evolution.py`** entrena y escribe la secuencia de activaciones en un
   archivo fijo: `experiments/evolution/sequence.npz` (siempre el mismo nombre).
3. **`webapp_evolution.py`** es un servidor **puro**: lee ese archivo y lo sirve.
   Cachea por `mtime`; si el archivo cambia en disco, el botón **Refrescar** de
   la página muestra el nuevo entrenamiento **sin reiniciar el servidor**.

La página muestra tres paneles: *Fixed image* (la entrada), *Firing (this
epoch)* (qué neuronas disparan en ese paso) y *Persistence trail* (integrador
con memoria: las neuronas persistentemente activas se vuelven blancas).
`ms/step` por defecto: 40 ms.

### Comandos (ver [comandos.md](comandos.md) para el detalle)

```powershell
# 1) definir la entrada (0 = horizontal, 90 = vertical)
.venv\Scripts\python.exe hebbian\single_line.py --angle 0 --out data\processed\hline\hline.npz

# 2) entrenar + escribir experiments\evolution\sequence.npz
.venv\Scripts\python.exe hebbian\gen_evolution.py --dataset data\processed\hline\hline.npz --image-index 0 --epochs 80 --lr 0.15 --inhib

# 3) servir (dejar corriendo) -> http://127.0.0.1:8000
.venv\Scripts\python.exe hebbian\webapp_evolution.py --port 8000
```

Para **re-entrenar sin reiniciar el servidor**: repetir el paso 2 (misma imagen
u otra) y pulsar **Refrescar** en la página. Solo hay que reiniciar el server si
se cambió `--port`/`--file` o se cerró el proceso; en ese caso liberar el puerto
antes de relanzar (ver comandos.md §4).

## Estilo de trabajo (pedido por el usuario)

Construir en capas y verificar cada una ejecutándola antes de continuar.
Reportar resultados con la evidencia real (salida de los comandos).
