# Comandos — visualización de entrenamiento (una sola línea)

Flujo para entrenar la capa competitiva Hebbiana sobre **una única imagen de
entrada** y ver, paso a paso en la webapp, cómo las neuronas van aprendiendo.

> El **entrenamiento y el servidor son independientes**:
>
> - `gen_evolution.py` (y `train_sequential.py`) entrenan y escriben la secuencia
>   en **dos** sitios: el archivo fijo `experiments/evolution/sequence.npz` (el
>   "último") **y** una copia archivada con timestamp en
>   `experiments/evolution/runs/run_YYYYMMDD-HHMMSS.npz` que **nunca se
>   sobrescribe**. Así cada entrenamiento queda registrado para revisarlo luego.
> - `webapp_evolution.py` es un servidor puro que **agrupa** esos runs por la NN
>   que los generó. La cabecera tiene **dos selectores**: **NN** (redes entrenadas,
>   la más reciente arriba con ★) y **secuencia** (los entrenamientos de esa NN,
>   el más reciente por defecto). El botón **Refrescar** vuelve a leer todo desde
>   disco, así que un entrenamiento nuevo aparece **sin reiniciar el servidor**.
>
>   Identidad de la NN: un run que continúa un modelo guardado (`--model` /
>   `--resume`) se agrupa bajo ese `model.npz` (todos sus entrenamientos juntos);
>   un run **fresco** (pesos aleatorios nunca guardados) es una NN de un solo uso.

Prefijo del intérprete del entorno virtual: `.venv\Scripts\python.exe`

> **Mantenimiento:** este archivo es la referencia viva de comandos. Cada vez que
> un script cambie (nuevos flags, defaults distintos, rutas nuevas), **todos los
> comandos de aquí deben actualizarse** para reflejarlo. No dejar ejemplos
> desfasados.

---

## 1. Definir la imagen de entrada

La entrada la determina el dataset. Genera la línea que quieras
(`--angle` en grados: `0` = horizontal, `90` = vertical), centrada y sin jitter:

```powershell
.venv\Scripts\python.exe hebbian\single_line.py --angle 0 --out data\processed\hline\hline.npz
```

Parámetros:

- `--angle` — ángulo en grados (`0` = horizontal, `90` = vertical).
- `--offset-y` — desplazamiento vertical en px. **Negativo = más arriba**,
  positivo = más abajo.
- `--offset-x` — desplazamiento lateral en px. Positivo = derecha,
  negativo = izquierda.
- `--width` — grosor de la línea en px (por defecto `2`).
- `--size` — lado de la imagen (por defecto `28`).
- `--out` — ruta del `.npz` de salida.

Rango útil del offset: ~±13 px antes de que la recta se salga del canvas 28×28.
En una horizontal, `--offset-x` no cambia nada (la recta cruza todo el ancho); el
offset lateral se nota en verticales u oblicuas. Ejemplo, horizontal 5 px arriba:

```powershell
.venv\Scripts\python.exe hebbian\single_line.py --angle 0 --offset-y -5 --out data\processed\hline\hline.npz
```

---

## 2. Entrenar y generar la secuencia (especificando la entrada)

Escribe el archivo fijo `experiments/evolution/sequence.npz` **y** archiva una
copia con timestamp en `experiments/evolution/runs/` (para revisarla luego en el
selector del visor):

```powershell
.venv\Scripts\python.exe hebbian\gen_evolution.py --dataset data\processed\hline\hline.npz --image-index 0 --epochs 80 --lr 0.15 --inhib
```

Parámetros:

- `--dataset` — archivo `.npz` de imágenes a usar (tu línea).
- `--image-index 0` — cuál imagen dentro del dataset (útil si el `.npz` tiene
  varias; con `lines.npz` puedes probar `--image-index 5`, etc.).
- `--epochs` / `--lr` — máximo de pasos (cota superior) y qué tan rápido aprende.
- `--inhib` — activa la inhibición lateral.
- `--min-persistence` — **stop por convergencia**. Detiene el entrenamiento
  cuando la *persistencia acumulada* alcanza esta fracción (p. ej. `0.7`): el
  0.7 del conjunto que dispara son las **mismas** neuronas encendidas sin
  interrupción ≥ `--persist-patience` épocas. Sin esta flag no hay early-stop.
- `--persist-patience` — épocas que una neurona debe llevar encendida seguida
  para contar como persistente (por defecto `5`).
- `--out` — ruta del archivo fijo de secuencia (por defecto el que lee el server).
- `--runs-dir` — carpeta donde se **archiva** cada corrida sin sobrescribir (por
  defecto `experiments/evolution/runs`); `--runs-dir ""` desactiva el archivado.
- `--model experiments\smoke\model.npz` — (opcional) parte de una red ya
  entrenada en vez de pesos frescos. Re-entrenar la **misma** NN así crea un run
  nuevo (queda en la lista); las corridas **no** se concatenan entre sí.

Ejemplo con stop por convergencia (deja `--epochs` alto como cota; para solo):

```powershell
.venv\Scripts\python.exe hebbian\gen_evolution.py --dataset data\processed\hline\hline.npz --image-index 0 --epochs 300 --lr 0.15 --inhib --min-persistence 0.7
```

**Regla de tabla de verdad** (`--learning-rule truth_table`, ver
[CLAUDE.md](CLAUDE.md)): actualización por conexión con `lr` chico. Con `lr=0.15`
colapsa a un oscilador global `0↔2500` en un paso; con `lr=0.001` sube suave y la
cola dispara primero (~época 39). Sin freno entra en ciclo límite, así que
conviene usar `--min-persistence` para congelar la primera meseta estable:

```powershell
.venv\Scripts\python.exe hebbian\gen_evolution.py --dataset data\processed\hline\hline.npz --image-index 0 --epochs 800 --lr 0.001 --inhib --learning-rule truth_table --min-persistence 0.7
```

---

## 2b. Analizar / graficar la convergencia de una secuencia

Lee `sequence.npz` y grafica las curvas de persistencia del conjunto que dispara
(retención, Jaccard y **persistencia acumulada** — esta última es el criterio de
convergencia, alineada con `--min-persistence` de arriba). Reporta la época de
convergencia y escribe `convergence.png` + `convergence.csv`:

```powershell
.venv\Scripts\python.exe hebbian\analyze_convergence.py --min-persistence 0.7 --patience 5
```

Parámetros:

- `--file` — secuencia a analizar (por defecto `experiments/evolution/sequence.npz`).
- `--min-persistence` — fracción de persistencia que marca la convergencia
  (debe coincidir con la usada al entrenar).
- `--patience` — épocas seguidas encendida para contar como persistente (= 5).
- `--out` / `--csv` — rutas del PNG y del CSV de salida.

---

## 2c. Entrenar un SET de rectas, imagen por imagen (secuencial)

Flujo distinto del de arriba: en vez de una sola imagen, se genera un **set** de
rectas y una **red nueva** las aprende **una a una** — cada imagen se presenta
repetidamente hasta cumplir el criterio de convergencia (persistencia acumulada,
condensado.md §7), y solo entonces se pasa a la siguiente, sobre la **misma** red.

Paso 1 — generar el set (10 rectas horizontales a distintas alturas):

```powershell
.venv\Scripts\python.exe hebbian\generate_hlines.py --n 10 --out data\processed\hlines_set\hlines.npz --preview
```

- `--n` — número de rectas (por defecto `10`).
- `--spread` — offset vertical máximo en px; las rectas se reparten en
  `[-spread, spread]` (por defecto `11`).
- `--width` / `--size` — grosor y lado de la imagen.
- `--preview` — escribe una tira PNG con todas las rectas para revisar a ojo.

Paso 2 — red nueva, entrenamiento secuencial hasta converger cada imagen:

```powershell
.venv\Scripts\python.exe hebbian\train_sequential.py --dataset data\processed\hlines_set\hlines.npz --run experiments\hlines_seq --min-persistence 0.7 --lr 0.15 --inhib
```

Con la **regla de tabla de verdad** y `lr` chico (así convergieron las 10 líneas
esta sesión, cada una con un ganador distinto; `--max-epochs` alto como cota):

```powershell
.venv\Scripts\python.exe hebbian\train_sequential.py --dataset data\processed\hlines_set\hlines.npz --run experiments\hlines_tt --min-persistence 0.7 --persist-patience 5 --lr 0.001 --inhib --learning-rule truth_table --max-epochs 400
```

- `--dataset` — el `.npz` del set (paso 1).
- `--min-persistence` — fracción de persistencia que marca la convergencia de
  cada imagen (por defecto `0.7`); mismo criterio que `gen_evolution.py`.
- `--persist-patience` — épocas seguidas encendida para contar como persistente (`5`).
- `--max-epochs` — tope por imagen si nunca converge (por defecto `200`).
- `--lr` / `--inhib` — tasa de aprendizaje y malla de inhibición lateral.
- `--resume model.npz` — (opcional) parte de una red ya entrenada.
- `--sequence` — archivo fijo de evolución a escribir para el visor (por defecto
  `experiments/evolution/sequence.npz`, el mismo que sirve `webapp_evolution.py`).
- `--runs-dir` — carpeta donde se **archiva** la corrida sin sobrescribir (por
  defecto `experiments/evolution/runs`); aparece en el selector del visor.
- Salida en `--run`: `model.npz` (red final) y `sequential.csv` (una fila por
  imagen con la época en que convergió, el ganador y su activación); **además**
  escribe la secuencia de evolución en `--sequence`.

**Verlo en el app:** este entrenador escribe la secuencia igual que
`gen_evolution.py`, así que con `webapp_evolution.py` corriendo (paso 3) basta
pulsar **Refrescar**. Como aquí se entrenan varias imágenes, el panel *Fixed
image* va cambiando y muestra la recta que se está entrenando en cada paso (el
archivo incluye un `imgseq` con la imagen por paso; los runs de una sola imagen
lo omiten y el panel queda fijo, como antes).

> Nota: con el criterio estricto por defecto (persistencia 0.7, umbral de disparo
> 0.40) no todas las imágenes convergen dentro del tope; el ganador aprende el
> coseno perfecto (`act≈1.0`) pero el borde del conjunto que dispara fluctúa por
> la inhibición. Para que converjan más, sube `--fire-threshold` (conjunto más
> chico y estable), baja `--min-persistence`, o sube `--max-epochs`.

---

## 3. Servir la visualización (se deja corriendo)

```powershell
.venv\Scripts\python.exe hebbian\webapp_evolution.py --port 8000
```

Luego abre: http://127.0.0.1:8000 (visor) o http://127.0.0.1:8000/test (pruebas).

En el visor, la cabecera tiene **dos selectores**: **NN** (las redes entrenadas,
la más reciente arriba con ★ y preseleccionada) y **secuencia** (los
entrenamientos de esa NN, el más reciente por defecto). Elige NN y luego la
secuencia que quieras revisar.

- `--file` — archivo fijo de respaldo (el "último"; por defecto
  `experiments/evolution/sequence.npz`).
- `--runs-dir` — carpeta de runs archivados a listar (por defecto
  `experiments/evolution/runs`; más reciente arriba).
- `--model` — modelo de la **NN actual** para `/test` (por defecto
  `lastexperiment/model.npz`).
- `--port` — puerto HTTP (por defecto `8000`).

En el flujo normal **no hace falta reiniciar** este servidor: para ver un nuevo
entrenamiento pulsa **Refrescar**, y para probar un modelo reentrenado pulsa
**Recargar NN** en `/test`. Solo se reinicia en los casos del §4b.

---

## 4. Re-entrenar SIN reiniciar el servidor

Con el servidor del paso 3 corriendo, vuelve a ejecutar el **paso 2** (misma
imagen u otra, con o sin `--model` para continuar la misma NN) y pulsa
**Refrescar** en la página. El nuevo entrenamiento aparece **arriba** (★): su NN
queda preseleccionada en el selector **NN** y su secuencia como la primera del
selector **secuencia**, y se carga; los anteriores siguen en las listas para
revisarlos. **No hace falta reiniciar nada.**
(De igual forma, para probar un modelo reentrenado en `/test`, pulsa **Recargar
NN**: relee `lastexperiment/model.npz` del disco sin reiniciar.)

## 4b. Reiniciar el servidor (cuándo y cómo)

Solo necesitas reiniciar si cambiaste un **flag de arranque** (`--port`,
`--file` o `--model`) o cerraste el proceso. En ese caso libera el puerto y
relánzalo (ajusta los flags que quieras):

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
.venv\Scripts\python.exe hebbian\webapp_evolution.py --port 8000
```

> **Nota — PowerShell vs cmd.exe:** todos los comandos de este archivo asumen
> **PowerShell** (el prompt suele terminar en `PS C:\...>`). Si tu prompt es
> `C:\...>` estás en cmd.exe y `Get-NetTCPConnection` dará
> *"no se reconoce como un comando"*: abre PowerShell escribiendo `powershell` y
> Enter. El comando `.venv\Scripts\python.exe ...` sí funciona en ambos. Para
> liberar el puerto **desde cmd.exe** el equivalente es:
>
> ```
> for /f "tokens=5" %a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /PID %a /F
> ```

---

## Referencia completa de parámetros

Todos los parámetros del proyecto, con su **valor por defecto actual**. Los
hiperparámetros del modelo viven en `CompetitiveLayer.__init__`
([hebbian/competitive_net.py](hebbian/competitive_net.py)) y se exponen como
flags en los scripts de entrenamiento; los flags propios de cada script van
después.

### A. Hiperparámetros del modelo (`CompetitiveLayer`)

Se persisten dentro de `model.npz` (`_HPARAMS`) y se reconstruyen en `load()`.

| Parámetro | Flag CLI | Default | Descripción |
|---|---|---|---|
| `n_in` | `--n-in` | `784` | dimensión de entrada (28×28). |
| `n_out` | (`--grid`²) | `2500` | nº de neuronas de salida = `grid_h·grid_w`. |
| `rule` | `--rule` | `above_mean` | gate de refuerzo de la regla base: `above_mean` \| `softmax` \| `wta`. |
| `reinforce_gain` | `--reinforce-gain` | `1.0` | ganancia global del refuerzo (solo regla `gate`). |
| `learning_rule` | `--learning-rule` | `gate` | regla de aprendizaje: `gate` (base continua) \| `truth_table` (por conexión). |
| `rule_n` | `--rule-n` | `1.1` | truth_table: factor de **aprendizaje disparado** (`+n·lr`). |
| `rule_m` | `--rule-m` | `0.3` | truth_table: factor de **desaprendizaje disparado** (`−m·lr`). |
| `rule_hr` | `--rule-hr` | `0.1` | truth_table: **inhibition rate** (`−hr`, fijo, no escalado por `lr`). |
| `grid_h`, `grid_w` | `--grid` | `50`, `50` | dimensiones del mapa de salida (cuadrado). |
| `inhib_on` | `--inhib` | `True` (`store_true`) | activa la malla de inhibición lateral. |
| `inhib_spacing` | `--inhib-spacing` | `5` | separación entre centros de inhibidores en la grilla. |
| `inhib_offset` | (no CLI) | `None` → `spacing//2` = `2` | desplazamiento del primer centro. |
| `inhib_radius` | `--inhib-radius` | `8` | radio de la región que cubre cada inhibidor. |
| `inhib_metric` | `--inhib-metric` | `cheby` | forma de la región: `cheby` (cuadrado) \| `manhattan` (rombo) \| `euclid` (disco). |
| `fire_threshold` | `--fire-threshold` | `0.40` | umbral de disparo (`a ≥ thr`); define "neurona disparada". |
| `inhib_K` | `--inhib-K` | `0.10` | fracción de la región encendida a partir de la cual el inhibidor "dispara". |
| `inhib_gain` | `--inhib-gain` | `1.5` | ganancia de la penalización de inhibición (solo regla `gate`). |
| `inhib_mode` | `--inhib-mode` | `fraction` | cómo se mide el exceso: `fraction` \| `hinge` \| `sigmoid` (solo regla `gate`). |
| `seed` | `--seed` | `0` | semilla RNG (init de pesos y barajado). |

> **`lr`** no es un atributo del modelo: se pasa por época a `learn_sample`/
> `train_epoch`, por eso permite annealing. En la regla `truth_table` escala a
> `+lr`, `+n·lr`, `−m·lr` (pero **no** a `−hr`). Valores usados esta sesión:
> `0.15` (colapso en 1 paso) y `0.001` (subida suave).

### B. `single_line.py` — una sola línea centrada

| Flag | Default | Descripción |
|---|---|---|
| `--angle` | `0.0` | ángulo en grados (`0` = horizontal, `90` = vertical). |
| `--size` | `28` | lado de la imagen. |
| `--width` | `2` | grosor de la línea en px. |
| `--offset-x` | `0.0` | desplazamiento lateral en px (positivo = derecha). |
| `--offset-y` | `0.0` | desplazamiento vertical en px (**negativo = arriba**). |
| `--out` | `data/processed/hline/hline.npz` | ruta de salida. |

### C. `generate_hlines.py` — set de N líneas horizontales

| Flag | Default | Descripción |
|---|---|---|
| `--n` | `10` | nº de líneas. |
| `--size` | `28` | lado de la imagen. |
| `--width` | `2` | grosor en px. |
| `--spread` | `11.0` | offset vertical máx en px; se reparten en `[-spread, spread]`. |
| `--out` | `data/processed/hlines_set/hlines.npz` | ruta de salida. |
| `--preview` | (off) | escribe una tira PNG con todas las líneas. |

### D. `generate_lines.py` — N líneas con ángulo/posición aleatorios

| Flag | Default | Descripción |
|---|---|---|
| `--n` | `1000` | nº de imágenes. |
| `--seed` | `0` | semilla RNG. |
| `--size` | `28` | lado de la imagen. |
| `--out` | `data/processed/lines_hebbian/lines.npz` | ruta de salida. |
| `--negatives` | (off) | guarda también los negativos (`255 - img`) en la clave `images_neg`. |
| `--preview` | (off) | escribe una grilla PNG `<out>.preview.png`. |

### E. `gen_evolution.py` — secuencia de una imagen fija

Además de todos los flags del modelo (sección A), tiene:

| Flag | Default | Descripción |
|---|---|---|
| `--dataset` | (requerido) | `.npz` de imágenes. |
| `--key` | `images` | clave del array de imágenes en el `.npz`. |
| `--model` | `None` | parte de un `model.npz` ya entrenado (si no, pesos frescos). |
| `--image-index` | `0` | cuál imagen del dataset usar como entrada fija. |
| `--epochs` | `80` | máximo de épocas (cota superior). |
| `--lr` | `0.15` | learning rate (constante). |
| `--min-persistence` | `None` | early-stop: para al alcanzar esta fracción de persistencia acumulada. |
| `--persist-patience` | `5` | épocas seguidas encendida para contar como persistente. |
| `--out` | `experiments/evolution/sequence.npz` | archivo fijo de secuencia a (sobre)escribir. |
| `--runs-dir` | `experiments/evolution/runs` | carpeta donde se archiva cada corrida (nunca se sobrescribe); `""` la desactiva. |

### F. `train.py` — entrenamiento config-driven (baraja todo el set)

Además de todos los flags del modelo (sección A), tiene:

| Flag | Default | Descripción |
|---|---|---|
| `--dataset` | `data/processed/lines_hebbian/lines.npz` | `.npz` de imágenes. |
| `--key` | `images` | clave del array en el `.npz`. |
| `--run` | `experiments/run` | directorio de salida (`model.npz` + `metrics.csv`). |
| `--epochs` | `10` | nº de épocas. |
| `--lr0` | `0.1` | learning rate inicial. |
| `--lr-min` | `0.1` | learning rate final (anneal lineal; igual a `--lr0` = constante). |
| `--resume` | `None` | reanuda desde un `model.npz`. |

### G. `train_sequential.py` — entrena imagen por imagen hasta converger

Además de todos los flags del modelo (sección A), tiene:

| Flag | Default | Descripción |
|---|---|---|
| `--dataset` | `data/processed/hlines_set/hlines.npz` | `.npz` del set. |
| `--key` | `images` | clave del array en el `.npz`. |
| `--run` | `experiments/hlines_seq` | directorio de salida (`model.npz` + `sequential.csv`). |
| `--lr` | `0.15` | learning rate (constante). |
| `--max-epochs` | `200` | tope por imagen si nunca converge. |
| `--min-persistence` | `0.7` | fracción de persistencia que marca convergencia de cada imagen. |
| `--persist-patience` | `5` | épocas seguidas encendida para contar como persistente. |
| `--sequence` | `experiments/evolution/sequence.npz` | archivo fijo de evolución a escribir para el visor. |
| `--runs-dir` | `experiments/evolution/runs` | carpeta donde se archiva la corrida (nunca se sobrescribe). |
| `--resume` | `None` | reanuda desde un `model.npz`. |

### H. `analyze_convergence.py` — análisis/gráfica de persistencia

| Flag | Default | Descripción |
|---|---|---|
| `--file` | `experiments/evolution/sequence.npz` | secuencia a analizar. |
| `--out` | `experiments/evolution/convergence.png` | PNG de salida (curvas de persistencia). |
| `--csv` | `experiments/evolution/convergence.csv` | CSV por época. |
| `--patience` | `5` | épocas seguidas encendida para contar como persistente. |
| `--min-persistence` | `0.7` | fracción que marca la convergencia (debe coincidir con la del entrenamiento). |

### I. Servidores web

`webapp_evolution.py` (visor de persistence trail + página de pruebas):

| Flag | Default | Descripción |
|---|---|---|
| `--file` | `experiments/evolution/sequence.npz` | archivo fijo de respaldo (el "último"). |
| `--runs-dir` | `experiments/evolution/runs` | carpeta de runs archivados a listar en el selector (más reciente arriba). |
| `--model` | `lastexperiment/model.npz` | modelo de la **NN actual** para la página `/test` (el último experimento; se recarga por `mtime`, no queda fijo). |
| `--port` | `8000` | puerto HTTP. |

El mismo servidor sirve **tres páginas** (mismo puerto), enlazadas entre sí desde
la cabecera del visor:

- `/` — **visor de entrenamiento** (persistence trail; §3 arriba).
- `/test` — **análisis agregado** de aplicar sets a la NN actual.
- `/apply` — **replay visual** de aplicar un set a la NN entrenada, entrada por
  entrada.

**Página de pruebas (`/test`):** botón **"Probar NN 🔬"** →
`http://127.0.0.1:8000/test`. Se elige entre todos los sets de `data/` (incluidos
los de entrenamiento) cuáles probar sobre la NN actual; cada set se **valida
contra la dimensión de entrada** de la red (los incompatibles quedan bloqueados y
no se aplican) y se muestran métricas agregadas (cobertura, neuronas ganadoras,
activación del ganador, disparos por entrada, neuronas muertas), un mapa 50×50 de
ganadoras, un mapa de fracción de disparo y el detalle por dataset y por entrada.

**Página de aplicación (`/apply`):** botón **"Aplicar set 🎯"** →
`http://127.0.0.1:8000/apply`. Reproduce, **entrada por entrada**, la respuesta de
la NN **congelada** (sin aprendizaje) sobre el dataset elegido en el desplegable
(solo se listan los compatibles; el cómputo se valida igual que en `/test`). Tres
paneles: *Input (this one)* (la entrada actual), *Firing (this input)* (neuronas
que disparan con esa entrada + ganador) y *Uso acumulado* (trail **monotónico**:
las neuronas se iluminan al disparar y no se apagan, construyendo el mapa de uso
del set conforme avanzas). Controles: dataset, play/pausa, ←/→, `ms/entrada`,
`trail speed`, `θ` y **Reset trail**.

En ambas páginas el botón **"Recargar NN"** vuelve a leer el modelo del disco, así
que tras reentrenar (nuevo `lastexperiment/`) basta pulsarlo — sin reiniciar el
servidor.

`webapp.py` (visor Original vs Negativo):

| Flag | Default | Descripción |
|---|---|---|
| `--model` | (requerido) | `model.npz` a cargar. |
| `--dataset` | (requerido) | `.npz` de imágenes. |
| `--key` | `images` | clave del array de imágenes. |
| `--neg-key` | `None` | clave del array de negativos (si aplica). |
| `--port` | `8000` | puerto HTTP. |

---

## Notas

- La red es **784 → 2500** (mapa 50×50).
- `ms/step` por defecto en la webapp: **40 ms** (ajustable con el slider).
- Paneles de la webapp: *Fixed image* (la entrada), *Firing (this epoch)* (qué
  neuronas disparan en ese paso) y *Persistence trail* (integrador con memoria:
  las neuronas persistentemente activas se vuelven blancas).
- Casilla **"pausar antes de cambiar de imagen"** (activada por defecto): en un
  run secuencial la animación se detiene en el **último frame** de cada imagen
  (su estado final) antes de pasar a la siguiente entrada; pulsar **Play**
  reanuda y cruza a la imagen siguiente. En runs de una sola imagen no hay
  cambios de entrada, así que no pausa.
- Botón **"Saltar ⏭"**: salta directo al **primer frame de la siguiente
  entrada** sin esperar la animación. El *persistence trail* se recalcula sobre
  los frames omitidos, así que el estado al que saltas es fiel (como si hubiera
  corrido). En el último bloque, salta de vuelta a la primera imagen.
- El indicador de estado (arriba a la derecha) muestra la hora del archivo
  cargado (`mtime`), útil para confirmar que el Refrescar tomó la versión nueva.
