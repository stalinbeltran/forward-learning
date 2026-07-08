# Comandos — visualización de entrenamiento (una sola línea)

Flujo para entrenar la capa competitiva Hebbiana sobre **una única imagen de
entrada** y ver, paso a paso en la webapp, cómo las neuronas van aprendiendo.

> El **entrenamiento y el servidor son independientes**:
>
> - `gen_evolution.py` entrena y escribe la secuencia en un archivo fijo
>   (`experiments/evolution/sequence.npz`).
> - `webapp_evolution.py` es un servidor puro que lee ese archivo. Cuando el
>   archivo cambia en disco, el botón **Refrescar** de la página muestra el nuevo
>   entrenamiento **sin reiniciar el servidor**.

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

Escribe/sobrescribe `experiments/evolution/sequence.npz`:

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
- `--out` — ruta del archivo de secuencia (por defecto el que lee el server).
- `--model experiments\smoke\model.npz` — (opcional) parte de una red ya
  entrenada en vez de pesos frescos.

Ejemplo con stop por convergencia (deja `--epochs` alto como cota; para solo):

```powershell
.venv\Scripts\python.exe hebbian\gen_evolution.py --dataset data\processed\hline\hline.npz --image-index 0 --epochs 300 --lr 0.15 --inhib --min-persistence 0.7
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

- `--dataset` — el `.npz` del set (paso 1).
- `--min-persistence` — fracción de persistencia que marca la convergencia de
  cada imagen (por defecto `0.7`); mismo criterio que `gen_evolution.py`.
- `--persist-patience` — épocas seguidas encendida para contar como persistente (`5`).
- `--max-epochs` — tope por imagen si nunca converge (por defecto `200`).
- `--lr` / `--inhib` — tasa de aprendizaje y malla de inhibición lateral.
- `--resume model.npz` — (opcional) parte de una red ya entrenada.
- Salida en `--run`: `model.npz` (red final) y `sequential.csv` (una fila por
  imagen con la época en que convergió, el ganador y su activación).

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

Luego abre: http://127.0.0.1:8000

- `--file` — secuencia a servir (por defecto `experiments/evolution/sequence.npz`).

---

## 4. Re-entrenar SIN reiniciar el servidor

Con el servidor del paso 3 corriendo, vuelve a ejecutar el **paso 2** (misma
imagen u otra) y pulsa **Refrescar** en la página. El servidor detecta el cambio
de archivo y muestra el nuevo entrenamiento. **No hace falta reiniciar nada.**

Solo necesitas reiniciar el servidor si cambiaste `--port`/`--file` o cerraste
el proceso. En ese caso, libera el puerto y relánzalo:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
.venv\Scripts\python.exe hebbian\webapp_evolution.py --port 8000
```

---

## Notas

- La red es **784 → 2500** (mapa 50×50).
- `ms/step` por defecto en la webapp: **40 ms** (ajustable con el slider).
- Paneles de la webapp: *Fixed image* (la entrada), *Firing (this epoch)* (qué
  neuronas disparan en ese paso) y *Persistence trail* (integrador con memoria:
  las neuronas persistentemente activas se vuelven blancas).
- El indicador de estado (arriba a la derecha) muestra la hora del archivo
  cargado (`mtime`), útil para confirmar que el Refrescar tomó la versión nueva.
