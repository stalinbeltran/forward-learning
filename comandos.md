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

---

## 1. Definir la imagen de entrada

La entrada la determina el dataset. Genera la línea que quieras
(`--angle` en grados: `0` = horizontal, `90` = vertical), centrada y sin jitter:

```powershell
.venv\Scripts\python.exe hebbian\single_line.py --angle 0 --out data\processed\hline\hline.npz
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
- `--epochs` / `--lr` — cuántos pasos y qué tan rápido aprende.
- `--inhib` — activa la inhibición lateral.
- `--out` — ruta del archivo de secuencia (por defecto el que lee el server).
- `--model experiments\smoke\model.npz` — (opcional) parte de una red ya
  entrenada en vez de pesos frescos.

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
