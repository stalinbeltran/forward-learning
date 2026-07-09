# Registro de experimentos — forward-learning

Bitácora **viva** de cada experimento (entrenamiento / generación de secuencia)
realizado sobre la red competitiva Hebbiana. Una fila por corrida, con su fecha,
el comando/script, el dataset, la regla de aprendizaje y los hiperparámetros que
la distinguen del default, más el resultado observado.

> **Mantenimiento (obligatorio):** cada vez que se corra un experimento, se
> agrega su fila **al final** de la tabla (más reciente abajo) y se **copia el
> último experimento** a la carpeta [lastexperiment/](lastexperiment/) (ver
> [CLAUDE.md](CLAUDE.md) → "Registro de experimentos"). Los defaults de cada flag
> están en [comandos.md](comandos.md) → "Referencia completa de parámetros"; aquí
> solo se anota lo que se apartó del default.

Las carpetas `experiments/` y `data/` están en `.gitignore` (artefactos
regenerables); esta bitácora y `lastexperiment/` son la memoria persistente de
qué se corrió y con qué parámetros.

## Tabla

| # | Fecha (hora local) | Experimento (`experiments/…`) | Script | Dataset | Regla | Hiperparámetros clave | Épocas | Resultado |
|--:|---|---|---|---|---|---|--:|---|
| 1 | 2026-07-08 06:55 | `smoke/` | `train.py` | `lines_hebbian/lines.npz` (1000 líneas) | `gate` (`above_mean`) | `lr0=0.1`, inhib default | 8 | Smoke test de `train.py`. `dead_units` baja 2345→1667, `coverage` sube ~0.027; red arranca a cubrir el mapa. |
| 2 | 2026-07-08 06:55 | `smoke_resume/` | `train.py --resume smoke/model.npz` | `lines_hebbian/lines.npz` | `gate` | `lr0=0.1`, reanuda desde `smoke` | 2 (ép. 9–10) | Prueba de reanudación. Continúa el descenso: `dead_units` 1597→1528. |
| 3 | 2026-07-08 09:47 | `hlines_seq/` | `train_sequential.py` | `hlines_set/hlines.npz` (10 h-líneas) | `gate` | `lr=0.15`, `--inhib`, `--min-persistence 0.7`, `--max-epochs 200` | 1523 tot. | Entrenamiento secuencial imagen-por-imagen. Solo **2/10** convergen (img 1 y 8); el resto agota las 200 épocas — el ganador aprende coseno ≈1.0 pero el borde del conjunto fluctúa. |
| 4 | 2026-07-08 15:14 | `tt_smoke/` | `train.py --learning-rule truth_table` | `lines_tt/lines.npz` | `truth_table` | `lr=0.15`, `n=1.1`, `m=0.3`, `hr=0.1` | 5 | Smoke de la regla de tabla de verdad con `lr` grande → **colapso**: oscilador global, `unique_winners=4` fijo, `dead_units=2478` congelado. Motiva usar `lr` chico. |
| 5 | 2026-07-08 15:43 | `evolution/` (`convergence.*`) | `gen_evolution.py` + `analyze_convergence.py` | `hline/hline.npz` (1 línea horizontal) | `truth_table` | `lr=0.001`, `--inhib`, `--min-persistence 0.7`, `epochs≈800` | ~800 | Evolución sobre una imagen fija para el visor. Análisis de convergencia: la cola dispara primero, sube suave; sin freno entra en ciclo límite (por eso `--min-persistence`). Genera `convergence.png/csv`. |
| 6 | 2026-07-08 15:53 | `hlines_tt/` | `train_sequential.py --learning-rule truth_table` | `hlines_set/hlines.npz` (10 h-líneas) | `truth_table` | `lr=0.001`, `n=1.1`, `m=0.3`, `hr=0.1`, `--inhib`, `--min-persistence 0.7`, `--persist-patience 5`, `--max-epochs 400` | 510 tot. | **Mejor resultado.** Las **10/10** líneas convergen, cada una con un ganador distinto (813, 112, 398, 1714, 1186, …), `persistence` 0.70–0.83. Conjunto que dispara chico y estable (59–146). Escribe también `evolution/sequence.npz` para el visor. |
| 7 | 2026-07-09 08:37 | `evolution/runs/` (`run_20260709-083648.npz`, `run_20260709-083701.npz`) | `gen_evolution.py` (×2) | `hline/hline.npz` (1 h-línea) | `gate` (run 1) · `truth_table` (run 2) | run 1: fresco, `--epochs 20`, `--lr 0.15`, `--inhib`; run 2: **misma NN** `--model lastexperiment/model.npz`, `--epochs 15`, `--lr 0.15`, `--inhib` | 20 · 15 | **Verificación del archivado por-run** (feature nueva). Cada corrida se archiva sin sobrescribir → 2 runs en `evolution/runs/`; el visor los lista con el más reciente arriba (★) y sirve cualquiera por `?file=`. Corridas de prueba (no un resultado de modelado). `lastexperiment/` **se dejó intacto** (sigue siendo el #6): `gen_evolution.py` no produce `model.npz`, así que no hay artefacto de modelo que copiar y sobrescribirlo borraría la NN canónica que usa `/test`. |

> **Nota sobre la reconstrucción inicial:** las filas 1–6 se reconstruyeron a
> partir de los artefactos (`metrics.csv` / `sequential.csv`), las fechas de sus
> archivos y [comandos.md](comandos.md); los hiperparámetros son los inferidos de
> esas fuentes, no un log exacto. De aquí en adelante cada fila se anota **en el
> momento de correr** el experimento, con los flags reales usados.

## Último experimento

La última corrida registrada es la **#7** (2026-07-09 08:37), pero fue una
**verificación** del archivado por-run de `gen_evolution.py` (sin `model.npz`).
Por eso [lastexperiment/](lastexperiment/) **sigue reflejando el #6 `hlines_tt`**
(2026-07-08 15:53) — el último experimento con un modelo real —, con su copia
completa (`model.npz` + `sequential.csv`) y `META.txt`. Es además la NN que la
página `/test` del visor usa como "NN actual", así que no se sobrescribió con una
corrida de prueba que la habría borrado.
