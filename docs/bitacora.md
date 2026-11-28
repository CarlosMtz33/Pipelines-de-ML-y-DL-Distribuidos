# Bitácora de experimentos

**Equipo:** Carlos Samuel Martínez Medrano, Jonathan Gamaliel Corona Silva, Alex Jael Tenjhay Ramírez, Diego Moreno Ojeda, Daniel Santiago Párraga Anguiano.

Todos los trabajos se enviaron **por el sistema distribuido** (`docker compose run --rm client batch ...`). Las tablas de resultados salen del comando `client bitacora`.

**Datos comunes a todos los experimentos:** nuScenes `v1.0-mini`, versión de datos `mini-795db8e5acae`. Partición oficial de nuScenes: prueba = `mini_val` (escenas 0103 y 0916); entrenamiento + validación = `mini_train`, de donde se apartan las escenas 0796 y 1094 para validar. Regresión: 18,538 objetos, objetivo `num_lidar_pts`, variables distancia, ancho, largo, alto, visibilidad y categoría. CNN: 13,341 recortes de 64 × 64 en 5 clases (moto, carro, autobús, peatón, bicicleta).

---

## REF: conjunto de referencia

| Campo | Contenido |
|---|---|
| ID | REF · `req-b9e49b7477e3` (lineal), `req-2620370bdf68` (MLP), `req-badac2382698` (CNN) |
| Hipótesis | _(escríbanla ustedes: ¿qué esperaban del MLP contra la regresión lineal?, ¿y de la CNN con más filtros o imágenes más grandes?)_ |
| Modelo y datos | Los tres modelos, con los datos comunes de arriba |
| Configuración | `grids/referencia.json` (34 trabajos), semilla 42. CNN con 72 épocas |
| Resultados | Ver `docs/bitacora_REF.md`. Mejor MLP: `e1829d32adea5b2b`, val_rmse = 58.76. Mejor CNN: `3ea99f4271e4eee6`, val_acc = 0.73 |
| Conclusión | _(escríbanla ustedes)_ |

---

## EXP-01: predecir log(1 + puntos de lidar)

| Campo | Contenido |
|---|---|
| ID | EXP-01 · `req-ebcab39e1ae3` (lineal), `req-f0f26ececa35` (MLP) |
| Hipótesis | _(escríbanla ustedes antes de correrlo. Pista: la mayoría de los objetos tiene menos de 20 puntos, pero algunos cercanos tienen miles)_ |
| Modelo y datos | Regresión lineal (Ridge, alpha 1) y MLP ([64, 32], lr 0.001, 60 épocas) |
| Configuración | `grids/experimentos/exp01_log_objetivo.json`: `target_transform` = `none` contra `log1p`, semilla 42 |
| Resultados | _(pegar la salida de `client bitacora`)_ |
| Conclusión | _(escríbanla ustedes)_ |

---

## EXP-02: pesos por clase en la CNN

| Campo | Contenido |
|---|---|
| ID | EXP-02 · `req-cd64f6afa65a` |
| Hipótesis | _(escríbanla ustedes. Pista: en entrenamiento hay 3,955 carros y solo 118 bicicletas y 164 motos)_ |
| Modelo y datos | CNN de 8 filtros, imágenes de 32 px, lr 0.001, 72 épocas |
| Configuración | `grids/experimentos/exp02_pesos_clase.json`: `class_weight` = `none` contra `balanced`, semilla 42 |
| Resultados | _(pegar la salida de `client bitacora`; comparen también la diagonal de la matriz de confusión normalizada con `client status <job_id>`)_ |
| Conclusión | _(escríbanla ustedes)_ |

---

## EXP-03: número de capas convolucionales

| Campo | Contenido |
|---|---|
| ID | EXP-03 · `req-be719eb726fd` |
| Hipótesis | _(escríbanla ustedes: ¿más capas da mejor exactitud? ¿cuánto más tarda?)_ |
| Modelo y datos | CNN de 8 filtros, imágenes de 32 px, lr 0.001, 72 épocas |
| Configuración | `grids/experimentos/exp03_capas_cnn.json`: `conv_layers` = 1, 2 y 3, semilla 42 |
| Resultados | _(pegar la salida de `client bitacora`)_ |
| Conclusión | _(escríbanla ustedes)_ |

---

## EXP-04: planificación FIFO contra LPT (experimento del sistema)

| Campo | Contenido |
|---|---|
| ID | EXP-04 · `req-f5ad15c01de4` (el mismo lote, corrido dos veces) |
| Hipótesis | _(escríbanla ustedes: ¿repartir primero los trabajos más largos reduce el tiempo total?)_ |
| Modelo y datos | 4 CNN de costo muy distinto (8 o 16 filtros × 32 o 64 px, 24 épocas) en 2 nodos con una CNN por nodo a la vez |
| Configuración | `grids/experimentos/exp04_planificacion.json`, una vez con `SCHEDULING=fifo` y otra con `SCHEDULING=lpt`, empezando cada una con `docker compose down -v` |
| Resultados | Tiempo total con FIFO: _(s)_. Tiempo total con LPT: _(s)_. Qué nodo corrió cada CNN y en qué orden: _(del log del coordinador)_ |
| Conclusión | _(escríbanla ustedes)_ |
