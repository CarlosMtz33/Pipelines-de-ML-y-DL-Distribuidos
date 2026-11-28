# Bitácora de experimentos

**Equipo:** Carlos Samuel Martínez Medrano, Jonathan Gamaliel Corona Silva, Alex Jael Tenjhay Ramírez, Diego Moreno Ojeda, Daniel Santiago Párraga Anguiano.

Todos los trabajos se enviaron **por el sistema distribuido** (`docker compose run --rm client batch ...`). Las tablas de resultados salen del comando `client bitacora`.

**Datos comunes a todos los experimentos:** nuScenes `v1.0-mini`, versión de datos `mini-795db8e5acae`. Partición oficial de nuScenes: prueba = `mini_val` (escenas 0103 y 0916); entrenamiento + validación = `mini_train`, de donde se apartan las escenas 0796 y 1094 para validar. Regresión: 18,538 objetos, objetivo `num_lidar_pts`, variables distancia, ancho, largo, alto, visibilidad y categoría. CNN: 13,341 recortes de 64 × 64 en 5 clases (moto, carro, autobús, peatón, bicicleta).

---

## REF: conjunto de referencia

| Campo | Contenido |
|---|---|
| ID | REF · `req-b9e49b7477e3` (lineal), `req-2620370bdf68` (MLP), `req-badac2382698` (CNN) |
| Hipótesis | Esperábamos que el MLP superara claramente a la regresión lineal: los puntos de lidar no crecen en línea recta con la distancia y el tamaño (caen aproximadamente con el cuadrado de la distancia), y una red con capas ocultas puede aprender esa curva. En la CNN esperábamos mejor exactitud con más filtros e imágenes de 64 px, a cambio de más tiempo de entrenamiento. |
| Modelo y datos | Los tres modelos, con los datos comunes de arriba |
| Configuración | `grids/referencia.json` (34 trabajos), semilla 42. CNN con 72 épocas |
| Resultados | Ver `docs/bitacora_REF.md`. Mejor MLP: `e1829d32adea5b2b`, val_rmse = 58.76. Mejor CNN: `3ea99f4271e4eee6`, val_acc = 0.73 |
| Conclusión | Se cumplió para la regresión: la regresión lineal tuvo R² negativo en prueba (entre −0.20 y −0.01, peor o igual que predecir siempre el promedio), mientras que el mejor MLP ([128, 64, 32], lr 0.0005) llegó a R² = 0.77 en prueba. Pero varios MLP con buen R² en validación salieron muy mal en prueba (por ejemplo, lr 0.005 dio R² de −1 a −3): solo la red más grande con lr bajo generalizó. En la CNN, 64 px subió un poco la exactitud de validación (0.72–0.73 contra 0.69–0.70), pero en prueba quedó casi igual (≈ 0.78) y tardó 4 veces más; 16 filtros no mejoró a 8. No vale la pena el costo extra. |

---

## EXP-01: predecir log(1 + puntos de lidar)

| Campo | Contenido |
|---|---|
| ID | EXP-01 · `req-ebcab39e1ae3` (lineal), `req-f0f26ececa35` (MLP) |
| Hipótesis | La mayoría de los objetos tiene muy pocos puntos de lidar (la mediana es 4), pero algunos objetos cercanos tienen miles. Esos valores extremos dominan el error cuadrático y jalan el ajuste. Si predecimos log(1 + puntos), la escala se comprime y esperamos que los dos modelos mejoren su R², sobre todo la regresión lineal. |
| Modelo y datos | Regresión lineal (Ridge, alpha 1) y MLP ([64, 32], lr 0.001, 60 épocas) |
| Configuración | `grids/experimentos/exp01_log_objetivo.json`: `target_transform` = `none` contra `log1p`, semilla 42 |
| Resultados | Ver `docs/bitacora_EXP-01.md`. R² en prueba: lineal −0.198 (sin log) → 0.121 (log1p); MLP 0.014 (sin log) → 0.805 (log1p) |
| Conclusión | Se cumplió, y mucho más en el MLP: su R² en prueba pasó de 0.01 a 0.81, y la regresión lineal de −0.20 a 0.12. Además, sin log el MLP tenía R² = 0.69 en validación pero 0.01 en prueba (los valores extremos lo hacían inestable); con log, validación y prueba coinciden (0.83 y 0.81). La regresión lineal sigue baja porque la relación no es lineal ni siquiera en escala logarítmica. |

---

## EXP-02: pesos por clase en la CNN

| Campo | Contenido |
|---|---|
| ID | EXP-02 · `req-cd64f6afa65a` |
| Hipótesis | En entrenamiento hay 3,955 carros y solo 118 bicicletas y 164 motos, así que sin pesos la red puede ignorar las clases raras y aun así tener buena exactitud. Esperamos que con class_weight = balanced suba la exactitud balanceada (y la diagonal de moto y bicicleta), aunque baje un poco la exactitud general. |
| Modelo y datos | CNN de 8 filtros, imágenes de 32 px, lr 0.001, 72 épocas |
| Configuración | `grids/experimentos/exp02_pesos_clase.json`: `class_weight` = `none` contra `balanced`, semilla 42 |
| Resultados | Ver `docs/bitacora_EXP-02.md`. Prueba: exactitud 0.781 (none) contra 0.782 (balanced); exactitud balanceada 0.408 (none) contra 0.441 (balanced) |
| Conclusión | Se cumplió, pero la mejora es pequeña: la exactitud balanceada en prueba subió de 0.408 a 0.441 y la exactitud general quedó igual (0.781 contra 0.782); en validación bajó un poco (0.712 a 0.699). Con tan pocas motos y bicicletas, los pesos por clase no alcanzan: harían falta más ejemplos o aumento de datos (por ejemplo, voltear las imágenes). |

---

## EXP-03: número de capas convolucionales

| Campo | Contenido |
|---|---|
| ID | EXP-03 · `req-be719eb726fd` |
| Hipótesis | Más capas convolucionales dan más capacidad, así que esperamos mejor exactitud con 2 y 3 capas que con 1, a cambio de más tiempo. Con imágenes de 32 px, 3 capas reducen la imagen a 4 × 4, así que la mejora podría detenerse en 2. |
| Modelo y datos | CNN de 8 filtros, imágenes de 32 px, lr 0.001, 72 épocas |
| Configuración | `grids/experimentos/exp03_capas_cnn.json`: `conv_layers` = 1, 2 y 3, semilla 42 |
| Resultados | Ver `docs/bitacora_EXP-03.md`. Prueba: exactitud 0.773 / 0.782 / 0.678 y balanceada 0.409 / 0.441 / 0.370 con 1 / 2 / 3 capas; tiempo 45 / 70 / 83 s. El trabajo de 2 capas es el mismo `job_id` del lote de referencia y, entrenado otra vez desde cero, dio exactamente las mismas métricas |
| Conclusión | Se cumplió solo en parte: 2 capas fue la mejor (exactitud 0.782 y balanceada 0.441 en prueba), 1 capa fue un poco peor (0.773 y 0.409) y 3 capas fue bastante peor (0.678 y 0.370) además de más lenta (83 s contra 70 s). Más capas no siempre ayuda: con imágenes de 32 px y 72 épocas, la red de 3 capas no alcanzó a aprender igual de bien. El trabajo de 2 capas es el mismo job_id del lote de referencia, y entrenado otra vez dio exactamente las mismas métricas, lo que confirma que el sistema es determinista. |

---

## EXP-04: planificación FIFO contra LPT (experimento del sistema)

| Campo | Contenido |
|---|---|
| ID | EXP-04 · `req-f5ad15c01de4` (el mismo lote, corrido dos veces) |
| Hipótesis | Con FIFO, si la CNN más pesada se reparte al final, un nodo la corre solo mientras el otro ya no tiene trabajo. Si se reparten primero los trabajos más largos (LPT), los dos nodos deberían terminar casi al mismo tiempo y el tiempo total del lote debería bajar. |
| Modelo y datos | 4 CNN de costo muy distinto (8 o 16 filtros × 32 o 64 px, 24 épocas) en 2 nodos con una CNN por nodo a la vez |
| Configuración | `grids/experimentos/exp04_planificacion.json`, una vez con `SCHEDULING=fifo` y otra con `SCHEDULING=lpt`, empezando cada una con `docker compose down -v` |
| Resultados | Tiempo total: **FIFO 236.5 s, LPT 190.4 s** (19.5 % menos). FIFO: un nodo corrió 5b16 (28 s) → 7a44 (44 s) → 7322 (162 s, la más pesada, al final) y el otro solo bdf6 (105 s), quedando más de 2 min sin trabajo. LPT: un nodo corrió 7322 (189 s) desde el inicio y el otro bdf6 (108 s) → 7a44 (43 s) → 5b16 (26 s); los dos terminaron con 11 s de diferencia. La CNN más pesada tardó más con LPT (189 s contra 162 s) porque corrió todo el tiempo junto con otra CNN en la misma computadora |
| Conclusión | Se cumplió: LPT terminó en 190.4 s contra 236.5 s con FIFO, un 19.5 % menos, y la diferencia entre el fin de los dos nodos bajó de unos 130 s a 11 s. La CNN más pesada tardó más con LPT (189 s contra 162 s) porque corrió todo el tiempo junto con otra CNN en la misma computadora, pero aun así LPT ganó. Por eso el coordinador usa LPT por defecto. |
