### EXP-02: resultados

| job_id | Modelo | Configuración | Semilla | Validación | Prueba | Tiempo (s) | Nodo | Reasig. |
|---|---|---|---|---|---|---|---|---|
| `3c7eb35b3b2f7998` | cnn | `{"batch_size":64,"class_weight":"balanced","conv_layers":2,"epochs":72,"filters":8,"img_size":32,"lr":0.001,"weight_decay":0.0}` | 42 | acc=0.6987 | acc=0.7820, bal=0.4405 | 70.4 | b2eae31f298a | 0 |
| `53e511e1176b9f8f` | cnn | `{"batch_size":64,"class_weight":"none","conv_layers":2,"epochs":72,"filters":8,"img_size":32,"lr":0.001,"weight_decay":0.0}` | 42 | acc=0.7117 | acc=0.7810, bal=0.4080 | 59.1 | b2eae31f298a | 0 |

- Mejor de `req-cd64f6afa65a`: `53e511e1176b9f8f` (val_acc=0.711715)
