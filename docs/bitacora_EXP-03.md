### EXP-03: resultados

| job_id | Modelo | Configuración | Semilla | Validación | Prueba | Tiempo (s) | Nodo | Reasig. |
|---|---|---|---|---|---|---|---|---|
| `3044738d8d097ecb` | cnn | `{"batch_size":64,"class_weight":"balanced","conv_layers":1,"epochs":72,"filters":8,"img_size":32,"lr":0.001,"weight_decay":0.0}` | 42 | acc=0.6787 | acc=0.7725, bal=0.4093 | 44.9 | 68a5de144e46 | 0 |
| `3c7eb35b3b2f7998` | cnn | `{"batch_size":64,"class_weight":"balanced","conv_layers":2,"epochs":72,"filters":8,"img_size":32,"lr":0.001,"weight_decay":0.0}` | 42 | acc=0.6987 | acc=0.7820, bal=0.4405 | 69.5 | 68a5de144e46 | 0 |
| `88d8b10fcd4c5b9d` | cnn | `{"batch_size":64,"class_weight":"balanced","conv_layers":3,"epochs":72,"filters":8,"img_size":32,"lr":0.001,"weight_decay":0.0}` | 42 | acc=0.5619 | acc=0.6776, bal=0.3695 | 83.0 | 5bc35875ba7d | 0 |

- Mejor de `req-be719eb726fd`: `3c7eb35b3b2f7998` (val_acc=0.698745)
