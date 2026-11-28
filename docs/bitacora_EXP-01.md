### EXP-01: resultados

| job_id | Modelo | Configuración | Semilla | Validación | Prueba | Tiempo (s) | Nodo | Reasig. |
|---|---|---|---|---|---|---|---|---|
| `b78c9fd4eae893e3` | linear | `{"alpha":1.0,"features":["distance","width","length","height","visibility","category"],"normalize":true,"target":"num_lidar_pts","target_transform":"none","variant":"ridge"}` | 42 | RMSE=187.312, R²=-0.176 | RMSE=160.461, R²=-0.198 | 0.1 | b2eae31f298a | 0 |
| `582217b9644bb0d2` | linear | `{"alpha":1.0,"features":["distance","width","length","height","visibility","category"],"normalize":true,"target":"num_lidar_pts","target_transform":"log1p","variant":"ridge"}` | 42 | RMSE=165.093, R²=0.086 | RMSE=137.408, R²=0.121 | 0.0 | eaaab3b02699 | 0 |
| `4d31932c4da6aa4a` | mlp | `{"batch_size":256,"epochs":60,"features":["distance","width","length","height","visibility","category"],"hidden":[64,32],"lr":0.001,"normalize":true,"target":"num_lidar_pts","target_transform":"none","weight_decay":0.0}` | 42 | RMSE=95.654, R²=0.693 | RMSE=145.570, R²=0.014 | 16.1 | b2eae31f298a | 0 |
| `b2faa27ca6749b72` | mlp | `{"batch_size":256,"epochs":60,"features":["distance","width","length","height","visibility","category"],"hidden":[64,32],"lr":0.001,"normalize":true,"target":"num_lidar_pts","target_transform":"log1p","weight_decay":0.0}` | 42 | RMSE=72.296, R²=0.825 | RMSE=64.711, R²=0.805 | 2.0 | eaaab3b02699 | 0 |

- Mejor de `req-ebcab39e1ae3`: `582217b9644bb0d2` (val_rmse=165.093)
- Mejor de `req-f0f26ececa35`: `b2faa27ca6749b72` (val_rmse=72.2959)
