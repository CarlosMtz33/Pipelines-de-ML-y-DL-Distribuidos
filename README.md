# Servidor distribuido de pipelines de entrenamiento

Proyecto final de Sistemas Distribuidos (UAQ, quinto semestre).

Sistema que recibe peticiones de entrenamiento de tres modelos (regresión lineal, MLP y CNN) sobre nuScenes mini, las convierte en trabajos con un `job_id` determinista y las reparte en paralelo entre varios nodos. Se comunica por gRPC, corre en contenedores Docker y tolera la caída de nodos y del coordinador.

## Integrantes

| Nombre | Matrícula |
|---|---|
| Carlos Samuel Martínez Medrano | 333383 |
| Jonathan Gamaliel Corona Silva | 294476 |
| Alex Jael Tenjhay Ramírez | 303196 |
| Diego Moreno Ojeda | 333354 |
| Daniel Santiago Párraga Anguiano | 333302 |
