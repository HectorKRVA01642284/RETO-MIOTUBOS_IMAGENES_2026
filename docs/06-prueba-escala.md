# 06 · Prueba de escala con subset_10GB

Se procesaron los 3,113 PNG en escala de grises de 2592 × 1944 registrados en el manifiesto. Los PNG ocupan 10.00 GB decimales. Se usaron los pesos de la época 11 y el umbral 0.80 fijados en validación, con la misma huella SHA-256 de la prueba etiquetada.

| Medición en una corrida | Resultado |
|---|---:|
| Imágenes procesadas | 3,113 |
| Tiempo total | 117.85 s |
| Rendimiento de extremo a extremo | 26.42 imágenes/s |
| Lectura, decodificación y reducción | 73.83 s |
| Inferencia y transferencia de datos | 38.07 s |
| Binarización y escritura de máscaras | 5.92 s |

La corrida utilizó una NVIDIA RTX 3050 Laptop, PyTorch 2.11.0 con CUDA 12.8, inferencia en float32, lotes de cuatro y seis hilos de lectura. El tiempo total incluye leer, reducir, inferir y escribir el ZIP con el índice. Excluye verificar el inventario, cargar el modelo, calentar la GPU y preparar el panel. La etapa más lenta fue la lectura, decodificación y reducción. El resultado describe esta corrida y puede variar según el almacenamiento y el equipo.

La red marcó al menos un píxel en 1,381 imágenes y dejó 1,732 sin detección. La cobertura media fue de 0.219 % y el percentil 90 de 0.738 %. Estas cifras describen la salida del modelo. En una muestra por cobertura se observó que la red marca parte de un borde curvo de alto contraste. El subconjunto carece de máscaras de referencia, por lo que no permite estimar métricas de calidad ni la frecuencia de ese error.

El ZIP de 1.62 MB contiene una máscara PNG binaria de 648 × 486 por imagen, el índice `predicciones.csv` y un archivo de instrucciones. Cada píxel predicho cubre un bloque de 4 × 4 en el original. Se verificó la integridad del ZIP, la correspondencia de nombres con el manifiesto, las dimensiones y el recuento de píxeles de las 3,113 máscaras.

## Reproducir

```powershell
python procesar_subset.py "$env:USERPROFILE\Downloads\subset_10GB\subset_10GB" resultados/unet-piloto/mejor.pt resultados/mascaras-subset.zip resultados/subset-10gb.json --panel resultados/subset-ejemplos.png
```

El [registro de la corrida](../experimentos/subset-10gb.json) conserva la configuración y los tiempos medidos.
