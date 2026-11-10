# 04 · U-Net del piloto

El modelo usa las 196 parejas de entrenamiento y 42 de validación de la [partición provisional](02-particion-piloto.md). La prueba queda reservada para la comparación final.

La U-Net tiene tres escalas de reducción y conexiones entre el codificador y el decodificador. Recibe un canal en escala de grises y produce un valor por píxel que se convierte en probabilidad con una sigmoide. Las imágenes se reducen a 648 × 486, la misma cuadrícula que usa la referencia clásica. Las máscaras para entrenamiento se reducen con vecino más cercano. Las métricas de validación se calculan a resolución original al contar qué píxeles cubre cada bloque predicho de 4 × 4.

Por época se extraen cuatro recortes de 192 × 192 por imagen. Cuando hay píxeles anotados, la mitad de los recortes en promedio incluye uno elegido al azar. Se aplica el mismo volteo a imagen y máscara. La media y desviación estándar se calculan solo en entrenamiento.

Se usa AdamW con tasa de aprendizaje 0.001 y decaimiento de pesos 0.0001, lote de ocho y semilla 2026. La pérdida combina, a partes iguales, entropía cruzada binaria y pérdida Dice. La entropía cruzada da peso cuatro a los píxeles positivos para responder al desbalance. Cada época se evalúa sobre imágenes completas de validación. Se guarda la época y el umbral (de 0.05 a 0.95 en pasos de 0.05) con mayor Dice micro.

## Resultado en validación

El piloto completó 12 épocas en una NVIDIA RTX 3050 Laptop de 4 GB con PyTorch 2.11.0, CUDA 12.8 y precisión mixta. La red tiene 271,693 parámetros. La mejor combinación fue la **época 11 con umbral 0.80**.

| Métrica sobre las mismas 42 imágenes | Línea base clásica | U-Net |
|---|---:|---:|
| Dice micro | 0.0823 | 0.5196 |
| IoU micro | 0.0429 | 0.3510 |
| Precisión micro | 0.0534 | 0.4469 |
| Sensibilidad micro | 0.1792 | 0.6206 |
| Dice medio en las 37 imágenes con máscara positiva | 0.0737 | 0.5066 |
| Imágenes con máscara vacía donde se marcó al menos un píxel | 5 de 5 | 0 de 5 |

La U-Net aumentó el solapamiento con la referencia y redujo el número de píxeles marcados fuera de ella. La precisión de 0.447 indica que todavía hay falsos positivos frecuentes respecto a las etiquetas disponibles. Las máscaras vacías no se han confirmado como negativos revisados. Estos resultados sirvieron para elegir la época y el umbral. La estimación final se hará con las 42 imágenes reservadas de prueba y la configuración de ambos métodos ya fijada.

El [registro de las 12 épocas](../experimentos/unet-validacion.json) incluye las métricas, los recuentos y la configuración.

## Reproducir

Instala las dependencias en un entorno de Python 3.12. El siguiente ejemplo usa la distribución de PyTorch para NVIDIA CUDA 12.8. Para CPU, cambia `cu128` por `cpu` y añade `--device cpu` al entrenamiento. Consulta la [guía oficial de PyTorch](https://pytorch.org/get-started/previous-versions/) para otras plataformas.

```powershell
python -m pip install -r requirements.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python unet.py "$env:USERPROFILE\Downloads\vt\vt" particiones/piloto resultados/unet-piloto
python curvas_unet.py resultados/unet-piloto/validacion.json experimentos/linea-base-validacion.json resultados/curvas-unet.png
```

La corrida guarda `mejor.pt` (pesos, normalización, época y umbral) y `validacion.json` (métricas por época y configuración). El repositorio excluye los pesos generados y las imágenes de microscopía.
