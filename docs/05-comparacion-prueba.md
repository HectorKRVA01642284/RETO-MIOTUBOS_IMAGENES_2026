# 05 · Comparación en prueba reservada

Se evaluaron las 42 imágenes de `particiones/piloto/prueba.txt` con los ajustes fijados en validación. La línea clásica usa contraste local, Otsu × 0.9 y apertura y cierre de 3 × 3. La U-Net usa los pesos de la época 11 y umbral 0.80. Cada predicción a 648 × 486 se compara con la máscara de referencia a 2592 × 1944 mediante el conteo exacto de los bloques de 4 × 4 que cubre.

| Métrica de prueba | Línea clásica | U-Net |
|---|---:|---:|
| Dice micro | 0.0551 | **0.3711** |
| IoU micro | 0.0283 | **0.2278** |
| Precisión micro | 0.0321 | **0.2667** |
| Sensibilidad micro | 0.1954 | **0.6099** |
| Dice medio en las 37 imágenes con máscara positiva | 0.0585 | **0.3508** |
| Imágenes con máscara vacía donde se marcó al menos un píxel | 5 de 5 | **2 de 5** |

Las métricas micro se calculan después de sumar los recuentos de píxeles en todas las imágenes. El Dice de la U-Net baja de 0.5196 en validación a 0.3711 en prueba, aunque supera la referencia clásica por 0.316 puntos. Detecta alrededor del 61 % de los píxeles anotados y aproximadamente el 27 % de los píxeles que marca coincide con las anotaciones.

| Grupo por nombre | Imágenes | Dice clásico | Dice U-Net |
|---|---:|---:|---:|
| `2d2dc951` | 2 | 0.086 | 0.602 |
| `6a55a585` | 13 | 0.017 | 0.089 |
| `b1db384e` | 5 | 0.034 | 0.350 |
| `e5f15412` | 2 | 0.000 | No definido |
| `eebfb204` | 6 | 0.103 | 0.548 |
| `fc183976` | 14 | 0.083 | 0.433 |

La U-Net mejora en los cinco grupos con píxeles anotados. En `e5f15412`, tanto las máscaras de referencia como las predicciones de la U-Net son vacías. El denominador de Dice es cero para ese grupo. Los tamaños desiguales de los grupos y la falta de metadatos de procedencia limitan la interpretación como prueba de generalización entre experimentos.

El panel cualitativo incluye los casos de Dice máximo, mediano y mínimo entre las imágenes con máscara positiva, más la imagen de máscara vacía donde la red marcó más píxeles. En el caso favorable, varios tramos se alinean con la referencia. En el caso de Dice cero, la red marca estructuras alargadas fuera de los dos tramos anotados y no coincide con ninguno. Algunas de esas regiones podrían ser estructuras visibles sin anotar. Los resultados se interpretan contra la referencia disponible, cuya exhaustividad queda por confirmar.

## Reproducir

```powershell
python evaluar_prueba.py "$env:USERPROFILE\Downloads\vt\vt" particiones/piloto experimentos/linea-base-validacion.json experimentos/unet-validacion.json resultados/unet-piloto/mejor.pt experimentos/comparacion-prueba.json --panel resultados/comparacion-prueba.png
```

El [registro de prueba](../experimentos/comparacion-prueba.json) conserva los recuentos por imagen y grupo, las configuraciones y la huella SHA-256 de los pesos usados en la corrida original. Un entrenamiento nuevo puede generar pesos y resultados ligeramente distintos según el equipo y las versiones instaladas.
