# 03 · Línea base clásica

Se compararon 16 configuraciones en las **42 imágenes de validación** del piloto. Cada imagen se redujo a 648 × 486 y se suavizó con un filtro gaussiano. Se compararon dos maneras de resaltar estructuras oscuras: invertir la intensidad global y restar la imagen suavizada de un fondo estimado. Para cada respuesta se probó Otsu con cuatro multiplicadores y se comparó aplicar o no una apertura seguida de un cierre morfológico de 3 × 3. El umbral de Otsu se obtiene de cada imagen, sin usar su máscara.

Se eligió por Dice micro, calculado al acumular verdaderos positivos, falsos positivos y falsos negativos a resolución original. Cada píxel marcado a escala reducida cubre un bloque de 4 × 4 en la imagen original. La mejor configuración fue **contraste local, Otsu × 0.9 y morfología**.

| Métrica de validación | Resultado |
|---|---:|
| Dice micro | 0.0823 |
| IoU micro | 0.0429 |
| Precisión micro | 0.0534 |
| Sensibilidad micro | 0.1792 |
| Dice medio en las 37 imágenes con máscara positiva | 0.0737 |

El método marcó regiones en las cinco imágenes con máscara vacía. En la revisión visual aparecen bordes y manchas oscuras fuera de las anotaciones, y varios tramos de bajo contraste quedaron sin detectar. Las discrepancias se miden contra las máscaras disponibles. La revisión inicial dejó pendiente confirmar la condición de las máscaras vacías y el criterio con el que se anotaron tramos cortos.

Estos son resultados de selección en validación. La partición de prueba queda reservada para comparar esta configuración fija con la U-Net una vez entrenada.

```powershell
python linea_base.py "$env:USERPROFILE\Downloads\vt\vt" particiones/piloto experimentos/linea-base-validacion.json
```

El [resultado reproducible](../experimentos/linea-base-validacion.json) conserva los recuentos y métricas de las 16 configuraciones. El conteo por bloques se comprobó frente a un conteo directo a resolución original en tres imágenes.

## Revisión inicial de `subset_10GB`

- Se encontraron 3,113 PNG en escala de grises de 2592 × 1944 y un `manifest.csv`. El total ocupa 10.00 GB decimales.
- Los 3,113 nombres y tamaños coinciden con el manifiesto. Se pudieron abrir los encabezados de todos los PNG y decodificar por completo una muestra de 24 con semilla 2026.
- El manifiesto contiene rutas de origen, destino y tamaño en bytes. Las rutas comparten un UUID de origen que no coincide con los UUID reconocibles de `vt`. El inventario no contiene etiquetas ni identifica placas o muestras de `vt`.
- La comparación de tamaños y bytes no encontró PNG idénticos a los de `vt`. Podrían existir imágenes relacionadas o los mismos píxeles con otra codificación. Se reserva el subconjunto para medir rendimiento de procesamiento y revisar predicciones cuando haya un modelo.
