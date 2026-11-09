# 01 · Auditoría de imágenes y anotaciones

**Estado:** revisión técnica completada. La interpretación de las etiquetas ambiguas y el origen de las imágenes quedan pendientes de confirmación.

Se revisaron las 306 parejas de `vt/images` y `vt/masks_png`. Todas las parejas coinciden por nombre y miden 2592 × 1944 píxeles. Las máscaras solo contienen 0 y 1. El 1.76 % de los píxeles está marcado como miotubo. Las 34 máscaras vacías tienen un archivo YOLO vacío y cero instancias en COCO. Esto confirma la coherencia de los archivos, pero queda por confirmar si son negativos revisados o imágenes pendientes de anotar.

| Hallazgo | Evidencia | Decisión propuesta para el piloto |
|---|---|---|
| 11 grupos con fotos idénticas y máscaras distintas | 23 archivos. Se compararon los píxeles decodificados y luego las máscaras de cada grupo. | Reservar los 23 hasta acordar una máscara de referencia por foto. Mantener cada grupo en una sola partición si se reincorpora. |
| Tres máscaras de cobertura extrema | Cada archivo YOLO original contiene un solo rectángulo alineado con los ejes. La máscara reproduce esa anotación. | Reservar los tres hasta confirmar qué representan los rectángulos. |
| 34 máscaras vacías | Sin polígonos en YOLO ni instancias en COCO. | Confirmar si son negativos revisados antes de contarlos como fondo. |
| 12 etiquetas sueltas y una vista de verificación faltante | Las etiquetas sueltas carecen de foto. La vista faltante sí tiene su pareja imagen–máscara. | Registrar el hallazgo. La ausencia de la vista de verificación no invalida la pareja. |

Los 23 duplicados y las tres máscaras rectangulares son casos distintos. Reservarlos deja **280 parejas candidatas**, incluidas las 34 máscaras vacías. Si estas siguen pendientes y se reservan también, quedarían **246 parejas candidatas para un piloto con máscaras positivas**. Estas cifras describen escenarios de revisión, no un conjunto de entrenamiento o prueba ya fijado.

## Rectángulos para confirmar

| Imagen | Cobertura de la máscara |
|---|---:|
| `31b7e487-42d5b90b-a80c-4524-b17f-99018fff2a56_A3_02_03_Best.png` | 72.79 % |
| `07722ad5-42d5b90b-a80c-4524-b17f-99018fff2a56_A3_03_04_Best.png` | 47.03 % |
| `857fb71d-1cb78525-a7e5-4436-8262-f33beb93d74d_B1_02_06_Best.png` | 29.25 % |

## Antes de separar entrenamiento, validación y prueba

1. Confirmar el significado de las máscaras vacías y de los rectángulos con quien preparó las anotaciones.
2. Pedir el mapa de procedencia de las imágenes, o la interpretación de los nombres. El COCO solo registra identificador, nombre y dimensiones por imagen. Los identificadores repetidos en los nombres son una pista, pero no acreditan por sí solos una placa, un pozo o una muestra independiente.
3. Con esa información, registrar las exclusiones y generar una partición por grupo cercana a 70/15/15. Revisar que cada grupo aparezca en una sola partición antes de medir desempeño.

Para ver los nombres de los 11 grupos, volver a ejecutar `auditar.py` como indica el README. El JSON incluye el campo `duplicate_mask_comparison` y el panel `casos_para_revisar.png` muestra un par en conflicto y la máscara de mayor cobertura.
