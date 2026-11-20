# Avance: depuración y modelo provisional E23

Se adopta provisionalmente el combinado **Swin-T + U-Net/ResNet-18**, con reducción por 4 y umbral guardado **0.57**. Los ensayos posteriores no demostraron una mejora material que justificara sustituirlo. No se cambian sus pesos para esta publicación.

## Depuración de datos

Se auditaron 306 parejas de `vt`. Se apartaron de las particiones 23 archivos por imágenes duplicadas con máscaras conflictivas y tres por máscaras de cobertura extrema. Quedan 280 parejas: **196 entrenamiento, 42 validación y 42 de prueba histórica**. Los originales no se borraron ni se modificaron. Se conservan 34 máscaras vacías cuyo significado sigue pendiente de confirmar.

La entrada se convierte a gris y se normaliza con estadísticas de entrenamiento. Se prepararon 15 máscaras derivadas de entrenamiento desde las anotaciones existentes, sin añadir nuevas etiquetas humanas. Esto es depuración y preparación de datos: **no es un filtrado visual de ruido validado sobre todas las imágenes**. El conjunto `subset_10GB` no se usó para ajustar esta campaña.

## Resultados comparables

Mismas 42 imágenes de validación; precisión y sensibilidad de **píxeles**. Las columnas comparan modelos dentro de una misma referencia.

| Referencia | Modelo | Precisión | Sensibilidad | Dice | IoU |
|---|---|---:|---:|---:|---:|
| PNG original, primaria histórica | Combinado anterior | 52.99 % | 64.12 % | 0.5802 | 0.4087 |
| PNG original, primaria histórica | E23 provisional | **58.68 %** | **64.18 %** | **0.6131** | **0.4420** |
| YOLO original, complementaria | Combinado anterior | 55.36 % | 64.88 % | 0.5974 | 0.4260 |
| YOLO original, complementaria | E23 provisional | **61.35 %** | **64.99 %** | **0.6312** | **0.4611** |

YOLO aquí significa el **formato de las anotaciones originales**, no que este modelo sea un detector YOLO. La auditoría encontró siete polígonos ausentes de los PNG: 64,666 píxeles adicionales en seis imágenes. Se evaluaron las mismas predicciones, sin entrenar de nuevo ni seleccionar otro umbral. Pasar de 58.68 % a 61.35 % entre referencias **no es aprendizaje adicional**. La mejora comparable frente al anterior es de 5.69 puntos en PNG o 5.99 puntos en YOLO.

## Límites que deben acompañar las cifras

- De 181 componentes anotados previamente cubiertos al menos a la mitad, E23 conserva 171 a ese nivel. Diez bajan de la mitad y **dos tienen solapamiento cero**. Esos componentes no son instancias celulares certificadas.
- La validación se reutilizó en numerosos ensayos y las anotaciones son parciales. No hay una prueba independiente nueva ni confirmación biológica de las detecciones fuera de la referencia.
- La agrupación por UUID de nombre no acredita independencia entre muestras biológicas.
- El candidato tarda aproximadamente tres veces más que el original en una medición local descriptiva de dos imágenes en RTX 3050; no mejora todos los aspectos.
- Una mayor cantidad de épocas no garantizó mejorar el equilibrio entre precisión y cobertura. La meta primaria de 60 % con PNG y conservación completa sigue pendiente.

## Uso y evidencia

El [perfil publicado](../configuracion/modelo-provisional.json) identifica exactamente el checkpoint y sus métricas. La [presentación de dos diapositivas](../presentaciones/Avance_modelo_provisional.pptx) resume los avances e incluye notas para exponerlos.

Los registros seleccionados contienen [conteos globales](../experimentos/E23-globales.json), [verificación y cobertura](../experimentos/E23-verificacion.json) y [comparación de referencias](../experimentos/E29-versiones.json). Algunas rutas y hashes de esos registros apuntan al archivo histórico local de la campaña; se conservan como procedencia, no como archivos adicionales obligatorios para inferencia.

Solo se publican el código necesario, este resumen, tres registros de evidencia, el perfil y la presentación. Los modelos descartados y los conjuntos de datos permanecen fuera de Git. El checkpoint elegido se distribuye como un único archivo en Releases, con SHA-256, por superar el límite habitual de tamaño de los archivos del repositorio.

El siguiente paso es reducir las omisiones y evaluar con referencias más completas e independientes, antes de atribuir una mejora generalizable.
