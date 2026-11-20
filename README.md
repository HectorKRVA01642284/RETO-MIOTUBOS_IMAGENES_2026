# Segmentación de miotubos en imágenes de microscopía

## Modelo provisional para continuar el proyecto

Se selecciona **Swin-T + combinado U-Net/ResNet-18 (E23)**. En las mismas 42 imágenes obtiene **61.35 % de precisión y 64.99 % de sensibilidad con las etiquetas YOLO originales**, como evaluación complementaria. La medición primaria histórica con PNG es **58.68 % y 64.18 %**. YOLO nombra el formato de las etiquetas; el modelo seleccionado es un combinado de segmentación con Swin-T.

Ambas mediciones usan los mismos pesos, predicciones y umbral: cambiar la referencia no representa aprendizaje adicional. Se adopta provisionalmente y conserva dos regiones completamente omitidas; la validación fue reutilizada y las etiquetas son parciales.

- [Resumen del avance, depuración y límites](docs/07-avance-modelo-provisional.md).
- [Presentación de dos diapositivas, con notas para exponer](presentaciones/Avance_modelo_provisional.pptx).
- [Perfil exacto, métricas y SHA-256](configuracion/modelo-provisional.json).
- [Checkpoint provisional en Releases](https://github.com/HectorKRVA01642284/RETO-MIOTUBOS_IMAGENES_2026/releases/tag/modelo-provisional-e23).

Se publican los avances necesarios para usar este candidato y comprobar sus resultados. Los datasets siguen en el OneDrive del curso, enlazado más abajo. Los pesos se distribuyen en Releases; los ensayos descartados permanecen en el archivo local de investigación.

### Ejecutar el candidato

Usa un entorno virtual con Python 3.11 o posterior compatible con PyTorch. Estas versiones corresponden al entorno de la campaña; la instalación siguiente permite inferencia en CPU. En una GPU utiliza una instalación de PyTorch compatible con tu dispositivo.

```bash
python -m pip install -r requirements-modelo.txt
python descargar_modelo.py
python predecir_checkpoint.py modelos/miotubos-e23-provisional.pt ruta/a/imagenes resultados/provisional --limite 3
```

La descarga comprueba tamaño y SHA-256. El checkpoint conserva **umbral 0.57 y reducción por 4**; no se elige otro umbral para mostrar mejores cifras. El exportador procesa PNG automáticamente y genera máscaras binarias 0/255 con la resolución de los originales y un manifiesto de ejecución. No necesita que se marquen regiones a mano. Quita `--limite 3` para procesar la carpeta completa.

Los resultados publicados se verificaron con lote 2 en la GPU de la campaña. El exportador portátil procesa una imagen a la vez; diferencias numéricas de lote o dispositivo pueden afectar píxeles muy próximos al umbral. Sus salidas nuevas son predicciones, no etiquetas confirmadas.

### Depuración completada

De 306 parejas auditadas, se apartaron 23 archivos duplicados con máscaras conflictivas y tres máscaras extremas. Quedan **280: 196 entrenamiento / 42 validación / 42 prueba histórica**. Los originales se conservan. La preparación incluye gris y normalización con estadísticas de entrenamiento. Un filtrado visual de ruido validado sigue pendiente; `subset_10GB` no se usó para ajustar esta campaña.

## Piloto histórico y acceso a los datos

Lo siguiente documenta el piloto previo y sus propios resultados; no atribuye sus métricas de prueba ni su velocidad al candidato E23.

El reto consiste en segmentar los miotubos píxel por píxel y comparar un método clásico de procesamiento de imágenes con una U-Net. La primera etapa es la auditoría del conjunto etiquetado `vt`.

El [informe final](docs/informe-final.md) resume el experimento. En 42 imágenes de prueba, la U-Net obtuvo Dice **0.371** y la línea clásica **0.055**. La [prueba de escala](docs/06-prueba-escala.md) procesó las **3,113 imágenes** de `subset_10GB` a **26.4 imágenes por segundo**.

## Acceso a los datasets

Los datos originales del curso están disponibles en [OneDrive institucional · Miotubos_2026](https://tecmx-my.sharepoint.com/shared?listurl=https%3A%2F%2Ftecmx%2Dmy%2Esharepoint%2Ecom%2Fpersonal%2Fgildardo%5Fsanchez%5Ftec%5Fmx1%2FDocuments&id=%2Fpersonal%2Fgildardo%5Fsanchez%5Ftec%5Fmx1%2FDocuments%2FMiotubos%5F2026&share=cgr%2DJdPhqM5oTJVDem32ysYjEgUCyPb9d42UTHNkbTutAy8ZUw). El acceso depende de los permisos de la carpeta y puede requerir iniciar sesión con la cuenta institucional.

| Contenido | Uso en el proyecto |
|---|---|
| `MyotubesAug_2026.pdf` | Enunciado, flujo de trabajo y entregables del reto. |
| `vt/` | Imágenes, anotaciones y máscaras para la auditoría, el entrenamiento y la evaluación. |
| `subset_10GB/` | Imágenes y manifiesto para procesar el conjunto a escala y revisar predicciones. |

1. Abre el enlace y descarga `vt` y, para reproducir la prueba de escala, `subset_10GB`.
2. Si se descargan como ZIP, descomprímelos. Conserva los nombres de los archivos para que coincidan con las listas de `particiones/piloto/`.
3. Para los scripts que usan `vt`, indica la carpeta que contiene directamente `images`, `masks_png`, `labels`, `verification` y `coco`. Para `procesar_subset.py`, indica la carpeta que contiene los PNG y `manifest.csv`.

OneDrive puede generar una carpeta adicional al descargar. En la ejecución original, las rutas fueron `Downloads/vt/vt` y `Downloads/subset_10GB/subset_10GB`. Ajusta los comandos a la ubicación donde hayas descomprimido tus datos. El código y las listas de partición se encuentran en este repositorio, y los datos se descargan desde la fuente institucional.

## Reproducir la auditoría

Usa Python 3.11 o posterior. Pasa la ruta de la carpeta `vt` que contiene directamente `images`, `masks_png`, `labels`, `verification` y `coco`.

```powershell
python -m pip install -r requirements.txt
python auditar.py "$env:USERPROFILE\Downloads\vt\vt" resultados/auditoria.json
```

La auditoría genera un JSON con los resultados por imagen y dos paneles de inspección. Los archivos de datos y los paneles que contienen imágenes originales se generan o conservan localmente.

## Hallazgos iniciales

| Medida | Resultado |
|---|---:|
| Parejas imagen–máscara | 306 |
| Resolución | 2592 × 1944 |
| Máscaras vacías | 34 |
| Píxeles etiquetados como miotubo | 1.76 % |
| Grupos de imágenes idénticas con máscaras distintas | 11 (23 archivos) |
| Etiquetas sin imagen correspondiente | 12 |
| Imágenes sin vista de verificación | 1 |

La coincidencia exacta se compara sobre los píxeles de la imagen decodificada. Se detectó una anotación rectangular que cubre el 72.79 % de su imagen. Conviene confirmar su validez y revisar las máscaras en conflicto antes de fijar la partición. Los nombres incluyen identificadores repetidos cuya relación con las placas, pozos y muestras todavía debe confirmarse.

## Partición provisional

Las listas para el piloto están en `particiones/piloto/`. Se generan con `python particionar.py resultados/auditoria.json particiones/piloto`. Consulta el [criterio de agrupación y sus límites](docs/02-particion-piloto.md) antes de usar estas listas.

La [línea base clásica](docs/03-linea-base.md) y la [U-Net](docs/04-unet-piloto.md) se ajustaron con la misma partición de validación. La [evaluación de prueba](docs/05-comparacion-prueba.md) compara ambas configuraciones fijas en las 42 imágenes reservadas. La revisión inicial de `subset_10GB` está al final del informe del paso 3.
