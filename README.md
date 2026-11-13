# Segmentación de miotubos en imágenes de microscopía

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
