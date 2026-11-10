# Segmentación de miotubos en imágenes de microscopía

El reto consiste en segmentar los miotubos píxel por píxel y comparar un método clásico de procesamiento de imágenes con una U-Net. La primera etapa es la auditoría del conjunto etiquetado `vt`.

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

La [línea base clásica](docs/03-linea-base.md) y la [U-Net](docs/04-unet-piloto.md) se ajustaron con la misma partición de validación. El siguiente experimento es comparar ambas configuraciones en la partición reservada de prueba. La revisión inicial de `subset_10GB` está al final del informe del paso 3.
