"""Exporta el ZIP del piloto como máscaras binarias a resolución original."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_STORED, ZipFile

from PIL import Image


REDUCED_SIZE = (648, 486)
ORIGINAL_SIZE = (2592, 1944)
AREA_FACTOR = (ORIGINAL_SIZE[0] // REDUCED_SIZE[0]) * (ORIGINAL_SIZE[1] // REDUCED_SIZE[1])


def expand_png(name_and_png):
    name, png = name_and_png
    with Image.open(BytesIO(png)) as mask:
        if mask.format != 'PNG' or mask.mode != '1' or mask.size != REDUCED_SIZE:
            raise ValueError(f'Se esperaba una máscara binaria de 648 × 486: {name}')
        expanded = mask.resize(ORIGINAL_SIZE, Image.Resampling.NEAREST)
    output = BytesIO()
    expanded.save(output, format='PNG', compress_level=1)
    return name, output.getvalue()


def export(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError('Usa archivos distintos para la entrada y la salida.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix='.zip.part', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(source) as archive:
            entries = list(csv.DictReader(StringIO(archive.read('predicciones.csv').decode('utf-8'))))
            names = [f'mascaras/{row["imagen"]}' for row in entries]
            mask_members = sorted(n for n in archive.namelist() if n.startswith('mascaras/') and not n.endswith('/'))
            if not entries or len(set(names)) != len(names) or sorted(names) != mask_members:
                raise ValueError('El índice no coincide con las máscaras del ZIP de entrada.')

            with ZipFile(temporary, 'w', ZIP_STORED) as output, ThreadPoolExecutor(max_workers=6) as workers:
                for start in range(0, len(names), 32):
                    batch = [(name, archive.read(name)) for name in names[start:start + 32]]
                    for name, png in workers.map(expand_png, batch):
                        output.writestr(name, png)
                    processed = start + len(batch)
                    if processed % 512 == 0 or processed == len(names):
                        print(f'{processed}/{len(names)} máscaras exportadas', flush=True)

                index = StringIO(newline='')
                writer = csv.writer(index)
                writer.writerow(('imagen', 'pixeles_positivos_2592x1944', 'fraccion_predicha'))
                for row in entries:
                    marked = int(row['pixeles_positivos_648x486']) * AREA_FACTOR
                    writer.writerow((row['imagen'], marked, marked / (ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1])))
                output.writestr('predicciones.csv', index.getvalue())
                output.writestr('LEEME.txt',
                                'Máscaras binarias PNG de 2592 × 1944, del mismo tamaño que el original.\n'
                                'Negro = fondo (0). Blanco = miotubo predicho (1).\n'
                                'Se expandieron las predicciones de 648 × 486 con vecino más cercano.\n'
                                'Cada píxel predicho se replica en un bloque de 4 × 4.\n'
                                'La exportación no añade detalle ni cambia las detecciones.\n')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('entrada', type=Path, help='ZIP de máscaras generado por procesar_subset.py')
    parser.add_argument('salida', type=Path, help='ZIP nuevo con máscaras de 2592 × 1944')
    args = parser.parse_args()
    export(args.entrada, args.salida)
