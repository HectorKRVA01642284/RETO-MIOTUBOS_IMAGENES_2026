"""Ejecuta la U-Net fijada sobre subset_10GB y mide el procesamiento por etapas."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath
from time import perf_counter
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from PIL import Image, ImageDraw
import torch

from auditar import panel_font
from linea_base import FACTOR
from unet import UNet, probability_maps


def inventory(dataset):
    with (dataset / 'manifest.csv').open(encoding='utf-8-sig', newline='') as source:
        declared = [(PurePosixPath(row['destino']).name, int(row['bytes'])) for row in csv.DictReader(source)]
    paths = sorted(dataset.glob('*.png'))
    expected = sorted((path.name, path.stat().st_size) for path in paths)
    if not declared or sorted(declared) != expected:
        raise ValueError('Los nombres y tamaños de los PNG deben coincidir con manifest.csv.')
    return paths, sum(size for _, size in declared)


def read_image(path):
    with Image.open(path) as image:
        if image.mode != 'L' or image.size != (2592, 1944):
            raise ValueError(f'Formato inesperado en {path.name}.')
        return np.array(image.resize((image.width // FACTOR, image.height // FACTOR), Image.Resampling.BOX))


def sample_panel(dataset, archive, rows, output):
    positives = sorted((row for row in rows if row[1] > 0), key=lambda row: (row[1], row[0]))
    empty = next((row for row in rows if row[1] == 0), None)
    samples = [('Sin detección', empty)] if empty else []
    if positives:
        samples.extend((caption, positives[int(q * (len(positives) - 1))])
                       for caption, q in [('Mediana de las positivas', .5), ('Percentil 90 de las positivas', .9),
                                          ('Mayor cobertura', 1)])
    width, height, gap = 350, 263, 20
    panel = Image.new('RGB', (3 * width + 4 * gap, 121 + len(samples) * (height + 37)), '#0d1926')
    draw = ImageDraw.Draw(panel)
    draw.text((gap, 16), 'subset_10GB | Predicciones de la U-Net', fill='#f1f8ff', font=panel_font(31))
    draw.text((gap, 57), 'Muestra ilustrativa sin máscaras de referencia. Verde: región predicha.',
              fill='#bacfdf', font=panel_font(20))
    for col, title in enumerate(('Microscopía', 'Máscara predicha', 'Superposición')):
        draw.text((gap + col * (width + gap), 94), title, fill='#aaf0d5', font=panel_font(19))
    with ZipFile(archive) as masks:
        for i, (caption, (name, _, fraction)) in enumerate(samples):
            with Image.open(dataset / name) as original, Image.open(BytesIO(masks.read(f'mascaras/{name}'))) as pred:
                raw = original.resize((width, height), Image.Resampling.BOX).convert('RGB')
                mask = pred.convert('L').resize((width, height), Image.Resampling.BOX)
            overlay = Image.composite(Image.new('RGB', raw.size, '#38eea8'), raw, mask.point(lambda x: x * .65))
            y = 121 + i * (height + 37)
            for col, tile in enumerate((raw, mask, overlay)):
                panel.paste(tile, (gap + col * (width + gap), y))
            draw.text((gap, y + height + 5), f'{caption} · {name} · {fraction:.2%} marcado',
                      fill='#c1d4e2', font=panel_font(17))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def process(args):
    paths, total_bytes = inventory(args.dataset)
    checkpoint = torch.load(args.pesos, map_location='cpu', weights_only=True)
    threshold = np.float32(checkpoint['umbral'])
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['modelo'])
    warmup_start = perf_counter()
    dummy = np.zeros((1944 // FACTOR, 2592 // FACTOR), dtype='uint8')
    probability_maps(model, [dummy] * args.batch, checkpoint['media'], checkpoint['std'], device)
    warmup = perf_counter() - warmup_start
    args.mascaras.parent.mkdir(parents=True, exist_ok=True)
    args.resumen.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.mascaras.with_name(args.mascaras.name + '.part')
    rows = []
    stages = dict(lectura_y_reduccion=0.0, inferencia_y_transferencia=0.0, binarizacion_y_escritura=0.0)
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, ZipFile(temporary, 'w', ZIP_STORED) as archive:
        for offset in range(0, len(paths), 32):
            subset = paths[offset:offset + 32]
            tick = perf_counter()
            images = list(pool.map(read_image, subset))
            stages['lectura_y_reduccion'] += perf_counter() - tick
            for j in range(0, len(subset), args.batch):
                tick = perf_counter()
                maps = probability_maps(model, images[j:j + args.batch], checkpoint['media'], checkpoint['std'], device)
                stages['inferencia_y_transferencia'] += perf_counter() - tick
                tick = perf_counter()
                for path, prob in zip(subset[j:j + args.batch], maps):
                    prediction = prob >= threshold
                    marked = int(prediction.sum())
                    rows.append((path.name, marked, marked / prediction.size))
                    png = BytesIO()
                    Image.fromarray(prediction).save(png, format='PNG')
                    archive.writestr(f'mascaras/{path.name}', png.getvalue())
                stages['binarizacion_y_escritura'] += perf_counter() - tick
            if len(rows) % 256 == 0 or len(rows) == len(paths):
                print(f'{len(rows)}/{len(paths)} imágenes · {perf_counter() - start:.1f} s', flush=True)
        index = StringIO(newline='')
        writer = csv.writer(index)
        writer.writerow(('imagen', 'pixeles_positivos_648x486', 'fraccion_predicha'))
        writer.writerows(rows)
        archive.writestr('predicciones.csv', index.getvalue())
        archive.writestr('LEEME.txt', 'Máscaras binarias de 648 × 486, negro = fondo y blanco = miotubo predicho.\n'
                         'Cada píxel de la máscara corresponde a un bloque de 4 × 4 en el PNG de origen.\n'
                         'Para volver a 2592 × 1944, usar vecino más cercano.\n'
                         'Son predicciones automáticas. subset_10GB no incluye referencias para evaluarlas.\n')
    elapsed = perf_counter() - start
    temporary.replace(args.mascaras)
    coverage = np.array([row[2] for row in rows])
    summary = {'imagenes': len(rows), 'bytes_entrada': total_bytes, 'resolucion_original': [2592, 1944],
               'resolucion_mascara': [648, 486], 'lote': args.batch, 'hilos_lectura': args.workers,
               'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
               'pytorch': str(torch.__version__), 'epoca': checkpoint['epoca'], 'umbral': checkpoint['umbral'],
               'sha256_pesos': sha256(args.pesos.read_bytes()).hexdigest(), 'precision_inferencia': 'float32',
               'segundos_calentamiento_excluidos': warmup, 'segundos_totales': elapsed,
               'imagenes_por_segundo': len(rows) / elapsed, 'segundos_por_etapa': stages,
               'bytes_zip_predicciones': args.mascaras.stat().st_size,
               'imagenes_con_deteccion': int((coverage > 0).sum()), 'imagenes_sin_deteccion': int((coverage == 0).sum()),
               'fraccion_predicha_media': float(coverage.mean()), 'fraccion_predicha_mediana': float(np.median(coverage)),
               'fraccion_predicha_p90': float(np.quantile(coverage, .9)), 'fraccion_predicha_maxima': float(coverage.max())}
    args.resumen.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.panel:
        sample_panel(args.dataset, args.mascaras, rows, args.panel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('pesos', type=Path)
    parser.add_argument('mascaras', type=Path, help='ZIP con las máscaras predichas')
    parser.add_argument('resumen', type=Path)
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.batch <= 32 or args.workers < 1:
        parser.error('Se requiere un lote de 1 a 32 y al menos un hilo de lectura.')
    process(args)
