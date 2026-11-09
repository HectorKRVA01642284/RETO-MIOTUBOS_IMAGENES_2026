"""Ajusta una línea base clásica usando exclusivamente la lista de validación."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


FACTOR = 4
CONFIGS = [(method, multiplier, clean) for method, multiplier, clean in product(
    ('global_oscuro', 'contraste_local'), (0.9, 1.0, 1.15, 1.35), (False, True),
)]


def responses(image):
    small = image.convert('L').resize((image.width // FACTOR, image.height // FACTOR),
                                     Image.Resampling.BOX)
    smooth = small.filter(ImageFilter.GaussianBlur(0.8))
    background = smooth.filter(ImageFilter.GaussianBlur(10))
    return {'global_oscuro': ImageOps.invert(smooth),
            'contraste_local': ImageChops.subtract(background, smooth)}


def otsu(response):
    hist = np.asarray(response.histogram(), dtype=np.float64)
    levels = np.arange(len(hist))
    weights = hist.cumsum()
    sums = (hist * levels).cumsum()
    valid = (weights > 0) & (weights < weights[-1])
    if not valid.any():
        return int(np.flatnonzero(hist)[0])
    scores = np.full(len(hist), -1.0)
    left, right = weights[valid], weights[-1] - weights[valid]
    mean_difference = sums[valid] / left - (sums[-1] - sums[valid]) / right
    scores[valid] = left * right * mean_difference ** 2
    return int(scores.argmax())


def segment(response, threshold, multiplier, clean):
    limit = round(threshold * multiplier)
    mask = response.point([255 if level > limit else 0 for level in range(256)])
    if clean:
        mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return mask


def pixel_counts(prediction, foreground_per_tile):
    marked = np.asarray(prediction) != 0
    tp = int(foreground_per_tile[marked].sum())
    fp = int(marked.sum()) * FACTOR ** 2 - tp
    fn = int(foreground_per_tile.sum()) - tp
    return tp, fp, fn


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(tp, fp, fn):
    return {'dice': ratio(2 * tp, 2 * tp + fp + fn),
            'iou': ratio(tp, tp + fp + fn),
            'precision': ratio(tp, tp + fp),
            'sensibilidad': ratio(tp, tp + fn)}


def validate(dataset, partitions):
    names = (partitions / 'validacion.txt').read_text(encoding='utf-8').splitlines()
    if not names or len(set(names)) != len(names):
        raise ValueError('La lista de validación debe contener nombres únicos.')
    results = {config: [] for config in CONFIGS}
    for name in names:
        with Image.open(dataset / 'images' / name) as image, \
                Image.open(dataset / 'masks_png' / name) as mask:
            if image.size != mask.size or any(size % FACTOR for size in image.size):
                raise ValueError(f'Tamaño incompatible con la cuadrícula de evaluación: {name}')
            fields = responses(image)
            # Una predicción reducida cubre un bloque de 4 × 4 píxeles al volver a tamaño original.
            foreground = (np.asarray(mask) != 0).reshape(mask.height // FACTOR, FACTOR,
                                                         mask.width // FACTOR, FACTOR).sum(axis=(1, 3))
        thresholds = {method: otsu(field) for method, field in fields.items()}
        for config in CONFIGS:
            method, multiplier, clean = config
            prediction = segment(fields[method], thresholds[method], multiplier, clean)
            results[config].append(pixel_counts(prediction, foreground))

    scores = []
    for (method, multiplier, clean), counts in results.items():
        tp, fp, fn = map(sum, zip(*counts))
        positive_dice = [metrics(t, f, n)['dice'] for t, f, n in counts if t + n > 0]
        scores.append({'metodo': method, 'multiplicador_otsu': multiplier, 'morfologia': clean,
                       'pixeles': {'tp': tp, 'fp': fp, 'fn': fn}, 'metricas_micro': metrics(tp, fp, fn),
                       'dice_promedio_en_positivas': float(np.mean(positive_dice)) if positive_dice else None,
                       'mascaras_vacias': sum(t + n == 0 for t, _, n in counts),
                       'mascaras_vacias_con_prediccion': sum(t + n == 0 and f > 0 for t, f, n in counts)})
    scores.sort(key=lambda item: item['metricas_micro']['dice'] or 0, reverse=True)
    return {'particion': 'validacion', 'imagenes': len(names), 'factor_reduccion': FACTOR,
            'criterio_seleccion': 'dice_micro', 'mejor': scores[0], 'configuraciones': scores}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('particiones', type=Path)
    parser.add_argument('salida', type=Path)
    args = parser.parse_args()
    report = validate(args.dataset, args.particiones)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8', newline='\n')
