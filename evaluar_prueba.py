"""Compara los dos métodos ya fijados sobre las imágenes reservadas de prueba."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from auditar import panel_font
from linea_base import metrics, otsu, pixel_counts, responses, segment
from particionar import group_id
from unet import UNet, load_pairs, probability_maps


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def aggregate(rows, method):
    fields = ('tp', 'fp', 'fn')
    totals = [sum(row[method]['pixeles'][key] for row in rows) for key in fields]
    positives = [row[method]['metricas']['dice'] for row in rows if row['pixeles_referencia'] > 0]
    empty = [row for row in rows if row['pixeles_referencia'] == 0]
    return {'imagenes': len(rows), 'pixeles': dict(zip(fields, totals)),
            'metricas_micro': metrics(*totals),
            'dice_promedio_en_positivas': float(np.mean(positives)) if positives else None,
            'mascaras_vacias': len(empty),
            'mascaras_vacias_con_prediccion': sum(row[method]['pixeles']['fp'] > 0 for row in empty)}


def example_panels(dataset, examples, output, threshold):
    width, height, gap = 243, 182, 15
    panel = Image.new('RGB', (5 * width + 6 * gap, 118 + len(examples) * (height + 37)), '#0d1926')
    draw = ImageDraw.Draw(panel)
    draw.text((gap, 15), 'Prueba reservada | Línea clásica y U-Net', fill='#f1f8ff', font=panel_font(32))
    draw.text((gap, 55), f'Casos ilustrativos · Probabilidad U-Net: negro = 0, amarillo = 1 · Umbral {threshold:.2f}',
              fill='#bacfdf', font=panel_font(18))
    titles = ('Microscopía', 'Máscara de referencia', 'Línea clásica', 'Probabilidad U-Net', 'Máscara U-Net')
    for col, title in enumerate(titles):
        draw.text((gap + col * (width + gap), 90), title, fill='#aaf0d5', font=panel_font(18))
    palette = ((0, 0, 4), (87, 16, 110), (188, 54, 84), (249, 142, 9), (252, 255, 164))
    for i, (caption, row, baseline, probability) in enumerate(examples):
        name = row['imagen']
        with Image.open(dataset / 'images' / name) as raw, Image.open(dataset / 'masks_png' / name) as mask:
            original = raw.resize((width, height), Image.Resampling.BOX)
            reference = mask.point([0] + [255] * 255).resize((width, height), Image.Resampling.BOX)
        prob_small = np.asarray(Image.fromarray(probability).resize((width, height), Image.Resampling.BOX))
        heatmap = Image.fromarray(np.stack([np.interp(prob_small, np.linspace(0, 1, 5), channel)
                                             for channel in zip(*palette)], axis=-1).astype('uint8'))
        predicted = Image.fromarray((probability >= threshold).astype('uint8') * 255)
        tiles = (original, reference, baseline.resize((width, height), Image.Resampling.BOX),
                 heatmap, predicted.resize((width, height), Image.Resampling.BOX))
        y = 118 + i * (height + 37)
        for col, tile in enumerate(tiles):
            panel.paste(tile.convert('RGB'), (gap + col * (width + gap), y))
        dice = row['unet']['metricas']['dice']
        score = f'{dice:.3f}' if dice is not None else 'no definido'
        draw.text((gap, y + height + 6), f'{caption} · {name} · Dice U-Net {score}',
                  fill='#c1d4e2', font=panel_font(15))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def run(args):
    names = (args.particiones / 'prueba.txt').read_text(encoding='utf-8').splitlines()
    if not names or len(names) != len(set(names)):
        raise ValueError('La lista de prueba debe contener nombres únicos.')
    groups = {group_id(name) for name in names}
    for split in ('entrenamiento', 'validacion'):
        earlier = (args.particiones / f'{split}.txt').read_text(encoding='utf-8').splitlines()
        if groups & {group_id(name) for name in earlier}:
            raise ValueError(f'Hay grupos compartidos entre prueba y {split}.')

    base = read_json(args.clasica)['mejor']
    validation = read_json(args.unet_validacion)
    checkpoint = torch.load(args.pesos, map_location='cpu', weights_only=True)
    if (validation['estado'] != 'completo' or checkpoint['epoca'] != validation['mejor']['epoca']
            or checkpoint['umbral'] != validation['mejor']['umbral']
            or checkpoint['media'] != validation['media_entrenamiento']
            or checkpoint['std'] != validation['std_entrenamiento']):
        raise ValueError('Los pesos deben corresponder a la configuración fijada en validación.')

    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['modelo'])
    threshold = np.float32(checkpoint['umbral'])
    pairs = load_pairs(args.dataset, names, training=False)
    rows, examples = [], []
    by_group = defaultdict(list)
    for start in range(0, len(pairs), 4):
        batch = pairs[start:start + 4]
        images, foregrounds = zip(*batch)
        probabilities = probability_maps(model, images, checkpoint['media'], checkpoint['std'], device)
        for name, foreground, prob in zip(names[start:start + 4], foregrounds, probabilities):
            # La línea clásica reduce la imagen internamente, por eso recibe el PNG original.
            with Image.open(args.dataset / 'images' / name) as raw:
                field = responses(raw)[base['metodo']]
            classical = segment(field, otsu(field), base['multiplicador_otsu'], base['morfologia'])
            predictions = {'clasica': classical, 'unet': prob >= threshold}
            row = {'imagen': name, 'grupo': group_id(name), 'pixeles_referencia': int(foreground.sum())}
            for method, prediction in predictions.items():
                counts = pixel_counts(prediction, foreground)
                row[method] = {'pixeles': dict(zip(('tp', 'fp', 'fn'), counts)), 'metricas': metrics(*counts)}
            rows.append(row)
            by_group[row['grupo']].append(row)
            if args.panel:
                examples.append((row, classical, prob))

    methods = ('clasica', 'unet')
    report = {'particion': 'prueba', 'imagenes': len(rows), 'grupos': len(groups),
              'configuracion_clasica': {key: base[key] for key in ('metodo', 'multiplicador_otsu', 'morfologia')},
              'configuracion_unet': {'epoca': checkpoint['epoca'], 'umbral': checkpoint['umbral'],
                                    'sha256_pesos': sha256(args.pesos.read_bytes()).hexdigest()},
              'resultados': {method: aggregate(rows, method) for method in methods},
              'por_grupo': {group: {method: aggregate(group_rows, method) for method in methods}
                            for group, group_rows in sorted(by_group.items())},
              'por_imagen': rows}
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.panel:
        positives = sorted((ex for ex in examples if ex[0]['pixeles_referencia']),
                           key=lambda ex: ex[0]['unet']['metricas']['dice'])
        if len(positives) >= 3:
            chosen = [('Mayor Dice', *positives[-1]), ('Dice intermedio', *positives[len(positives) // 2]),
                      ('Menor Dice', *positives[0])]
        else:
            chosen = [('Máscara positiva', *ex) for ex in positives]
        empties = [ex for ex in examples if not ex[0]['pixeles_referencia']]
        if empties:
            chosen.append(('Máscara vacía', *max(empties, key=lambda ex: ex[0]['unet']['pixeles']['fp'])))
        example_panels(args.dataset, chosen, args.panel, threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('particiones', type=Path)
    parser.add_argument('clasica', type=Path, help='JSON con la configuración clásica elegida')
    parser.add_argument('unet_validacion', type=Path)
    parser.add_argument('pesos', type=Path)
    parser.add_argument('salida', type=Path)
    parser.add_argument('--panel', type=Path)
    run(parser.parse_args())
