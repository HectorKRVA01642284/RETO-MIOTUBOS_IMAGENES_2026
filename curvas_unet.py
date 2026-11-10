"""Dibuja la pérdida de entrenamiento y el Dice de validación de una corrida."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from auditar import panel_font


def chart(draw, records, box, title, read, color, baseline=None):
    left, top, right, bottom = box
    values = [read(record) for record in records]
    lower = 0 if baseline is not None else min(values) - 0.02
    upper = max(values + ([] if baseline is None else [baseline])) + 0.02
    draw.text((left, top - 43), title, fill='#d8e8f4', font=panel_font(25))
    for i in range(5):
        fraction = i / 4
        y = bottom - (bottom - top) * fraction
        draw.line((left, y, right, y), fill='#294154')
        draw.text((left - 10, y), f'{lower + (upper - lower) * fraction:.2f}',
                  anchor='rm', fill='#a9bfce', font=panel_font(16))
    if baseline is not None:
        y = bottom - (bottom - top) * (baseline - lower) / (upper - lower)
        draw.line((left, y, right, y), fill='#f5b975', width=2)
    points = []
    for i, record in enumerate(records):
        x = left + (right - left) * i / max(len(values) - 1, 1)
        y = bottom - (bottom - top) * (values[i] - lower) / (upper - lower)
        points.append((x, y))
        if len(records) <= 14 or i % 4 == 0 or i == len(records) - 1:
            draw.text((x, bottom + 11), str(record['epoca']), anchor='mt',
                      fill='#a9bfce', font=panel_font(15))
    if len(points) > 1:
        draw.line(points, fill=color, width=4)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    draw.text(((left + right) / 2, bottom + 44), 'Época', anchor='mt',
              fill='#a9bfce', font=panel_font(17))


def render(report, baseline, output):
    canvas = Image.new('RGB', (1120, 510), '#0d1926')
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 23), 'U-Net | Entrenamiento del piloto', fill='#f3f8ff', font=panel_font(34))
    draw.text((38, 70), 'La época y el umbral se eligieron en validación.',
              fill='#b8cbdc', font=panel_font(20))
    records = report['historial']
    chart(draw, records, (81, 166, 517, 366), 'Pérdida de entrenamiento',
          lambda row: row['perdida_entrenamiento'], '#76b9ff')
    chart(draw, records, (642, 166, 1074, 366), 'Dice micro de validación',
          lambda row: row['metricas_micro']['dice'] or 0, '#64e9b7', baseline)
    best = report['mejor']
    draw.text((38, 462), f'Mejor U-Net: {best["metricas_micro"]["dice"]:.3f} · época {best["epoca"]} · '
              f'Línea clásica (naranja): {baseline:.3f}', fill='#d8e8f4', font=panel_font(20))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unet', type=Path)
    parser.add_argument('linea_base', type=Path)
    parser.add_argument('salida', type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.linea_base.read_text(encoding='utf-8'))['mejor']['metricas_micro']['dice']
    render(json.loads(args.unet.read_text(encoding='utf-8')), baseline, args.salida)
