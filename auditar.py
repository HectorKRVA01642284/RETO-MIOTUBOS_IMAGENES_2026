from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def panel(root, out, names):
    width, height = 352, 264
    pad = 24
    row = height + 42
    canvas = Image.new('RGB', (3 * width + 4 * pad, 124 + row * len(names)), '#0d1926')
    draw = ImageDraw.Draw(canvas)
    title, body, small = [panel_font(size) for size in (34, 20, 16)]
    draw.text((pad, 19), 'Reto de miotubos | Muestra de vt', '#f1f8ff', font=title)
    draw.text((pad, 64), 'Anotaciones de referencia · El verde marca los píxeles etiquetados como miotubo.', '#a9c5d7', font=body)
    for j, label in enumerate(('Imagen de microscopía', 'Máscara binaria', 'Máscara sobre la imagen')):
        draw.text((pad + j * (width + pad), 98), label, '#aaf0d5', font=body)
    for i, (caption, name) in enumerate(names):
        with Image.open(root / 'images' / name) as image, Image.open(root / 'masks_png' / name) as mask:
            pixels = np.asarray(mask)
            coverage = np.count_nonzero(pixels) / pixels.size
            original = image.resize((width, height), Image.Resampling.LANCZOS).convert('RGB')
            binary = Image.fromarray((pixels != 0).astype('uint8') * 255)
            thumbnail = binary.resize((width, height), Image.Resampling.BOX)
            overlay = Image.composite(Image.new('RGB', original.size, '#36e9a4'), original, thumbnail.point(lambda x: x * 0.58))
        y = 124 + i * row
        for j, tile in enumerate((original, thumbnail, overlay)):
            canvas.paste(tile, (pad + j * (width + pad), y))
        draw.text((pad, y + height + 5), f'{caption} · {name} · {coverage:.2%} de cobertura', '#c1d4e2', font=small)
    canvas.save(out)


def panel_font(size):
    for name in ('arial.ttf', 'DejaVuSans.ttf', 'LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def audit(root, out):
    images = {p.stem: p for p in (root / 'images').glob('*.png')}
    masks = {p.stem: p for p in (root / 'masks_png').glob('*.png')}
    labels = {p.stem: p for p in (root / 'labels').glob('*.txt')}
    checks = {p.stem for p in (root / 'verification').glob('*.png')}
    coco = json.loads((root / 'coco' / 'annotations.json').read_text(encoding='utf-8-sig'))
    counts = Counter(item['image_id'] for item in coco['annotations'])
    coco_by_name = {Path(item['file_name']).stem: item['id'] for item in coco['images']}
    fingerprints = defaultdict(list)
    rows = []
    for name in sorted(images.keys() & masks.keys()):
        with Image.open(images[name]) as image, Image.open(masks[name]) as mask:
            pixels = np.asarray(image)
            labels_mask = np.asarray(mask)
            values = np.bincount(labels_mask.ravel(), minlength=256)
            fingerprint = hashlib.sha256(pixels.tobytes()).hexdigest()
            fingerprints[fingerprint].append(name)
            instances = counts[coco_by_name[name]] if name in coco_by_name else None
            rows.append(dict(file=images[name].name, width=image.width, height=image.height,
                             mode=image.mode, intensity_min=int(pixels.min()), intensity_max=int(pixels.max()),
                             mask_values=np.flatnonzero(values).tolist(),
                             foreground=float(values[1:].sum() / values.sum()),
                             size_match=image.size == mask.size, coco_instances=instances,
                             verification=name in checks, label=name in labels,
                             pattern='cell' if re.search(r'-cell\d+$', name, re.I) else
                                     'Best' if re.search(r'_[A-Z]\d+_\d+_\d+_Best$', name) else 'otro'))
    report = dict(rows=rows, images_without_mask=sorted(images.keys() - masks.keys()),
                  masks_without_image=sorted(masks.keys() - images.keys()),
                  extra_labels=sorted(labels.keys() - images.keys()),
                  no_verification=sorted(images.keys() - checks),
                  exact_duplicates=[names for names in fingerprints.values() if len(names) > 1],
                  coco_fields=list(coco), coco_image_fields=list(coco['images'][0]))
    duplicate_conflicts = []
    strongest = None
    for group in report['exact_duplicates']:
        group_masks = []
        for name in group:
            with Image.open(masks[name]) as mask:
                group_masks.append(np.array(mask))
        difference, first, second = max(
            (float(np.mean(left != right)), group[i], group[j])
            for i, left in enumerate(group_masks)
            for j, right in enumerate(group_masks) if i < j
        )
        duplicate_conflicts.append(dict(files=[images[name].name for name in group],
                                        max_mask_difference_fraction=difference))
        if strongest is None or difference > strongest[0]:
            strongest = difference, first, second
    report['duplicate_mask_comparison'] = duplicate_conflicts
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if strongest is not None and strongest[0] > 0:
        _, first, second = strongest
        outlier = max(rows, key=lambda row: row['foreground'])['file']
        panel(root, out.parent / 'casos_para_revisar.png', [
            ('Duplicado A', images[first].name), ('Duplicado B', images[second].name),
            ('Mayor cobertura', outlier),
        ])


def examples_panel(root, out):
    rows = json.loads(out.read_text(encoding='utf-8'))['rows']
    sorted_rows = sorted(rows, key=lambda row: (row['foreground'], row['file']))
    positives = [row for row in sorted_rows if row['foreground'] > 0]
    examples = [(caption, positives[int(q * (len(positives) - 1))]['file'])
                for caption, q in [('Cobertura baja', .1), ('Cobertura intermedia', .5), ('Cobertura alta', .9)]]
    panel(root, out.parent / 'muestra_miotubos.png', examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('salida', type=Path)
    parser.add_argument('--solo-panel', action='store_true')
    args = parser.parse_args()
    if not args.solo_panel:
        audit(args.dataset, args.salida)
    examples_panel(args.dataset, args.salida)
