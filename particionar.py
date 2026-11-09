"""Crea la partición provisional de vt a partir del JSON de auditar.py."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re


SEED = 2026
SPLITS = ('entrenamiento', 'validacion', 'prueba')
RATIOS = (0.70, 0.15, 0.15)
MIN_HELD_OUT_GROUPS = 5
UNKNOWN_GROUP = 'sin_uuid'
UUID_PATTERN = re.compile(
    r'[0-9a-f]{8}-([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})'
    r'_[a-z]\d+_\d+_\d+_Best\.png', re.IGNORECASE,
)
RECTANGLES = {
    '31b7e487-42d5b90b-a80c-4524-b17f-99018fff2a56_A3_02_03_Best.png',
    '07722ad5-42d5b90b-a80c-4524-b17f-99018fff2a56_A3_03_04_Best.png',
    '857fb71d-1cb78525-a7e5-4436-8262-f33beb93d74d_B1_02_06_Best.png',
}


def group_id(filename):
    match = UUID_PATTERN.fullmatch(filename)
    return match[1].lower() if match else UNKNOWN_GROUP


def choose_groups(groups):
    names = sorted(name for name in groups if name != UNKNOWN_GROUP)
    if len(names) < 2 * MIN_HELD_OUT_GROUPS:
        raise ValueError('Hay muy pocos grupos con UUID para validación y prueba.')
    sizes = {name: len(rows) for name, rows in groups.items()}
    empty = {name: sum(row['foreground'] == 0 for row in rows)
             for name, rows in groups.items()}
    target_size = [round(ratio * sum(sizes.values())) for ratio in RATIOS]
    target_empty = [round(ratio * sum(empty.values())) for ratio in RATIOS]
    rng = random.Random(SEED)
    best_score = None
    for _ in range(10000):
        rng.shuffle(names)
        validation_end = min(range(MIN_HELD_OUT_GROUPS, len(names) - MIN_HELD_OUT_GROUPS + 1),
                             key=lambda end: abs(sum(sizes[n] for n in names[:end]) - target_size[1]))
        test_end = min(range(validation_end + MIN_HELD_OUT_GROUPS, len(names) + 1),
                       key=lambda end: abs(sum(sizes[n] for n in names[validation_end:end]) - target_size[2]))
        validation = names[:validation_end]
        test = names[validation_end:test_end]
        train = names[test_end:] + ([UNKNOWN_GROUP] if UNKNOWN_GROUP in groups else [])
        candidate = (train, validation, test)
        if not train:
            continue
        size_error = sum(abs(sum(sizes[n] for n in split) - target)
                         for split, target in zip(candidate, target_size))
        empty_error = sum(abs(sum(empty[n] for n in split) - target)
                          for split, target in zip(candidate, target_empty))
        score = (size_error, empty_error, max(sizes[n] for n in validation + test))
        if best_score is None or score < best_score:
            best_score, best = score, candidate
    if best_score is None:
        raise ValueError('No se encontró una división con las tres particiones no vacías.')
    return best


def partition(audit, output):
    rows = {row['file']: row for row in audit['rows']}
    exclusions = {name: 'rectangulo_atipico' for name in RECTANGLES}
    for group in audit['duplicate_mask_comparison']:
        if group['max_mask_difference_fraction'] > 0:
            for name in group['files']:
                exclusions[name] = 'imagen_identica_con_mascaras_en_conflicto'
    if len(rows) != len(audit['rows']) or not exclusions.keys() <= rows.keys():
        raise ValueError('El inventario no coincide con las exclusiones registradas para vt.')

    groups = defaultdict(list)
    for name, row in rows.items():
        if name not in exclusions:
            groups[group_id(name)].append(row)

    assignment = {}
    manifests = {}
    for split, group_names in zip(SPLITS, choose_groups(groups)):
        for name in group_names:
            for row in groups[name]:
                assignment[row['file']] = split
        manifests[split] = sorted(row['file'] for name in group_names for row in groups[name])

    for group in audit['exact_duplicates']:
        splits = {assignment[stem + '.png'] for stem in group if stem + '.png' in assignment}
        if len(splits) > 1:
            raise ValueError('Se detectó un duplicado exacto en más de una partición.')

    output.mkdir(parents=True, exist_ok=True)
    summary = {'estado': 'provisional', 'semilla': SEED, 'pares_totales': len(rows),
               'exclusiones': dict(sorted(exclusions.items())), 'particiones': {}}
    for split, filenames in manifests.items():
        (output / f'{split}.txt').write_text('\n'.join(filenames) + '\n', encoding='utf-8', newline='\n')
        summary['particiones'][split] = {
            'imagenes': len(filenames),
            'mascaras_vacias': sum(rows[name]['foreground'] == 0 for name in filenames),
            'grupos': sorted({group_id(name) for name in filenames}),
        }
    (output / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
                                         encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('auditoria', type=Path, help='JSON generado por auditar.py')
    parser.add_argument('salida', type=Path, help='Carpeta para listas de archivos y resumen')
    args = parser.parse_args()
    partition(json.loads(args.auditoria.read_text(encoding='utf-8')), args.salida)
