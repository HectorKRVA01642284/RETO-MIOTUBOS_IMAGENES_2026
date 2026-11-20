"""Exporta máscaras con el umbral y la geometría de un checkpoint, sin interacción."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from inferencia_teselas import predict_tiled
from unet import build_model, probability_maps


def run(args):
    if args.imagenes.resolve() == args.salida.resolve():
        raise ValueError('La salida debe ser distinta de la carpeta de imágenes originales.')
    names = sorted(args.imagenes.glob('*.png'))
    if args.limite is not None:
        if args.limite < 1:
            raise ValueError('El límite debe ser positivo.')
        names = names[:args.limite]
    if not names:
        raise ValueError('No se encontraron imágenes PNG.')
    torch.set_num_threads(6)
    torch.backends.cudnn.deterministic = True
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    factor = cp.get('factor_reduccion', 4)
    if cp.get('inferencia_tta') and cp.get('inferencia_teselas'):
        raise ValueError('No se admite combinar TTA y ventanas nativas en este exportador.')
    if cp.get('inferencia_teselas') and factor != 1:
        raise ValueError('Las ventanas nativas requieren factor_reduccion=1.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cp.get('arquitectura', 'unet'), cp['media'], cp['std'])
    model.load_state_dict(cp['modelo'])
    model.to(device, memory_format=torch.channels_last).eval()
    args.salida.mkdir(parents=True, exist_ok=True)
    records = []
    start = perf_counter()
    for path in names:
        with Image.open(path) as image:
            width, height = image.size
            if width % factor or height % factor:
                raise ValueError(f'Tamaño incompatible con factor {factor}: {path.name}')
            raw = np.array(image.convert('L').resize((width//factor, height//factor), Image.Resampling.BOX))
        if cp.get('inferencia_teselas'):
            prob = predict_tiled(model, raw, cp['media'], cp['std'], device, **cp['inferencia_teselas'])
        elif cp.get('inferencia_tta'):
            from inferencia_tta import predict_tta
            prob = predict_tta(model, [raw], cp['media'], cp['std'], device,
                               cp['inferencia_tta'].get('preservar_umbral'))[0]
        else:
            prob = probability_maps(model, [raw], cp['media'], cp['std'], device)[0]
        mask = Image.fromarray((prob >= np.float32(cp['umbral'])).astype('uint8') * 255)
        if factor != 1:
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        mask.save(args.salida/path.name)
        records.append({'imagen': path.name, 'ancho': width, 'alto': height,
                        'pixeles_predichos': int(np.count_nonzero(np.asarray(mask)))})
    report = {'checkpoint_sha256': sha256(args.checkpoint.read_bytes()).hexdigest(),
              'arquitectura': cp.get('arquitectura', 'unet'), 'factor_reduccion': factor,
              'inferencia_teselas': cp.get('inferencia_teselas'), 'umbral': cp['umbral'],
              'inferencia_tta': cp.get('inferencia_tta'),
              'segundos': perf_counter()-start, 'dispositivo': str(device), 'imagenes': records,
              'nota': 'Predicciones automáticas; no son anotaciones verificadas.'}
    (args.salida/'predicciones.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(f'{len(records)} máscaras guardadas en {args.salida}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('imagenes', type=Path)
    parser.add_argument('salida', type=Path)
    parser.add_argument('--limite', type=int)
    run(parser.parse_args())
