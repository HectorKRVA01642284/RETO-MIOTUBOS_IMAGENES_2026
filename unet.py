"""Entrena una U-Net binaria con las listas de entrenamiento y validación de vt."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from linea_base import FACTOR, metrics


THRESHOLDS = np.arange(5, 100, 5, dtype=np.float32) / 100
PATCH = 192
PATCHES_PER_IMAGE = 4
WIDTHS = (12, 24, 48, 96)
LEARNING_RATE = 0.001
POSITIVE_WEIGHT = 4.0


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(conv_block(a, b) for a, b in zip((1,) + WIDTHS, WIDTHS))
        self.up = nn.ModuleList(nn.ConvTranspose2d(a, b, 2, stride=2)
                                for a, b in zip(WIDTHS[:0:-1], WIDTHS[-2::-1]))
        self.decoder = nn.ModuleList(conv_block(2 * width, width) for width in WIDTHS[-2::-1])
        self.output = nn.Conv2d(WIDTHS[0], 1, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        x = F.pad(x, (0, -width % 8, 0, -height % 8), mode='replicate')
        skips = []
        for block in self.encoder[:-1]:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.encoder[-1](x)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            x = block(torch.cat((up(x), skip), dim=1))
        return self.output(x)[..., :height, :width]


def load_pairs(root, names, training):
    def load(name):
        with Image.open(root / 'images' / name) as image, Image.open(root / 'masks_png' / name) as mask:
            width, height = image.size
            if image.size != mask.size or any(size % FACTOR for size in image.size):
                raise ValueError(f'Tamaño incompatible: {name}')
            reduced = (width // FACTOR, height // FACTOR)
            x = np.array(image.convert('L').resize(reduced, Image.Resampling.BOX))
            if training:
                y = (np.array(mask.resize(reduced, Image.Resampling.NEAREST)) != 0).astype(np.uint8)
                return x, y, np.flatnonzero(y)
            area = (np.asarray(mask) != 0).reshape(reduced[1], FACTOR, reduced[0], FACTOR)
            return x, area.sum(axis=(1, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(load, names))


def training_batches(pairs, rng, batch_size):
    order = rng.permutation(np.repeat(np.arange(len(pairs)), PATCHES_PER_IMAGE))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        inputs = np.empty((len(indices), 1, PATCH, PATCH), dtype=np.uint8)
        targets = np.empty_like(inputs)
        for i, index in enumerate(indices):
            image, mask, positive = pairs[index]
            height, width = image.shape
            top, left = rng.integers(height - PATCH + 1), rng.integers(width - PATCH + 1)
            if len(positive) and rng.random() < 0.5:
                y, x = divmod(int(rng.choice(positive)), width)
                top = np.clip(y - rng.integers(PATCH // 4, PATCH * 3 // 4), 0, height - PATCH)
                left = np.clip(x - rng.integers(PATCH // 4, PATCH * 3 // 4), 0, width - PATCH)
            window = np.s_[top:top + PATCH, left:left + PATCH]
            flip_y, flip_x = rng.choice((-1, 1), 2)
            inputs[i, 0] = image[window][::flip_y, ::flip_x]
            targets[i, 0] = mask[window][::flip_y, ::flip_x]
        yield inputs, targets


def input_tensor(images, mean, std, device):
    x = torch.as_tensor(images, device=device, dtype=torch.float32)
    return x.sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)


def training_loss(logits, target, pos_weight):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probabilities = logits.sigmoid()
    overlap = (2 * (probabilities * target).sum() + 1) / (probabilities.sum() + target.sum() + 1)
    return 0.5 * bce + 0.5 * (1 - overlap)


@torch.inference_mode()
def probability_maps(model, images, mean, std, device):
    model.eval()
    x = input_tensor(np.stack(images)[:, None], mean, std, device)
    return model(x).float().sigmoid()[:, 0].cpu().numpy()


def threshold_counts(probability, foreground):
    bins = np.searchsorted(THRESHOLDS, probability.ravel(), side='right')
    predictions = np.bincount(bins, minlength=len(THRESHOLDS) + 1) * FACTOR ** 2
    overlaps = np.bincount(bins, weights=foreground.ravel(), minlength=len(THRESHOLDS) + 1).astype('int64')
    tp = overlaps[::-1].cumsum()[::-1][1:]
    fp = predictions[::-1].cumsum()[::-1][1:] - tp
    fn = int(foreground.sum()) - tp
    return np.stack((tp, fp, fn), axis=-1)


def evaluate(model, pairs, mean, std, device, batch_size):
    per_image = []
    for start in range(0, len(pairs), batch_size):
        images, foregrounds = zip(*pairs[start:start + batch_size])
        probabilities = probability_maps(model, images, mean, std, device)
        per_image.extend(threshold_counts(p, f) for p, f in zip(probabilities, foregrounds))
    counts = np.stack(per_image)
    totals = counts.sum(axis=0)
    scores = [metrics(*(int(value) for value in row)) for row in totals]
    best = max(range(len(scores)), key=lambda i: scores[i]['dice'] or 0)
    tp, fp, fn = (int(value) for value in totals[best])
    positives = [metrics(int(t), int(f), int(n))['dice'] for t, f, n in counts[:, best] if t + n > 0]
    return {'umbral': round(float(THRESHOLDS[best]), 2), 'metricas_micro': scores[best],
            'pixeles': {'tp': tp, 'fp': fp, 'fn': fn},
            'dice_promedio_en_positivas': float(np.mean(positives)) if positives else None,
            'mascaras_vacias': int(((counts[:, best, 0] + counts[:, best, 2]) == 0).sum()),
            'mascaras_vacias_con_prediccion': sum(int(t + n == 0 and f > 0) for t, f, n in counts[:, best])}


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    amp = device.type == 'cuda'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    names = [(args.particiones / f'{split}.txt').read_text(encoding='utf-8').splitlines()
             for split in ('entrenamiento', 'validacion')]
    if any(not group or len(set(group)) != len(group) for group in names) or set(names[0]) & set(names[1]):
        raise ValueError('Se necesitan listas no vacías, disjuntas y sin nombres repetidos.')
    print(f'Cargando {len(names[0])} imágenes de entrenamiento y {len(names[1])} de validación en {device}.', flush=True)
    train_pairs = load_pairs(args.dataset, names[0], training=True)
    val_pairs = load_pairs(args.dataset, names[1], training=False)
    mean = float(np.mean([image.mean() for image, _, _ in train_pairs]))
    second_moment = np.mean([np.square(image, dtype=np.float32).mean() for image, _, _ in train_pairs])
    std = float(np.sqrt(max(second_moment - mean ** 2, 1e-6)))
    model = UNet().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    positive_weight = torch.tensor(POSITIVE_WEIGHT, device=device)
    args.salida.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    report = {'estado': 'en_progreso', 'semilla': args.seed, 'epocas_solicitadas': args.epochs,
              'entrenamiento': len(train_pairs), 'validacion': len(val_pairs), 'factor_reduccion': FACTOR,
              'media_entrenamiento': mean, 'std_entrenamiento': std, 'hardware': torch.cuda.get_device_name(device) if amp else 'CPU',
              'pytorch': str(torch.__version__), 'precision_mixta': amp, 'lote': args.batch,
              'parche': PATCH, 'parches_por_imagen': PATCHES_PER_IMAGE, 'canales': WIDTHS,
              'optimizador': 'AdamW', 'learning_rate': LEARNING_RATE, 'peso_positivo_bce': POSITIVE_WEIGHT,
              'parametros': sum(p.numel() for p in model.parameters()), 'historial': history}
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        patch_count = 0
        for images, masks in training_batches(train_pairs, rng, args.batch):
            inputs = input_tensor(images, mean, std, device)
            target = torch.as_tensor(masks, device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(inputs)
            loss = training_loss(logits, target, positive_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach() * len(images)
            patch_count += len(images)
        loss_value = loss_sum.item() / patch_count
        if not np.isfinite(loss_value):
            raise FloatingPointError('La pérdida de entrenamiento no es finita.')
        validation = evaluate(model, val_pairs, mean, std, device, min(4, args.batch))
        record = {'epoca': epoch, 'perdida_entrenamiento': loss_value, **validation,
                  'segundos': round(perf_counter() - start, 2)}
        history.append(record)
        score = validation['metricas_micro']['dice'] or 0
        if score > best_score:
            best_score = score
            report['mejor'] = record
            torch.save({'modelo': model.state_dict(), 'media': mean, 'std': std,
                        'umbral': validation['umbral'], 'epoca': epoch}, args.salida / 'mejor.pt')
        if epoch == args.epochs:
            report['estado'] = 'completo'
        (args.salida / 'validacion.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Época {epoch}/{args.epochs} · pérdida {loss_value:.4f} · Dice validación {score:.4f} · '
              f'umbral {validation["umbral"]:.2f} · {record["segundos"]:.1f} s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('particiones', type=Path)
    parser.add_argument('salida', type=Path)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error('Se necesita al menos una época y un lote positivo.')
    train(args)
