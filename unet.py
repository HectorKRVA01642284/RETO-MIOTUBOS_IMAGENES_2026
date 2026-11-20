"""Entrena una U-Net binaria con las listas de entrenamiento y validación de vt."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
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


def build_model(architecture='unet', mean=0.0, std=1.0, pretrained=False):
    if architecture == 'ensemble_swin':
        from swin_unet import EnsembleSwin
        return EnsembleSwin(mean, std)
    if architecture == 'ensemble_unet_resnet18':
        return Ensemble(mean, std)
    if architecture == 'resnet18_unet':
        from resnet_unet import ResNetUNet
        return ResNetUNet(mean, std, pretrained)
    if architecture == 'unet':
        return UNet()
    raise ValueError(f'Arquitectura no publicada en este perfil: {architecture}')


class Ensemble(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        from resnet_unet import ResNetUNet
        self.members = nn.ModuleList((UNet(),ResNetUNet(mean,std)))

    def forward(self,x):
        probability = (self.members[0](x).float().sigmoid()+self.members[1](x).float().sigmoid())/2
        return torch.logit(probability.clamp(1e-6,1-1e-6))


class WeightedEnsemble(Ensemble):
    def __init__(self,mean,std):
        super().__init__(mean,std)
        self.register_buffer('alpha',torch.tensor(.5))

    def forward(self,x):
        probability=(1-self.alpha)*self.members[0](x).float().sigmoid()+self.alpha*self.members[1](x).float().sigmoid()
        return torch.logit(probability.clamp(1e-6,1-1e-6))


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


def load_pairs(root, names, training, factor=FACTOR, mask_root=None):
    def load(name):
        with Image.open(root / 'images' / name) as image, Image.open((mask_root if mask_root is not None else root / 'masks_png') / name) as mask:
            width, height = image.size
            if image.size != mask.size or any(size % factor for size in image.size):
                raise ValueError(f'Tamaño incompatible: {name}')
            reduced = (width // factor, height // factor)
            x = np.array(image.convert('L').resize(reduced, Image.Resampling.BOX))
            if training:
                y = (np.array(mask.resize(reduced, Image.Resampling.NEAREST)) != 0).astype(np.uint8)
                return x, y, np.flatnonzero(y)
            area = (np.asarray(mask) != 0).reshape(reduced[1], factor, reduced[0], factor)
            return x, area.sum(axis=(1, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(load, names))


def training_batches(pairs, rng, batch_size, rotations=False, patch=PATCH):
    order = rng.permutation(np.repeat(np.arange(len(pairs)), PATCHES_PER_IMAGE))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        inputs = np.empty((len(indices), 1, patch, patch), dtype=np.uint8)
        targets = np.empty_like(inputs)
        for i, index in enumerate(indices):
            image, mask, positive = pairs[index]
            height, width = image.shape
            top, left = rng.integers(height - patch + 1), rng.integers(width - patch + 1)
            if len(positive) and rng.random() < 0.5:
                y, x = divmod(int(rng.choice(positive)), width)
                top = np.clip(y - rng.integers(patch // 4, patch * 3 // 4), 0, height - patch)
                left = np.clip(x - rng.integers(patch // 4, patch * 3 // 4), 0, width - patch)
            window = np.s_[top:top + patch, left:left + patch]
            flip_y, flip_x = rng.choice((-1, 1), 2)
            inputs[i, 0] = image[window][::flip_y, ::flip_x]
            targets[i, 0] = mask[window][::flip_y, ::flip_x]
            if rotations:
                turns = int(rng.integers(4))
                inputs[i, 0] = np.rot90(inputs[i, 0], turns)
                targets[i, 0] = np.rot90(targets[i, 0], turns)
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


def threshold_counts(probability, foreground, factor=FACTOR, thresholds=THRESHOLDS):
    bins = np.searchsorted(thresholds, probability.ravel(), side='right')
    predictions = np.bincount(bins, minlength=len(thresholds) + 1) * factor ** 2
    overlaps = np.bincount(bins, weights=foreground.ravel(), minlength=len(thresholds) + 1).astype('int64')
    tp = overlaps[::-1].cumsum()[::-1][1:]
    fp = predictions[::-1].cumsum()[::-1][1:] - tp
    fn = int(foreground.sum()) - tp
    return np.stack((tp, fp, fn), axis=-1)


def evaluate(model, pairs, mean, std, device, batch_size, factor=FACTOR):
    per_image = []
    for start in range(0, len(pairs), batch_size):
        images, foregrounds = zip(*pairs[start:start + batch_size])
        probabilities = probability_maps(model, images, mean, std, device)
        per_image.extend(threshold_counts(p, f, factor) for p, f in zip(probabilities, foregrounds))
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
    train_pairs = load_pairs(args.dataset, names[0], training=True, factor=args.factor)
    val_pairs = load_pairs(args.dataset, names[1], training=False, factor=args.factor)
    mean = float(np.mean([image.mean() for image, _, _ in train_pairs]))
    second_moment = np.mean([np.square(image, dtype=np.float32).mean() for image, _, _ in train_pairs])
    std = float(np.sqrt(max(second_moment - mean ** 2, 1e-6)))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get('arquitectura','unet') != args.architecture:
            raise ValueError('La arquitectura no coincide con el checkpoint para afinación.')
        # Los pesos se afinan con la normalización usada para entrenarlos.
        mean, std = checkpoint['media'], checkpoint['std']
    model = build_model(args.architecture,mean,std,args.pretrained and not args.resume).to(device,memory_format=torch.channels_last)
    if args.resume:
        model.load_state_dict(checkpoint['modelo'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    positive_weight = torch.tensor(args.positive_weight, device=device)
    validation_batch = min(2 if args.factor == 2 else 4, args.batch)
    args.salida.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    report = {'estado': 'en_progreso', 'semilla': args.seed, 'epocas_solicitadas': args.epochs,
              'entrenamiento': len(train_pairs), 'validacion': len(val_pairs), 'factor_reduccion': args.factor,
              'media_entrenamiento': mean, 'std_entrenamiento': std, 'hardware': torch.cuda.get_device_name(device) if amp else 'CPU',
              'pytorch': str(torch.__version__), 'precision_mixta': amp, 'lote': args.batch,
              'parche': args.patch, 'parches_por_imagen': PATCHES_PER_IMAGE,
              'canales': WIDTHS if args.architecture == 'unet' else [64,64,128,256,512],
              'optimizador': 'AdamW', 'learning_rate': args.lr, 'peso_positivo_bce': args.positive_weight,
              'arquitectura': args.architecture, 'preentrenado_imagenet':args.pretrained,
              'rotaciones_90': args.rotations, 'scheduler': 'coseno' if args.resume or args.architecture != 'unet' else 'constante',
              'afinacion_desde': args.resume.name if args.resume else None,
              'sha256_origen': sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
              'parametros': sum(p.numel() for p in model.parameters()), 'historial': history}
    if args.resume:
        initial = evaluate(model, val_pairs, mean, std, device, validation_batch, args.factor)
        report['inicial'] = initial
        best_score = initial['metricas_micro']['dice'] or 0
        report['mejor'] = {'epoca': 0, **initial}
        torch.save({'modelo': model.state_dict(), 'media': mean, 'std': std,
                    'umbral': initial['umbral'], 'epoca': 0, 'factor_reduccion': args.factor,
                    'arquitectura':args.architecture}, args.salida / 'mejor.pt')
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        patch_count = 0
        for images, masks in training_batches(train_pairs, rng, args.batch, args.rotations, args.patch):
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
        validation = evaluate(model, val_pairs, mean, std, device, validation_batch, args.factor)
        record = {'epoca': epoch, 'perdida_entrenamiento': loss_value, **validation,
                  'segundos': round(perf_counter() - start, 2)}
        history.append(record)
        if args.resume or args.architecture != 'unet':
            scheduler.step()
        score = validation['metricas_micro']['dice'] or 0
        if score > best_score:
            best_score = score
            report['mejor'] = record
            torch.save({'modelo': model.state_dict(), 'media': mean, 'std': std,
                        'umbral': validation['umbral'], 'epoca': epoch, 'factor_reduccion': args.factor,
                        'arquitectura':args.architecture}, args.salida / 'mejor.pt')
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
    parser.add_argument('--resume', type=Path, help='Pesos para afinación con un optimizador nuevo')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--rotations', action='store_true', help='Añade rotaciones de 0, 90, 180 y 270 grados')
    parser.add_argument('--factor', type=int, choices=(2,4), default=FACTOR)
    parser.add_argument('--patch', type=int, choices=(192,256,384), default=PATCH)
    parser.add_argument('--architecture', choices=('unet','resnet18_unet'),default='unet')
    parser.add_argument('--pretrained',action='store_true')
    parser.add_argument('--positive-weight',type=float,default=POSITIVE_WEIGHT)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.lr <= 0 or args.positive_weight <= 0:
        parser.error('Se necesita al menos una época y un lote positivo.')
    train(args)
