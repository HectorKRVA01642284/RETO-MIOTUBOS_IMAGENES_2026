"""Promedia cuatro vistas alineadas; permite conservar todas las detecciones base."""

import numpy as np
import torch

from unet import probability_maps


@torch.inference_mode()
def predict_tta(model, images, mean, std, device, preserve_threshold=None):
    maps = []
    for flip_y, flip_x in ((False, False), (False, True), (True, False), (True, True)):
        transformed = [im[::-1 if flip_y else 1, ::-1 if flip_x else 1] for im in images]
        probability = probability_maps(model, transformed, mean, std, device)
        if flip_y: probability = probability[:, ::-1, :]
        if flip_x: probability = probability[:, :, ::-1]
        maps.append(probability.copy())
    average = np.mean(np.stack(maps), axis=0)
    if preserve_threshold is not None:
        if not 0 < preserve_threshold < 1: raise ValueError('Umbral base inválido.')
        average = np.where(maps[0] >= np.float32(preserve_threshold), 1., average)
    return average
