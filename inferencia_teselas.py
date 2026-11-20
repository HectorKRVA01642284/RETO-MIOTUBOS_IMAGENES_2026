"""Predicción a resolución nativa mediante ventanas solapadas y mezcla suave."""

import numpy as np
import torch

from unet import probability_maps


def starts(length, tile, stride):
    last=max(0,length-tile)
    return sorted(set(list(range(0,last+1,stride))+[last]))


@torch.inference_mode()
def predict_tiled(model, image, mean, std, device, tile=512, stride=384, batch=2):
    if not 0<stride<=tile or batch<1:
        raise ValueError('Se requiere un paso positivo menor o igual que la ventana y un lote positivo.')
    height,width=image.shape
    padded=np.pad(image,((0,max(0,tile-height)),(0,max(0,tile-width))),mode='edge')
    positions=[(top,left) for top in starts(padded.shape[0],tile,stride)
               for left in starts(padded.shape[1],tile,stride)]
    total=np.zeros(padded.shape,dtype=np.float32)
    weights=np.zeros_like(total)
    window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype('float32')
    for offset in range(0,len(positions),batch):
        subset=positions[offset:offset+batch]
        crops=[padded[top:top+tile,left:left+tile] for top,left in subset]
        maps=probability_maps(model,crops,mean,std,device)
        for (top,left),prob in zip(subset,maps):
            total[top:top+tile,left:left+tile]+=prob*window
            weights[top:top+tile,left:left+tile]+=window
    if not np.all(weights>0):
        raise ValueError('La cuadrícula de ventanas dejó píxeles sin cubrir.')
    return (total/weights)[:height,:width]
