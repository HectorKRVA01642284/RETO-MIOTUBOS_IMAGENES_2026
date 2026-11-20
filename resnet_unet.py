"""Codificador ResNet-18 de TorchVision y decodificador de segmentación propio."""

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
                         nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class ResNetUNet(nn.Module):
    def __init__(self, mean, std, pretrained=False, depth=18):
        super().__init__()
        from torchvision.models import resnet18,resnet34,ResNet18_Weights,ResNet34_Weights
        if depth==18:encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif depth==34:encoder=resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        else:raise ValueError('Codificador no soportado')
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layers = nn.ModuleList((encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4))
        self.decoder = nn.ModuleList((block(512+256,128), block(128+128,64),
                                      block(64+64,32), block(32+64,24)))
        self.output = nn.Sequential(block(24,16), nn.Conv2d(16,1,1))
        self.register_buffer('gray_mean', torch.tensor(float(mean)))
        self.register_buffer('gray_std', torch.tensor(float(std)))
        self.register_buffer('rgb_mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_std', torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self, x):
        height,width = x.shape[-2:]
        x = (x*self.gray_std+self.gray_mean)/255
        x = (x.repeat(1,3,1,1)-self.rgb_mean)/self.rgb_std
        x = F.pad(x,(0,-width%32,0,-height%32),mode='replicate')
        stem = self.stem(x)
        skips = [stem]
        x = self.pool(stem)
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        for decoder,skip in zip(self.decoder,reversed(skips[:-1])):
            x = F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x = decoder(torch.cat((x,skip),dim=1))
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        return self.output(x)[...,:height,:width]
