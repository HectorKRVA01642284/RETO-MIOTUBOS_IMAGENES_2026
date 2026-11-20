"""Codificador Swin-T de TorchVision y decodificador propio con GroupNorm.

Los pesos generales del codificador son ImageNet. Las capas de detalle y
segmentación se entrenan en vt. No replica una publicación llamada Swin-Unet.
"""
import torch
from torch import nn
from torch.nn import functional as F


def block(a,b):
    groups=8 if b%8==0 else 4
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(groups,b),nn.GELU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(groups,b),nn.GELU())


class SwinUNet(nn.Module):
    def __init__(self,mean,std,pretrained=False):
        super().__init__()
        from torchvision.models import swin_t,Swin_T_Weights
        self.encoder=swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None).features
        self.decoder=nn.ModuleList((block(768+384,128),block(128+192,64),block(64+96,32)))
        self.raw=block(1,12);self.output=nn.Sequential(block(32+12,24),nn.Conv2d(24,1,1))
        nn.init.constant_(self.output[-1].bias,-2.)
        self.register_buffer('gray_mean',torch.tensor(float(mean)));self.register_buffer('gray_std',torch.tensor(float(std)))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,x):
        h,w=x.shape[-2:];x=F.pad(x,(0,-w%32,0,-h%32),mode='replicate')
        rgb=(x*self.gray_std+self.gray_mean)/255;rgb=(rgb.repeat(1,3,1,1)-self.rgb_mean)/self.rgb_std
        skips=[]
        for i,layer in enumerate(self.encoder):
            rgb=layer(rgb)
            if i in (1,3,5,7):skips.append(rgb.permute(0,3,1,2).contiguous())
        decoded=skips[-1]
        for layer,skip in zip(self.decoder,reversed(skips[:-1])):
            decoded=F.interpolate(decoded,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            decoded=layer(torch.cat((decoded,skip),dim=1))
        decoded=F.interpolate(decoded,size=x.shape[-2:],mode='bilinear',align_corners=False)
        result=self.output(torch.cat((decoded,self.raw(x)),dim=1))
        return result[...,:h,:w]


class EnsembleSwin(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        from unet import Ensemble
        self.base=Ensemble(mean,std);self.candidate=SwinUNet(mean,std)
        self.register_buffer('alpha',torch.tensor(.5))

    def forward(self,x):
        probability=(1-self.alpha)*self.base(x).float().sigmoid()+self.alpha*self.candidate(x).float().sigmoid()
        return torch.logit(probability.clamp(1e-6,1-1e-6))
