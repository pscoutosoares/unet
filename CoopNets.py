import torch
import torch.nn as nn
from GOOGLENET import GOOGLENETmodel
from UNET import VGGNet, UNETsota
class CoopNets(nn.Module):
    def __init__(self):
        super().__init__()
        self.netA = UNETsota()
        self.netB = GOOGLENETmodel()
    def forward(self,x):
        x = x.unsqueeze(1)
        fx = self.netA(x)
        gx = self.netB(x-fx)
        return gx