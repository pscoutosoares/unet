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
        ux = self.netA(x)
        #pred = self.netB(ux-x)+x
        pred = self.netB(ux)

        return pred
