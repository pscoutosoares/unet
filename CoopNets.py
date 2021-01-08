import torch
import cv2
import torch.nn as nn
from GOOGLENET import GOOGLENETmodel
from UNET import VGGNet, UNETsota
class CoopNets(nn.Module):
    def __init__(self):
        super().__init__()
        self.netA = UNETsota()
        self.netB = GOOGLENETmodel()
    def forward(self,x,locale, save=False):
        x = x.unsqueeze(1)
        temp_x = x.cpu().numpy()       
        fx = self.netA(x)
        gx = self.netB(x+fx)
        
        if(save):
            N, _, h, w = temp_x.shape
            img_x = x.cpu().numpy().reshape(N, h, w)[0]
            cv2.imwrite(locale +"_in.png", img_x * 255)  

            img_fx = fx.cpu().detach().numpy().reshape(N, h, w)[0]
            cv2.imwrite(locale+"_fx.png", img_fx * 255)  

            img_gx = gx.cpu().detach().numpy().reshape(N, h, w)[0]
            cv2.imwrite(locale+"_gx.png", img_gx * 255)  

        return gx