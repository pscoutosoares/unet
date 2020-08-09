# -*- coding: utf-8 -*-

from __future__ import print_function
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader import Tomographic_Dataset

from UNET import VGGNet, UNETmodel, UNETsota
from GOOGLENET import GOOGLENETmodel
from CoopNets import CoopNets
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import misc

import os

ssim_loss = False
crop      = True
weighted = False

projs = 4
net = "CoopNets"

if ssim_loss:
    net = net+"-SSIM-LOSS"
if weighted:
    net = net+"-MSE-WEIGHTED"
if crop:
    net = net+"-CROPPED"

batch_size = 20 #antes 10
epochs     = 100

momentum   = 0.5
w_decay    = 0 #antes 1e-5

#after each 'step_size' epochs, the 'lr' is reduced by 'gama'
lr         = 0.00001 # antes le-4 (VGG-UNET)
step_size  = 100
gamma      = 0.5

configs         = "{}-model-{}-projs".format(net,projs)

train_file      = "train3.csv"
val_file        = "validation3.csv"
input_dir       = "./resized_train_ld/"
target_dir      = "./output/"

validation_accuracy = np.zeros((epochs,1))

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, configs)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
print("GPU Available: ",use_gpu, " number: ",len(num_gpu))

train_data = Tomographic_Dataset(csv_file=train_file, phase='train', train_csv=train_file, input_dir=input_dir, target_dir=target_dir, crop=crop)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)


#directory of training files is passed to obtain the mean value of the images in the trained set which is not trained in the CNN
val_data = Tomographic_Dataset(csv_file=val_file, phase='val', flip_rate=0, train_csv=train_file, input_dir=input_dir, target_dir=target_dir, crop=crop)
val_loader = DataLoader(val_data, batch_size=2, num_workers=4)

if net.startswith('VGG-UNET'):
    print("VGG-UNET SELECTED!")
    vgg_model = VGGNet(pretrained=False, requires_grad=True, remove_fc=True)
    fcn_model = UNETmodel(pretrained_net=vgg_model)
elif net.startswith('GOOGLENET'):
    print("GOOGLENET SELECTED!")
    fcn_model = GOOGLENETmodel()
elif net.startswith('UNET-SOTA'):
    print("UNET-SOTA SELECTED!")
    fcn_model = UNETsota()
elif net.startswith('CoopNets'):
    print("CoopNets SELECTED!")
    fcn_model = CoopNets()
else:
    print("NO MODULE SELECTED!!")

if use_gpu:
    ts = time.time()
    if net.startswith('VGG-UNET'):
        vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

if ssim_loss:
    criterion = pytorch_ssim.SSIM()
else:
    criterion = nn.MSELoss()

optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

ssim_train = []
ssim_validation = []
psnr_train = []
psnr_validation = []

def train():
    hit = 0
    delta = 0.00001
    for epoch in range(epochs):
        scheduler.step()
        if epoch > 2 and abs(validation_accuracy[epoch-2]-validation_accuracy[epoch-1]) < delta:
            hit = hit + 1
        else:
            hit = 0
        if hit == 5:
            break

        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            if ssim_loss:
                loss = - criterion(outputs, labels)
            elif weighted:
                loss = mse_weighted.my_loss(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            output = outputs.data.cpu().numpy()
            N, _, h, w = output.shape
            pred = output.transpose(0, 2, 3, 1).reshape(-1, 1).reshape(N, h, w)

            target = batch['l'].cpu().numpy().reshape(N, h, w)
            psnr = []
            ssim = []
            for i in range(N):
                d1 = pred[i]
                d2 = target[i]
                psnr.append(compare_psnr(d1 - np.mean(d1), d2 - np.mean(d2)))
                ssim.append(compare_ssim(d1 - np.mean(d1), d2 - np.mean(d2)))
            psnr_train.append(np.mean(psnr))
            ssim_train.append(np.mean(ssim))

            if iter % 100 == 0:
                print("Train: epoch{}, iter{}, loss: {}, SSIM: {}, PSNR: {}".format(epoch, iter, loss.item(),np.mean(ssim), np.mean(psnr)))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

        val(epoch)

def val(epoch):
    fcn_model.eval()
    total_mse = []

    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, 1).reshape(N, h, w)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
        psnr = []
        ssim = []
        for i in range(N):
            d1 = pred[i]
            d2 = target[i]
            psnr.append(compare_psnr(d1 - np.mean(d1), d2 - np.mean(d2)))
            ssim.append(compare_ssim(d1 - np.mean(d1), d2 - np.mean(d2)))
        psnr_validation.append(np.mean(psnr))
        ssim_validation.append(np.mean(ssim))

        for p, t in zip(pred, target):
            total_mse.append(mse_acc(p, t))


    mse_accs = np.mean(total_mse)
    validation_accuracy[epoch] = mse_accs

    print("val: epoch{}, mse_acc: {}, ssim: {} ,  psnr: {}".format(epoch, mse_accs,  np.mean(ssim), np.mean(psnr) ))


def mse_acc(pred, target):

    return np.mean(np.square(pred-target))


if __name__ == "__main__":
    start = time.time()
    train()
    end = time.time()
    duration = end - start

    d = datetime(1, 1, 1) + timedelta(seconds=int(duration))
    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))
    print("mean SSIM-Train{}, Mean ssim-Validation: {}, Mean psnr Train: {}, Mean psnr Validation: {}".format(
        np.mean(ssim_train), np.mean(ssim_validation), np.mean(psnr_train), np.mean(psnr_validation)))
    np.save('validation_accuracy_{}-model-{}-projs.npy'.format(net,projs), validation_accuracy)
