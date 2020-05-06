import torch
import time
import os
from torch.autograd import Variable
import scipy.misc
import scipy
from data_loader import Tomographic_Dataset
from torch.utils.data import Dataset, DataLoader
from data_utils import data_mean_value
import numpy as np
import ntpath
from matplotlib import pyplot as plt
from torchvision import utils
#from skimage.morphology import disk
#from skimage.filters.rank import median

from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from math import log10
net             = 'UNET-article'
#net             = 'ZERO-ROT-VGG-UNET'
projs           =  4
input_dir       = "./resized_train_ld/"
target_dir      = "./output/"
means           = data_mean_value("test4.csv", input_dir) / 255.

model_src = "./models/UNET-Article-CROPPED-model-4-projs"


def mse_acc(pred, target):
    return np.mean(np.square(pred - target))



def evaluate_img():

    test_data = Tomographic_Dataset(csv_file="test4.csv", phase='val', flip_rate=0, train_csv="train3.csv",
                                    input_dir=input_dir, target_dir=target_dir)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=1)

    fcn_model = torch.load(model_src)
    n_tests = len(test_data.data)

    print("{} files for testing....".format(n_tests))

    folder = "./results-{}-{}-projs/".format(net, projs)
    if not os.path.exists(folder):
        os.makedirs(folder)

    execution_time = np.zeros((n_tests, 1))
    count = 0
    psnr = []
    ssim = []
    ssim_img = []
    psnr_img = []

    for iter, batch in enumerate(test_loader):

        name = batch['file'][0]
        dest = os.path.join(folder, name[0:len(name)-3])
        if not os.path.exists(dest):
            os.mkdir(dest)

        #print(batch['X'].shape)
        #type(batch['X'])
        input = Variable(batch['X'].cuda())
        input = input.unsqueeze(1)
        print(input.shape)
        start = time.time()
        outputs = fcn_model(input)
        end = time.time()
        elapsed = end-start
        execution_time[count] = elapsed
        #print('execution: {} seconds'.format(elapsed))
        print(elapsed)
        count = count + 1

        output = outputs.data.cpu().numpy()

        N, _, h, w = output.shape
        y = output[0, 0, :, :]
        pred = output.transpose(0, 2, 3, 1).reshape(-1, 1).reshape(N, h, w)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
        d1 = pred[0]
        d2 = target[0]
        psnr.append(compare_psnr(d1 - np.mean(d1), d2 - np.mean(d2)))
        ssim.append(compare_ssim(d1 - np.mean(d1), d2 - np.mean(d2)))

        img_batch = batch['X']
        img_batch.add_(means[0])

        grid = utils.make_grid(img_batch)
        x = grid.numpy()[::-1].transpose((1, 2, 0))
        final_rec = x[:, :, 0] - y

        #final_rec = y+0.5

        original = scipy.misc.imread(batch['o'][0], flatten=True)

        ssim_img.append(compare_ssim(original - np.mean(original), final_rec - np.mean(final_rec)))
        psnr_img.append(compare_psnr(original - np.mean(original), final_rec - np.mean(final_rec), data_range=255))

        #final_rec = np.transpose(final_rec)

        scipy.misc.imsave(dest+'/target-residual.png', target[0,:,:])
        scipy.misc.imsave(dest+'/residual.png', y)
        scipy.misc.imsave(dest+'/final_rec.png', final_rec)
        scipy.misc.imsave(dest+'/input.png', x)
        scipy.misc.imsave(dest+'/original.png', original)

        # print("mean: {}".format(np.mean(execution_time[1:n_tests])))
        #print("executed {} of {}\n".format(iter,len(test_loader)))

    print("ssim_filter: {},  psnr_filter: {},   ssim-img: {}, psnr-img: {}".format(np.mean(ssim), np.mean(psnr), np.mean(ssim_img), np.mean(psnr_img)))

    #print("mean: {}".format(np.mean(execution_time[1:n_tests])))
    #print("std: {}".format(np.std(execution_time[1:n_tests])))



if __name__ == "__main__":
    print('a')
    evaluate_img()

