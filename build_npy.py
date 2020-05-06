# title                 :build_npy.py
# description           :It converts label images into .npy format
# author                :Dr. Luis Filipe Alves Pereira (luis.filipe@ufrpe.br or luisfilipeap@gmail.com)
# date                  :2019-05-16
# version               :1.0
# notes                 :Please let me know if you find any problem in this code
# python_version        :3.6
# numpy_version         :1.16.3
# scipy_version         :1.2.1
# matplotlib_version    :3.0.3
# pilow_version         :6.0.0
# pandas_version        :0.24.2
# pytorch_version       :1.1.0
# ==============================================================================


import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

"""
Parameters for converting output images into .npy files


low_quality_dir:    directory of the low quality images
high_quality_dir:   directory of the high quality images (the files must have the same names that those in the directory of low quality images)
target_dir:         directory where the .npy files will be saved
files_ext:          extension of images files at low_quality_dir and high_quality_dir
debug:              flag to allow intermediate visualization
residual_learning:  flag to activate the residual learning scheme discussed in the literature

"""


low_quality_dir = './resized_train_ld/'
high_quality_dir = './resized_train_gt/'
target_dir = "./output/"

files_ext = '.png'

debug = False
residual_learning = False

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

for file in os.listdir(high_quality_dir):

        if file.endswith(files_ext):

            input_img = misc.imread(os.path.join(low_quality_dir,file), mode='F')

            output_img = misc.imread(os.path.join(high_quality_dir,file), mode='F')
            #output_img = np.transpose(output_img)

            if residual_learning:
                target = ((input_img-output_img)/255)
                target = (target - np.amin(target))/(np.amax(target) - np.amin(target))
                #target = output_img/255 - 0.5
            else:
                target = output_img/255
                target = (target - np.amin(target)) / (np.amax(target) - np.amin(target))

            if debug:
                print('min: {} max: {}'.format(np.min(target), np.max(target)))
                plt.figure()
                plt.imshow(target, cmap='gray', vmin=np.min(target), vmax=np.max(target))
                plt.show()
                break
            else:
                np.save(target_dir+file[0:len(file)-3]+'npy', target)
                print(target_dir+file+ " Done!")



