import os
import h5py
from math import  floor
import numpy as np
from os import listdir
from os.path import join, isdir, expanduser
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torchvision



'''
Description:
       Load Lytro Illum Images for Training
Output:
     - Input : 8 * 8  * 16
     - Target: 16 * 16  * 64
'''

def im2double(im):

    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return im.astype(np.float) / info.max  # Divide all values by the largest possible value in the datatype

    # return im.astype(np.float) * 256 / info.max  # Divide all values by the largest possible value in the datatype


def get_num_patches(height,width,patchSize,stride):

    numPatchesX = floor((width - patchSize) / stride) + 1
    numPatchesY = floor((height - patchSize) / stride) + 1
    numPatches = numPatchesY * numPatchesX
    return numPatches


def get_patches(input, patchSize, stride):
    [height, width, n_colour,angulars ] = input.shape

    numPatches = (floor((width - patchSize) / stride) + 1) * (floor((height - patchSize) / stride) + 1)
    patches = np.zeros((numPatches,patchSize, patchSize, 3, angulars))

    count = -1
    for iY in np.arange(0, height - patchSize + 1, stride):
        for iX in np.arange(0, width - patchSize + 1, stride):
            count = count + 1
            patches[count,:, :, :,:] = input[iY: iY + patchSize, iX: iX + patchSize, :,:]
    return patches



def __creat_h5_file_train__():

    height =256   #SF=374,HCI=521,EPFL=434
    width = 448   #SF=540,HCI=521.EPFL=624
    N = 7
    M = 4
    patchSize = 32
    stride =32
    upsscale_factor = 4
    n_colour = 3

    dataFolder = '/home/vim/fcr_dataset/vimeo_septuplet/RGB_sequences_4_2_mat_train/'
    outputFolder = '/home/vim/fcr_dataset/vimeo_septuplet/'

    pathDir = os.listdir(dataFolder)
    numImg = 20
    #numImg = len(pathDir)  #训练文件夹的数据总数目
    # name = pathDir[2]

   # print(length)
    numPatches = get_num_patches(height,width,patchSize,stride)  #一张图片分成的pathes数量
    numTotalPatches = numPatches * numImg #总共的pathces数量
    # numImg = 1


    fileName = outputFolder + 'subset_train_vimeo_4_2_RGB.h5'
    # fileName = outputFolder + 'test_input_16.h5'

    i = 0

    for ns in range(0,numImg):
        i = i+1
        if i<=numImg:

          if ns ==0:
            file = h5py.File(fileName, 'w')
            Target_dset = file.create_dataset("Target", (numPatches, patchSize, patchSize,n_colour,N),
                                              maxshape=(None, patchSize, patchSize,n_colour,N),
                                              dtype=np.float32)
            Input_dset = file.create_dataset("Input", (numPatches, patchSize // upsscale_factor, patchSize // upsscale_factor,n_colour,M),
                                             maxshape=(None, patchSize // upsscale_factor, patchSize // upsscale_factor,n_colour,M),
                                              dtype=np.float32)
          else:
            file = h5py.File(fileName, 'a')
    #
          Target_dset.resize([ns * numPatches + numPatches, patchSize, patchSize, n_colour,N])  # 先变大，再写入，一次写入numPatches个图片，内容写入后，写一次数组又变大
          Input_dset.resize([ns * numPatches + numPatches, patchSize // upsscale_factor, patchSize // upsscale_factor,n_colour,M])
    #

        # print('********************************')
          print('Working on the "%s" dataset (%d of %d)' % (pathDir[ns][0:- 4], ns + 1, numImg), flush=True)
    #
        # print('Loading input light field ...', end=' ')

          child = os.path.join(dataFolder,pathDir[ns])
        # child = "G:/EPI_SR_EH/Dataset/test_mat/Backlight_1.mat"
          data = loadmat(child)
          LF_Y = data['Y_new']
          LF_input = data['input_new']
        # LF_G = data['G']
   #
          pTaImgs = get_patches(LF_Y, patchSize, stride)  # 得到四维numpy nump * H * W *  N
          pInImgs = get_patches(LF_input, patchSize//upsscale_factor, stride//upsscale_factor)  # 得到四维numpy nump * H * W * c * N

        # plt.figure()
        # plt.subplot(211)  # 第一张图中的第1张子图s
        # plt.imshow(pTaImgs[1,:,:,32],cmap='gray')
        # plt.subplot(212)  # 第一张图中的第1张子图
        # plt.imshow(pInImgs[1,:,:,32],cmap='gray')
        # plt.show()

          Target_dset[ns * numPatches : ns * numPatches + numPatches] = pTaImgs
          Input_dset[ns* numPatches : ns * numPatches + numPatches] = pInImgs

          print('Done')
          print('**********************************')


    print('Train data processed')
    file.close() #close file





if __name__ == '__main__':
    __creat_h5_file_train__()



    #
    # f_YUV = h5py.File("/home/vim/fcr_dataset/vimeo_septuplet/train_vimeo_4_2_RGB.h5", 'r')  # 打开h5文件
    # f_YUV.keys()  # 可以查看所有的主键
    # T = f_YUV['Target']
    # I = f_YUV['Input']
    # input = I[50 ]  # 取出主键为data的所有的键值
    # target = T[50 ]  # 取出主键为data的所有的键值
    # f_YUV.close()
    # # # #
    # #
    # in_img = input[:,:,:,2]
    # tar_img = target[:,:,:,3]
    #
    # plt.figure()
    # plt.subplot(211)  # 第一张图中的第1张子图
    # plt.imshow(in_img)
    # plt.subplot(212)  # 第一张图中的第1张子图
    # plt.imshow(tar_img)
    # plt.show()
