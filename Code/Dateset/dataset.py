#!/usr/bin/python
# -- coding: utf-8 --
import h5py
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor, CenterCrop, Resize
# import matplotlib.pyplot as plt
import torch
from os import listdir
from os.path import join
import random
# import cv2
import scipy.ndimage
import scipy.misc
import numpy as np
from math import floor
from PIL import Image
from PIL.Image import NEAREST, BILINEAR, BICUBIC, LANCZOS, BOX, HAMMING
from scipy.io import loadmat


def is_image_file(filename):
    '''
    Check wheather the file is a image file.
    :param filename: name of the file
    :return: bool value shows that whether it is a image
    '''
    # return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])
    return any(filename.endswith(extension) for extension in [".png"])

def load_img(filepath):
    '''
    Load the image and get the luminance data.
    :param filepath: path of the image.
    :return: luminance data
    '''

    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class LoadH5(data.Dataset):  # 实参为训练集数据的路径。'Train\91.h5'的数据。image_h5=train_dir=Train\91.h5
    def __init__(self, image_h5):
        super(LoadH5, self).__init__()

        self.to_tensor = ToTensor()
        # pytorch在读入图片的时候需要做transform变换，其中transform一般都需要ToTensor()操作，相当于将转换为tensor
        # 对于一个图片img，调用ToTensor转化成张量的形式
        # https://blog.csdn.net/qq_37385726/article/details/81811466

        # 定义imput和label
        self.input_patch = []
        self.target_patch = []

        # f_YUV = h5py.File (image_h5, 'r')  # 打开h5文件
        # self.input_patch = f_YUV['Input']
        # self.target_patch = f_YUV['Target']
        with h5py.File(image_h5 ,'r') as hf:  # 读取h5的数据的方式。（不太理解原理，但大概应该是91.h5里面已经包含了data和label的数据，data为LR，label为HR，同时打包在91.h5中）
            self.input_patch = np.array(hf.get('Input'))
            self.target_patch = np.array(hf.get('Target'))

    def __getitem__(self, index):
        input_image = self.input_patch[index]
        target_image = self.target_patch[index]
        input_data = torch.from_numpy(input_image)
        # input_data = input_data.permute(2,3,0,1)
        target_data = torch.from_numpy(target_image)
        # target_data = target_data.permute(2,3,0,1)

        return input_data, target_data

    def __len__(self):
        return len(self.input_patch)

class LoadImg(data.Dataset):
    def __init__(self, image_dir,args):
        super(LoadImg, self).__init__()

        f = open(image_dir)
        self.image_filenames = f.readlines()  # 读取全部内容 ，并以列表方式返回

        # self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.to_tensor = ToTensor()
        self.fig = args.fig
        self.use_augmentation = False
        self.height = 256
        self.width =448
        self.num =7
        self.patch_size = 64

    def __getitem__(self, index):


        # data = loadmat(self.image_filenames[index])
        Imgname = self.image_filenames[index]
        path = join("/home/vim/fcr_dataset/vimeo_septuplet/sequences/", Imgname[:-1])
        # path = join("./videoset/", Imgname[:-1])
        target = torch.zeros(self.patch_size,self.patch_size,3,self.num)
        input = torch.zeros(self.patch_size//4,self.patch_size//4,3,4)
        crop_x = random.randint(0, self.width - self.patch_size)
        crop_y = random.randint(0, self.height - self.patch_size)
        box = (crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size)
        n = 0
        # filenames = [join(path, x) for x in listdir(path) if is_image_file(x)]
        for i in range(self.num):
            filename = path+"/im"+str(i+1)+".png"
            OriginalImg = Image.open(filename)
            ta = OriginalImg.crop(box)
            ta_n = np.array(ta)
            target[:,:,:,i]=torch.from_numpy(ta_n)


            if i%2==0:
                inp = ta.resize((self.patch_size//4,self.patch_size//4),resample=BICUBIC)
                inp_n = np.array(inp)
                input[:,:,:,n]=torch.from_numpy(inp_n)
                n=n+1

        return input/255,target/255


            # target = OriginalImg[crop_y: crop_y + self.patch_size, crop_x: crop_x + self.patch_size, :, :]
        # OriginalImg = data['Y_new']
        # inputImg = data['input_new']
        # height = inputImg.shape[0]
        # width = inputImg.shape[1]

        # target_image = torch.from_numpy(OriginalImg).float()


        # if self.fig == 'train':
        #     height = OriginalImg.shape[0]
        #     width = OriginalImg.shape[1]
        #     # if self.use_augmentation:
        #     #  # randomly rescale image
        #     #      if random.random() <= 0.5:
        #     #         scale = random.choice([0.9, 0.8, 0.7, 0.6])
        #     #         input_image = input_image.resize((int(input_image.width * scale), int(input_image.height * scale)), resample=Image.BICUBIC)
        #     #
        #     #     # randomly rotate image
        #     #      if random.random() <= 0.5:
        #     #         input_image = input_image.rotate(random.choice([90, 180, 270]), expand=True)
        # # randomly crop patch from training set
        #     crop_x = random.randint(0, width - self.patch_size)
        #     crop_y = random.randint(0, height - self.patch_size)
        #     # ta_crop_x = crop_x/
        #     target = OriginalImg[  crop_y : crop_y + self.patch_size, crop_x : crop_x + self.patch_size, :, :]
        #     input = target[::4,::4,:,::2]
        #     # input = inputImg[  crop_y : crop_y + self.patch_size, crop_x : crop_x + self.patch_size, :, :]
        #     target = torch.from_numpy(target).float()
        #     input = torch.from_numpy(input).float()
        #
        #     return input, target
        # if self.fig == 'test':
        #     target = torch.from_numpy(OriginalImg).float()
        #     input = torch.from_numpy(inputImg).float()
        #     return input, target,Imgname




    def __len__(self):
        return len(self.image_filenames)

class LoadImgtest(data.Dataset):
    def __init__(self, image_dir,args):
        super(LoadImgtest, self).__init__()

        f = open(image_dir)
        self.image_filenames = f.readlines()  # 读取全部内容 ，并以列表方式返回

        # self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.to_tensor = ToTensor()
        self.fig = args.fig
        self.use_augmentation = False
        self.height = 256
        self.width =448
        self.num =7
        self.patch_size = 64

    def __getitem__(self, index):


        # data = loadmat(self.image_filenames[index])
        Imgname = self.image_filenames[index]
        path = join("./videoset/", Imgname[:-1])
        target = torch.zeros(self.height,self.width ,3,self.num)
        input = torch.zeros(self.height//4,self.width//4,3,4)
        # crop_x = random.randint(0, self.width - self.patch_size)
        # crop_y = random.randint(0, self.height - self.patch_size)
        # box = (crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size)
        n = 0
        # filenames = [join(path, x) for x in listdir(path) if is_image_file(x)]
        for i in range(self.num):
            filename = path+"/im"+str(i+1)+".png"
            OriginalImg = Image.open(filename)
            # ta = OriginalImg.crop(box)
            ta_n = np.array(OriginalImg)
            target[:,:,:,i]=torch.from_numpy(ta_n)


            if i%2==0:
                inp = OriginalImg.resize((self.width//4,self.height//4),resample=BICUBIC)
                inp_n = np.array(inp)
                input[:,:,:,n]=torch.from_numpy(inp_n)
                n=n+1

        return input/255,target/255,Imgname


            # target = OriginalImg[crop_y: crop_y + self.patch_size, crop_x: crop_x + self.patch_size, :, :]
        # OriginalImg = data['Y_new']
        # inputImg = data['input_new']
        # height = inputImg.shape[0]
        # width = inputImg.shape[1]

        # target_image = torch.from_numpy(OriginalImg).float()


        # if self.fig == 'train':
        #     height = OriginalImg.shape[0]
        #     width = OriginalImg.shape[1]
        #     # if self.use_augmentation:
        #     #  # randomly rescale image
        #     #      if random.random() <= 0.5:
        #     #         scale = random.choice([0.9, 0.8, 0.7, 0.6])
        #     #         input_image = input_image.resize((int(input_image.width * scale), int(input_image.height * scale)), resample=Image.BICUBIC)
        #     #
        #     #     # randomly rotate image
        #     #      if random.random() <= 0.5:
        #     #         input_image = input_image.rotate(random.choice([90, 180, 270]), expand=True)
        # # randomly crop patch from training set
        #     crop_x = random.randint(0, width - self.patch_size)
        #     crop_y = random.randint(0, height - self.patch_size)
        #     # ta_crop_x = crop_x/
        #     target = OriginalImg[  crop_y : crop_y + self.patch_size, crop_x : crop_x + self.patch_size, :, :]
        #     input = target[::4,::4,:,::2]
        #     # input = inputImg[  crop_y : crop_y + self.patch_size, crop_x : crop_x + self.patch_size, :, :]
        #     target = torch.from_numpy(target).float()
        #     input = torch.from_numpy(input).float()
        #
        #     return input, target
        # if self.fig == 'test':
        #     target = torch.from_numpy(OriginalImg).float()
        #     input = torch.from_numpy(inputImg).float()
        #     return input, target,Imgname




    def __len__(self):
        return len(self.image_filenames)

# if __name__ == '__main__':
#
#     # PATH = './Dataset/test\\general_10_eslf.png'
#     img = read_illum_images_test('./Dataset/test\\general_10_eslf.png',test_scale=2)