#!/usr/bin/python
# -- coding: utf-8 --

from os.path import join
from Dataset.dataset import LoadH5,LoadImg,LoadImgtest
import torch.utils.data as data

def get_h5_set(train_set):  # 实参为'Train\91.h5'
    '''
    Load H5 dataset.
    :param train_set: the filename of the dataset
    :return: the loaded data
    '''
    train_dir = join("./Dataset/", train_set)
    # test_dir = join("./Dataset/test/", train_set)
    # 1、返回通过指定字符连接序列中元素后生成的新字符串。
    # 2、此处的join为os.path.join。将两个路径拼接在一起


    # return LoadH5(test_dir)
    return LoadH5(train_dir)  # 这个函数定义在dataset中
    # 此步骤返回一系列图片数据给train_set


def get_img_set(test_set,args):  # 实参为'Test/set5.mat'
    '''
    Load images file data.
    :param train_set: the folder name of the images in
    :return: the loaded data
    '''
    # test_dir = join("./Dataset/", test_set)

    return LoadImg(test_set,args)

def get_img_testset(test_set,args):  # 实参为'Test/set5.mat'
    '''
    Load images file data.
    :param train_set: the folder name of the images in
    :return: the loaded data
    '''
    # test_dir = join("./Dataset/", test_set)

    return LoadImgtest(test_set,args)

# def get_img_train(test_set):  # 实参为'Test/set5.mat'
#     '''
#     Load images file data.
#     :param train_set: the folder name of the images in
#     :return: the loaded data
#     '''
#     # test_dir = join("./Dataset/", test_set)
#
#     return LoadImg(test_set)

class H5Dataset(data.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        # print(index)
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]