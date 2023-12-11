import os

import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import skimage.io
import random


class ODISource():
    def __init__(self):
        self.train_mos = []
        self.test_mos = []
        self.train_names = []
        self.test_names = []
        self.__feed_set('CVIQ')
        self.train_mos = np.concatenate(self.train_mos)
        self.test_mos = np.concatenate(self.test_mos)

    def __train_validate(self, dmos, files):
        assert len(dmos) == len(files)
        num_videos = len(files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        test_index = index[1::6]
        train_index = np.delete(index, test_index)
        np.random.shuffle(train_index)
        return train_index, test_index

    def __get_dmos(self, indicator):
        mat = sio.loadmat('./datasets/ODI_enhance.mat')
        mos = mat[indicator]
        mos = np.array(mos).flatten()
        name = mat['index_' + indicator]
        name = np.array(name).flatten()
        return mos, name

    def __gen_file_name(self, indicator: str, name):
        namebox = []
        for n in name:
            namebox.append(indicator + '/%d.png' % (n))
        return namebox

    def __feed_set(self, indicator: str):
        mos, files = self.__get_dmos(indicator)
        train_index, test_index = self.__train_validate(mos, files)
        self.train_mos.append(mos[train_index])
        self.test_mos.append(mos[test_index])
        self.train_names = self.train_names + self.__gen_file_name(indicator, files[train_index])
        self.test_names = self.test_names + self.__gen_file_name(indicator, files[test_index])


class ODIDataPro(Dataset):  # 继承Dataset
    def __init__(self, root_dir, images, dmos):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.images = images  # 目录里的所有文件
        self.dmos = dmos
        conv_op = torch.nn.Conv2d(3, 3, kernel_size=3, padding=0, bias=False)
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3  # 将sobel算子转换为适配卷积操作的卷积核
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
        conv_op.weight.data = torch.from_numpy(sobel_kernel)
        self.conv_op = conv_op.half().to('cuda')

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def SI(self, v):
        v = torch.from_numpy(v).half().to('cuda')
        si = self.conv_op(v)
        si = torch.std(si.view(1, 3, -1), dim=-1)
        si = torch.mean(si)
        return si

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = skimage.io.imread(img_path)
        img = skimage.transform.resize(img, (1920, 3840))
        img = (img * 255).astype(np.uint8)

        img = np.transpose(img, [2, 0, 1])
        img=img[np.newaxis,:,:,:]

        v1= img[:, :,0:384, :]
        v2 = img[:, :,384:384 * 2, :]
        v3 = img[:, :,384 * 2:384 * 3, :]
        v4 = img[:, :,384 * 3:384 * 4, :]
        v5 = img[:, :,384 * 4:384 * 5, :]

        label = self.dmos[index]
        label = np.ones((3, 1)) * label

        s1 = self.SI(v1)
        s2 = self.SI(v2)

        s4 = self.SI(v4)
        s5 = self.SI(v5)

        if s1 > s5:
            v = v1
        else:
            v = v5
        if s2 > s4:
            vv = v2
        else:
            vv = v4

        sample = {'v1': v, 'v2': vv, 'v3': v3, 'label': label, 'vname': image_index}  # 根据图片和标签创建字典
        return sample  # 返回该样本


class ODIData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, images, dmos):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.images = images  # 目录里的所有文件
        self.dmos = dmos

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = skimage.io.imread(img_path)
        img = skimage.transform.resize(img, (1920, 3840))
        img = (img * 255).astype(np.uint8)

        img = np.transpose(img, [2, 0, 1])
        v1 = img[:, 0:384, :]
        v2 = img[:, 384:384 * 2, :]
        v3 = img[:, 384 * 2:384 * 3, :]
        v4 = img[:, 384 * 3:384 * 4, :]
        v5 = img[:, 384 * 4:384 * 5, :]

        label = self.dmos[index]

        sample = {'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4, 'v5': v5, 'label': label, 'vname': image_index}  # 根据图片和标签创建字典
        return sample  # 返回该样本