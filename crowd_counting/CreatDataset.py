import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import color
from skimage import io

class CreateDatasetFromImages(Dataset):
    def __init__(self, label_path, img_path):

        self.img_path = img_path
        self.to_tensor = transforms.ToTensor()  # 将数据转换成tensor形式

        # 读取 csv 文件
        # 利用pandas读取label_csv文件
        self.data_info = pd.read_csv(label_path, header=None)
        # 文件第一列包含img文件的名称
        self.img_arr = np.asarray(self.data_info.iloc[1:, 1])
        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[1:, 2:])

        # 计算 length
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        # 从 csv_arr中得到索引对应的文件名
        single_img_name = self.img_arr[index]

        # 读取图像文件
        img_as_img = io.imread(self.img_path + single_img_name + ".jpg")
        # 生成灰度图
        imgGray = color.rgb2gray(img_as_img)
        imgGray = imgGray.astype('float32')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        imgGray = transform(imgGray)
        imgGray = imgGray.expand(1, 30, 300)

        # 得到图像的 label
        label = self.label_arr[index]
        label = torch.Tensor([int(x) for x in label])

        a_temp = torch.Tensor([1, 0, 0])
        b_temp = torch.Tensor([0, 1, 0])
        c_temp = torch.Tensor([0, 0, 1])
        if torch.equal(label, a_temp):
            label = torch.Tensor([0])
        elif torch.equal(label, b_temp):
            label = torch.Tensor([1])
        elif torch.equal(label, c_temp):
            label = torch.Tensor([2])


        return imgGray, label   # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.data_len

def CreatDataset(label_path, img_path):
    return CreateDatasetFromImages(label_path, img_path)