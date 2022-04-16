import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import color
from skimage import io

class DatasetFromcsv(Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path

        self.data_info = pd.read_csv(label_path, header=None)
        self.img_arr = np.asarray(self.data_info.iloc[1:, 1])
        self.label_arr = np.asarray(self.data_info.iloc[1:, 2:])

        self.data_len = len(self.data_info.index) - 1

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __getitem__(self, index):
        single_img_name = self.img_arr[index]
        img_data = io.imread(self.img_path + single_img_name + ".jpg", as_gray=True)
        img_data = self.data_transform(img_data.T)

        label = self.label_arr[index]
        label = self.label_process(label)
        return img_data, label

    def __len__(self):
        return self.data_len

    def label_process(self,label):
        label = torch.Tensor([int(x) for x in label])
        a_temp = torch.Tensor([1, 0, 0, 0, 0])
        b_temp = torch.Tensor([0, 1, 0, 0, 0])
        c_temp = torch.Tensor([0, 0, 1, 0, 0])
        d_temp = torch.Tensor([0, 0, 0, 1, 0])
        e_temp = torch.Tensor([0, 0, 0, 0, 1])
        if torch.equal(label, a_temp):
            label = torch.Tensor([0])
        elif torch.equal(label, b_temp):
            label = torch.Tensor([1])
        elif torch.equal(label, c_temp):
            label = torch.Tensor([2])
        elif torch.equal(label, d_temp):
            label = torch.Tensor([3])
        elif torch.equal(label, e_temp):
            label = torch.Tensor([4])
        return label


def CreatDataset(img_path, label_path):
    return DatasetFromcsv(img_path, label_path)