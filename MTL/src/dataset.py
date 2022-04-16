import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
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
        # img_data = img_data.unsqueeze(0)

        label = self.label_arr[index]
        label = torch.Tensor([int(x) for x in label])

        return img_data, label

    def __len__(self):
        return self.data_len

def get_data(img_path, label_path):
    return DatasetFromcsv(img_path, label_path)

def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return loader