import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from fall_detection_with_signalenhancement.activity_segmentation import activity_segment



class DatasetFromcsv(Dataset):
    def __init__(self, csv_path, label_path):
        self.csv_path = csv_path

        self.data_info = pd.read_csv(label_path, header=None)
        self.csv_arr = np.asarray(self.data_info.iloc[1:, 1])
        self.label_arr = np.asarray(self.data_info.iloc[1:, 2:])

        self.data_len = len(self.data_info.index) - 1

        self.data_transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize([0.5], [0.5])
        ])

    def __getitem__(self, index):
        single_csv_name = self.csv_arr[index]
        csv_data = pd.read_csv(self.csv_path + single_csv_name + ".csv", header=None)
        csv_data = self.data_preprocess(csv_data)
        # T_s, T_e = self.activity_segmentation(csv_data, 20, 1)
        # csv_data = csv_data[T_s:T_e, :]
        csv_data = self.data_transform(csv_data)

        label = self.label_arr[index]
        label = self.label_process_2class(label)
        return csv_data, label

    def __len__(self):
        return self.data_len

    def data_preprocess(self, data):
        data = data.values
        data = data[1:, 1:]
        return data

    def label_process(self, label):
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

    def label_process_2class(self,label):
        label = torch.Tensor([int(x) for x in label])
        a_temp = torch.Tensor([1, 0])
        b_temp = torch.Tensor([0, 1])
        if torch.equal(label, a_temp):
            label = torch.Tensor([0])
        elif torch.equal(label, b_temp):
            label = torch.Tensor([1])
        return label

    def activity_segmentation(self, csv_data, w, step):
        return activity_segment(csv_data, w, step)

def CreatDataset(csv_path, label_path):
    return DatasetFromcsv(csv_path, label_path)