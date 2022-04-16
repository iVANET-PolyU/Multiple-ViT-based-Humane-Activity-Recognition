import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import os
import matplotlib.pyplot as plt
sys.path.append("../../..")

def plot_img(data = "train"):
    if data == "train":
        path = '../Data/MTL/train/DWT'
    elif data == "dev":
        path = '../Data/MTL/dev/DWT'
    elif data == "test":
        path = '../Data/MTL/test/DWT'
    else:
        print("please input correct dataset type")
    csv_path = os.path.join(path, './CSV_file/')
    for file in filter(lambda x: (x[-4:] == '.csv'), os.listdir(csv_path)):
        file_path = os.path.join(csv_path, file)
        basename = os.path.splitext(file)[0]
        data = pd.read_csv(file_path, header=None)
        data = preprocessing.normalize(data)
        plt.axis('off')
        fig1 = plt.gcf()
        fig1.set_size_inches(3, 0.9)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.imshow(data.T,interpolation = "nearest", aspect = "auto", cmap='jet')
        plt.savefig(path+'/img_file/'+basename+'.jpg', transparent=True, dpi=100, pad_inches = 0)
        plt.show()


def labelfile_generation(data = "train"):
    if data == "train":
        path = '../Data/MTL/train/DWT'
    elif data == "dev":
        path = '../Data/MTL/dev/DWT'
    elif data == "test":
        path = '../Data/MTL/test/DWT'
    else:
        print("please input correct dataset type")
    filename = []
    zero = []
    one = []
    oneplus = []
    fall = []
    walk = []
    liedown = []
    sitdown = []
    standup = []
    static = []
    dynamic = []

    csv_path = os.path.join(path, './img_file/')
    for file in filter(lambda x: (x[-4:] == '.jpg'), os.listdir(csv_path)):
        basename = os.path.splitext(file)[0]
        filename.append(basename)
        if 'fall' in basename:
            fall.append(1)
            walk.append(0)
            liedown.append(0)
            sitdown.append(0)
            standup.append(0)
            dynamic.append(1)
            static.append(0)
        elif 'sitdown' in basename:
            fall.append(0)
            walk.append(0)
            liedown.append(0)
            sitdown.append(1)
            standup.append(0)
            dynamic.append(1)
            static.append(0)
        elif 'walk' in basename:
            fall.append(0)
            walk.append(1)
            liedown.append(0)
            sitdown.append(0)
            standup.append(0)
            dynamic.append(1)
            static.append(0)
        elif 'standup' in basename:
            fall.append(0)
            walk.append(0)
            liedown.append(0)
            sitdown.append(0)
            standup.append(1)
            dynamic.append(1)
            static.append(0)
        elif 'liedown' in basename:
            fall.append(0)
            walk.append(0)
            liedown.append(1)
            sitdown.append(0)
            standup.append(0)
            dynamic.append(1)
            static.append(0)
        elif 'sitting' in basename:
            fall.append(0)
            walk.append(0)
            liedown.append(0)
            sitdown.append(0)
            standup.append(0)
            dynamic.append(0)
            static.append(1)
        elif 'standing' in basename:
            fall.append(0)
            walk.append(0)
            liedown.append(0)
            sitdown.append(0)
            standup.append(0)
            dynamic.append(0)
            static.append(1)

        if 'ZERO' in basename:
            zero.append(1)
            one.append(0)
            oneplus.append(0)
            fall.append(0)
            walk.append(0)
            liedown.append(0)
            sitdown.append(0)
            standup.append(0)
            dynamic.append(0)
            static.append(0)
        elif 'ONE' in basename:
            zero.append(0)
            one.append(1)
            oneplus.append(0)
        elif 'TWO' in basename:
            zero.append(0)
            one.append(0)
            oneplus.append(1)
        elif 'THREE' in basename:
            zero.append(0)
            one.append(0)
            oneplus.append(1)
        else:
            zero.append(0)
            one.append(1)
            oneplus.append(0)

    file_label = list(zip(filename, zero, one, oneplus, fall, walk, liedown, sitdown, standup, dynamic, static))
    name = ['filename', 'zero', 'one', 'oneplus', 'fall', 'walk', 'liedown', 'sitdown', 'standup', 'dynamic', 'static']
    test = pd.DataFrame(columns=name, data=file_label)
    labelname = os.path.join(path, './label/label.csv')
    test.to_csv(labelname)

if __name__ == "__main__":
    plot_img(data="train")
    plot_img(data="dev")
    plot_img(data="test")
    labelfile_generation(data="train")
    labelfile_generation(data="dev")
    labelfile_generation(data="test")
