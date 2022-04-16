import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import os
import matplotlib.pyplot as plt
sys.path.append("../..")

def plot_img(data = "train"):
    if data == "train":
        path = '../Data/fall_detection/train'
    elif data == "dev":
        path = '../Data/fall_detection/dev'
    elif data == "test":
        path = '../Data/fall_detection/test'
    else:
        print("please input correct dataset type")
    csv_path = os.path.join(path, './csv_file/')
    for file in filter(lambda x: (x[-4:] == '.csv'), os.listdir(csv_path)):
        file_path = os.path.join(csv_path, file)
        basename = os.path.splitext(file)[0]
        data = pd.read_csv(file_path, header=None)
        data = preprocessing.normalize(data)
        plt.axis('off')
        fig1 = plt.gcf()
        fig1.set_size_inches(0.9, 3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.imshow(data.T,interpolation = "nearest", aspect = "auto", cmap='jet')
        plt.savefig(csv_path+'/image/'+basename+'.jpg', transparent=True, dpi=100, pad_inches = 0)
        plt.show()

def labelfile_generation(data = "train"):
    if data == "train":
        path = '../Data/fall_detection/train'
    elif data == "dev":
        path = '../Data/fall_detection/dev'
    elif data == "test":
        path = '../Data/fall_detection/test'
    else:
        print("please input correct dataset type")
    filename = []
    fall = []
    walk = []
    liedown = []
    sitdown = []
    standup = []

    img_path = os.path.join(path, './img_file/')
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                basename = os.path.splitext(file)[0]
                filename.append(basename)
                if basename[0:3] == 'ZER':
                    person0.append(1)
                    person1.append(0)
                    person2.append(0)
                elif basename[0:3] == 'ONE':
                    person0.append(0)
                    person1.append(1)
                    person2.append(0)
                elif basename[0:3] == 'TWO':
                    person0.append(0)
                    person1.append(0)
                    person2.append(1)
                elif basename[0:3] == 'THR':
                    person0.append(0)
                    person1.append(0)
                    person2.append(1)

                # if basename.find("sitting") == -1:
                #     sitting.append(0)
                # else:
                #     sitting.append(1)
                #
                # if basename.find("standing") == -1:
                #     standing.append(0)
                # else:
                #     standing.append(1)
                #
                # if basename.find("standup") == -1:
                #     standup.append(0)
                # else:
                #     standup.append(1)
                #
                # if basename.find("sitdown") == -1:
                #     sitdown.append(0)
                # else:
                #     sitdown.append(1)
                #
                # if basename.find("walk") == -1:
                #     walk.append(0)
                # else:
                #     walk.append(1)

    # file_label = list(zip(filename, person0, person1, person2, sitting, walk, standing, sitdown, standup))
    file_label = list(zip(filename, person0, person1, person2))
    # name=['filename', 'person0', 'person1', 'person2', 'sitting', 'walk', 'standing', 'sitdown', 'standup']
    name = ['filename', 'person0', 'person1', 'person2']
    test=pd.DataFrame(columns=name, data=file_label)
    #labelname = os.path.join(path, './label/label_8classes.csv')
    labelname = os.path.join(path, './label/label_3classes.csv')
    test.to_csv(labelname)

if __name__ == "__main__":
    plot_img(data="train")
    #plot_img(data="dev")
    # plot_img(data="test")
    # labelfile_generation(data="train")
    #labelfile_generation(data="dev")
    # labelfile_generation(data="test")