import os
import pandas as pd

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
    sitdown = []
    walk = []
    standup = []
    liedown = []

    csv_path = os.path.join(path, './Signal_enhancement/')
    for file in filter(lambda x: (x[-4:] == '.csv'), os.listdir(csv_path)):
        basename = os.path.splitext(file)[0]
        filename.append(basename)
        if basename[0:3] == 'fal':
            fall.append(1)
            sitdown.append(0)
            walk.append(0)
            standup.append(0)
            liedown.append(0)
        elif basename[0:3] == 'sit':
            fall.append(0)
            sitdown.append(1)
            walk.append(0)
            standup.append(0)
            liedown.append(0)
        elif basename[0:3] == 'wal':
            fall.append(0)
            sitdown.append(0)
            walk.append(1)
            standup.append(0)
            liedown.append(0)
        elif basename[0:3] == 'sta':
            fall.append(0)
            sitdown.append(0)
            walk.append(0)
            standup.append(1)
            liedown.append(0)
        elif basename[0:3] == 'lie':
            fall.append(0)
            sitdown.append(0)
            walk.append(0)
            standup.append(0)
            liedown.append(1)

    file_label = list(zip(filename, fall, sitdown, walk, standup, liedown))
    name = ['filename', 'fall', 'sitdown', 'walk', 'standup', 'liedown']
    test=pd.DataFrame(columns=name, data=file_label)
    labelname = os.path.join(path, './label/label.csv')
    test.to_csv(labelname)

def labelfile_generation_2(data = "train"):
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
    unfall = []

    csv_path = os.path.join(path, './Signal_enhancement/')
    for file in filter(lambda x: (x[-4:] == '.csv'), os.listdir(csv_path)):
        basename = os.path.splitext(file)[0]
        filename.append(basename)
        if basename[0:3] == 'fal':
            fall.append(1)
            unfall.append(0)
        elif basename[0:3] == 'sit':
            fall.append(0)
            unfall.append(1)
        elif basename[0:3] == 'wal':
            fall.append(0)
            unfall.append(1)
        elif basename[0:3] == 'sta':
            fall.append(0)
            unfall.append(1)
        elif basename[0:3] == 'lie':
            fall.append(0)
            unfall.append(1)

    file_label = list(zip(filename, fall, unfall))
    name = ['filename', 'fall', 'unfall']
    test=pd.DataFrame(columns=name, data=file_label)
    labelname = os.path.join(path, './label/label_2class.csv')
    test.to_csv(labelname)

def labelfile_generation_3(data = "train"):
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
    nonfall = []

    csv_path = os.path.join(path, './Antenna_selection/')
    for file in filter(lambda x: (x[-4:] == '.csv'), os.listdir(csv_path)):
        basename = os.path.splitext(file)[0]
        filename.append(basename)
        if basename[0:3] == 'fal':
            fall.append(1)
            nonfall.append(0)
        else:
            fall.append(0)
            nonfall.append(1)

    file_label = list(zip(filename, fall, nonfall))
    name = ['filename', 'fall', 'nonfall']
    test=pd.DataFrame(columns=name, data=file_label)
    labelname = os.path.join(path, './label/label_2class_antenna_selection.csv')
    test.to_csv(labelname)



if __name__ == "__main__":
    # labelfile_generation(data="train")
    # #labelfile_generation(data="dev")
    # labelfile_generation(data="test")
    labelfile_generation_3(data="train")
    #labelfile_generation_2(data="dev")
    labelfile_generation_3(data="test")


