import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def variance(data):
    n = len(data)
    mean = sum(data) / n
    deviations = [(x - mean) ** 2 for x in data]
    var = sum(deviations) / (n - 1)
    return var


def NISE(raw_data, i, w, P, step):
    '''
    :param raw_data: csi_matrix after antenna selection, 300*30
    :param window: sliding window
    :param iteration:the number of iterations
    :param step:the step size of window movement
    :return: the enhanced sequential data
    '''

    S_i = raw_data[:, i]
    S = S_i
    ST = []
    for m in range(0, P):
        N = len(S)
        k = 0
        while k + w <= N:
            vk = variance(S[k:k + w])
            ST.append(vk)
            k += step
        S = ST
        ST = []
    return S

def SE_csv(data, i):
    SE_data = NISE(data, i, 40, 3, 1)
    return SE_data

def signal_enhance(path, csv_file):
    # path = './Data/crowd_counting/train/Antenna_selection'
    # csv_file = 'ONE_210705_bc611_walk_2nd_023.dat.csv'
    csv_path = os.path.join(path, csv_file)
    data = pd.read_csv(csv_path, header=None)
    data = data.values
    # fig = plt.figure(figsize=(20, 10), dpi=100)
    # ax1 = fig.add_subplot(121)
    # ax1.set_title(csv_file)
    # plt.xlabel('Package')
    # plt.ylabel('CSI Amplitude')
    # plt.plot(data)
    total_SE_data = np.zeros((30, 183))
    for i in range(0, 30):
        SE_data = SE_csv(data, i)
        total_SE_data[i, :] = SE_data
    return total_SE_data
    #     ax2 = fig.add_subplot(122)
    #     ax2.set_title('Signal enhancement')
    #     plt.xlabel('Package')
    #     plt.ylabel('CSI Amplitude')
    #     plt.plot(SE_data)
    # plt.show()

def creat_csv_file(path, csv_file, target_path):
    total_SE_data = signal_enhance(path, csv_file)
    data = pd.DataFrame(total_SE_data.T)
    output_file = os.path.join(target_path, csv_file)
    data.to_csv(output_file)

if __name__ == "__main__":
    path = '../Data/fall_detection/test/Antenna_selection/'
    target_path = '../Data/fall_detection/test/Signal_enhancement/'
    for file in filter(lambda x: (x[-4:] == '.csv'), os.listdir(path)):
        creat_csv_file(path, file, target_path)