import argparse
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from fall_detection_with_signalenhancement.creat_dataset import CreatDataset
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='LSTM Model Training')
parser.add_argument('--outf', default='../model/LSTM/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='../model/LSTM/LSTM.pth',
                    help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 120
BATCH_SIZE = 16
LR = 0.001
Input_dim = 30
Hidden_dim = 100
Layer_dim = 2
Output_dim = 2

trainset = CreatDataset(
    csv_path='../Data/fall_detection/train/Signal_enhancement/',
    label_path='../Data/fall_detection/train/label/label_2class.csv')

testset = CreatDataset(
    csv_path='../Data/fall_detection/test/Signal_enhancement/',
    label_path='../Data/fall_detection/test/label/label_2class.csv')

trainloader = DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=False)

testloader = DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    shuffle=False)


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # One time step
        r_out, _ = self.rnn(x, None)
        out = self.fc(r_out[:, -1, :])
        return out

rnn = RNNModel(Input_dim, Hidden_dim, Layer_dim, Output_dim).to(device)

writer = SummaryWriter()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=LR)

if __name__ == "__main__":
    # for epoch in range(EPOCH):
    #     rnn.train()
    #     sum_loss = 0.0
    #     correct = 0.0
    #     total = 0.0
    #     # if (epoch + 1) % 30 == 0:
    #     #     LR /= 10
    #     for step, (inputs, labels) in enumerate(trainloader):
    #         length = len(trainloader)
    #         labels = labels.squeeze(1)
    #         labels = labels.long()
    #         inputs = inputs.squeeze(1)
    #         inputs = inputs.float()
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #
    #         outputs = rnn(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         sum_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += predicted.eq(labels.data).cpu().sum()
    #         print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
    #               % (epoch + 1, (step + 1 + epoch * length), sum_loss / (step + 1), 100. * float(correct) / total))
    #
    #         writer.add_scalar('Train/loss', sum_loss / (step + 1), epoch + 1)
    #         writer.add_scalar('Train/accuracy', 100. * float(correct) / total, epoch + 1)
    #
    #     print("Waiting Test!")
    #     with torch.no_grad():
    #         testcorrect = 0.0
    #         total = 0.0
    #         for data in testloader:
    #             rnn.eval()
    #             inputs, labels = data
    #             labels = labels.squeeze(1)
    #             labels = labels.long()
    #             inputs = inputs.squeeze(1)
    #             inputs = inputs.float()
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = rnn(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             testcorrect += predicted.eq(labels.data).cpu().sum()
    #
    #         print('测试分类准确率为：%.3f%%' % (100 * float(testcorrect) / total))
    #         acc = 100. * float(testcorrect) / total
    #         torch.save(rnn.state_dict(), '%s/LSTM_2class_%03d.pth' % (args.outf, epoch + 1))
    #         writer.add_scalar('Test/accuracy', acc, epoch + 1)
    #
    # print("Training Finished, TotalEPOCH=%d" % EPOCH)

    pthfile = r'C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\model\LSTM\LSTM_2class_107.pth'
    rnn.load_state_dict(torch.load(pthfile))
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in testloader:
        labels = labels.squeeze(1)
        labels = labels.long()
        inputs = inputs.squeeze(1)
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = rnn(inputs)

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ('fall', 'non-fall')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('C:/Users/user/Desktop/Lecture Materials/Dissertation/Workspace_CSI/Confusion_matrix/LSTM_03.png')


