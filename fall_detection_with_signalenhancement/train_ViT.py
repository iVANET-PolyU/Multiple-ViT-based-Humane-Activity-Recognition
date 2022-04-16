from fall_detection_with_signalenhancement.ViT import ViT
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

parser = argparse.ArgumentParser(description='ViT Model Training')
parser.add_argument('--outf', default='../model/ViT/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='../model/ViT/ViT.pth',
                    help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 150
BATCH_SIZE = 16
LR = 0.001
Num_classes = 2
Dim = 30
Depth = 6
Heads = 8
Mlp_dim = 100
Dropout = 0.1
Emb_dropout = 0.1

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

v = ViT(num_classes=Num_classes, dim=Dim, depth=Depth, heads=Heads, mlp_dim=Mlp_dim,
        dropout=Dropout, emb_dropout=Emb_dropout).to(device)

writer = SummaryWriter()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(v.parameters(), lr=LR)

# for epoch in range(EPOCH):
#     epoch_loss = 0
#     epoch_accuracy = 0
#
#     for data, label in trainloader:
#         label = label.squeeze(1)
#         label = label.long()
#         data = data.squeeze(1)
#         data = data.float()
#         data = data.to(device)
#         label = label.to(device)
#
#         output = v(data)
#         loss = criterion(output, label)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         acc = (output.argmax(dim=1) == label).float().mean()
#         epoch_accuracy += acc / len(trainloader)
#         epoch_loss += loss / len(trainloader)
#
#     with torch.no_grad():
#         epoch_val_accuracy = 0
#         epoch_val_loss = 0
#         for data, label in testloader:
#             label = label.squeeze(1)
#             label = label.long()
#             data = data.squeeze(1)
#             data = data.float()
#             data = data.to(device)
#             label = label.to(device)
#
#             val_output = v(data)
#             val_loss = criterion(val_output, label)
#
#             acc = (val_output.argmax(dim=1) == label).float().mean()
#             epoch_val_accuracy += acc / len(testloader)
#             epoch_val_loss += val_loss / len(testloader)
#
#     writer.add_scalar('Train/loss', epoch_loss, epoch + 1)
#     writer.add_scalar('Train/accuracy', epoch_accuracy, epoch + 1)
#     writer.add_scalar('Test/accuracy', epoch_val_accuracy, epoch + 1)
#
#     print(
#         f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
#     )
#
#     torch.save(v.state_dict(), '%s/ViT_2class_%03d.pth' % (args.outf, epoch + 1))

pthfile = r'C:\Users\user\Desktop\Lecture Materials\Dissertation\Workspace_CSI\model\ViT\ViT_2class_128.pth'
v.load_state_dict(torch.load(pthfile))
y_pred = []
y_true = []

# iterate over test data
for inputs, labels in testloader:
    labels = labels.squeeze(1)
    labels = labels.long()
    inputs = inputs.squeeze(1)
    inputs = inputs.float()
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = v(inputs)

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
plt.figure(figsize=(6, 3))
sn.heatmap(df_cm, annot=True)
plt.savefig('C:/Users/user/Desktop/Lecture Materials/Dissertation/Workspace_CSI/Confusion_matrix/ViT_2class_02.png')
