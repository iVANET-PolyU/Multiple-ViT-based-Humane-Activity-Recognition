import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from GPU_server.ViT_fall_detection.creat_dataset import CreatDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from GPU_server.ViT_fall_detection.Vision_transformer import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Hyperparameters initialization
EPOCH = 150
BATCH_SIZE = 64
LR = 0.001

# Model parameters initialization
Num_classes = 5
Dim = 90
Depth = 4
Heads = 8
Mlp_dim = 200
Dropout = 0.1
Emb_dropout = 0.1

# Creat train and validation dataset
trainset = CreatDataset(
    img_path='../../Data/fall_detection/train/DWT_file/image/',
    label_path='../../Data/fall_detection/train/DWT_file/label/label.csv')
trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=False)

testset = CreatDataset(
    img_path='../../Data/fall_detection/test/DWT_file/image/',
    label_path='../../Data/fall_detection/test/DWT_file/label/label.csv')
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)

devset = CreatDataset(
    img_path='../../Data/fall_detection/dev/DWT_file/image/',
    label_path='../../Data/fall_detection/dev/DWT_file/label/label.csv')
devloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = ViT(num_classes=Num_classes, dim=Dim, depth=Depth, heads=Heads, mlp_dim=Mlp_dim, dropout=Dropout, emb_dropout=Emb_dropout).to(device)
# model = ResNet50().to(device)
# model = nn.DataParallel(model)
# model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.1)

# Train model
Train_loss_list = []
Train_accuracy_list = []
Dev_accuracy_list = []
Dev_loss_list = []
best_accuracy = 0
best_epoch = 0
path = "result"
dir_name = os.path.join(path, './ViT_DWT_6')
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

for epoch in range(EPOCH):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in trainloader:
        label = label.squeeze(1)
        label = label.long()
        data = data.squeeze(1)
        data = data.float()
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(trainloader)
        epoch_loss += loss / len(trainloader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in devloader:
            label = label.squeeze(1)
            label = label.long()
            data = data.squeeze(1)
            data = data.float()
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(devloader)
            epoch_val_loss += val_loss / len(devloader)

    print(
        f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

    if epoch_val_accuracy > best_accuracy:
        best_accuracy = epoch_val_accuracy
        best_epoch = epoch + 1

    if epoch_val_accuracy > 0.8:
        torch.save(model.state_dict(), './result/ViT_DWT_6/model_%03d.pth' % (epoch + 1))

    Train_loss_list.append(epoch_loss)
    Train_accuracy_list.append(epoch_accuracy)
    Dev_accuracy_list.append(epoch_val_accuracy)
    Dev_loss_list.append(epoch_val_loss)

# Record result
# 1. creat accuracy_loss csv file
file_label = list(zip(Train_loss_list, Train_accuracy_list, Dev_accuracy_list, Dev_loss_list))
name = ["Train_loss", "Train_accuracy", "Dev_accuracy", "Dev_loss"]
test = pd.DataFrame(columns=name, data=file_label)
dataname = os.path.join(dir_name, './accuracy_loss.csv')
test.to_csv(dataname)

# 2. creat accuracy_loss image
fig = plt.figure(figsize=(20, 30), dpi=100)
ax1 = fig.add_subplot(311)
ax1.set_title('Loss')
plt.xlim(1, EPOCH)
plt.ylabel('Loss')
plt.plot(Train_loss_list)
plt.plot(Dev_loss_list)
plt.legend(['train', 'dev'], loc='upper left')
ax2 = fig.add_subplot(312)
ax2.set_title('Train Accuracy')
plt.xlim(1, EPOCH)
plt.ylabel('Accuracy')
plt.plot(Train_accuracy_list)
ax3 = fig.add_subplot(313)
ax3.set_title('Test Accuracy')
plt.xlim(1, EPOCH)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(Dev_accuracy_list)
imagename = os.path.join(dir_name, './accuracy_loss.jpg')
plt.savefig(imagename)
plt.show()

# 3. Generate confusion_matrix
pthfile = os.path.join(dir_name, './model_%03d.pth' % best_epoch)
model.load_state_dict(torch.load(pthfile))
y_pred = []
y_true = []
test_accuracy = 0

for inputs, labels in testloader:
    labels = labels.squeeze(1)
    labels = labels.long()
    inputs = inputs.squeeze(1)
    inputs = inputs.float()
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    test_acc = (outputs.argmax(dim=1) == labels).float().mean()
    test_accuracy += test_acc / len(testloader)

    outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
    y_pred.extend(outputs)  # Save Prediction

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

print(f"Testset accuracy of best epoch is {test_accuracy:.4f}\n")

# constant for classes
classes = ('fall', 'walk', 'liedown', 'sitdown', 'standup')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis], index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(6, 3))
sn.heatmap(df_cm, annot=True)
matrix_name = os.path.join(dir_name, './confusion_matrix.jpg')
plt.savefig(matrix_name)












