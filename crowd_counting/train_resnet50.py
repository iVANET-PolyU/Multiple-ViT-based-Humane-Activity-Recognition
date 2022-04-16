import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
from crowd_counting.TEST import ResNet50
from crowd_counting.CreatDataset import CreatDataset

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='UNet Model Training')
parser.add_argument('--outf', default='./model/ResNet50/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='./model/ResNet50/ResNet50.pth',
                    help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

# 超参数设置
EPOCH = 120  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 16  # 批处理尺寸(batch_size)
LR = 0.1  # 学习率

# 准备数据集并预处理

trainset = CreatDataset(
    label_path='/Data/crowd_counting/train/label/label_3classes.csv',
    img_path='/Data/crowd_counting/train/img_file/')  # 训练数据集

testset = CreatDataset(
        label_path ='/Data/crowd_counting/test/label/label_3classes.csv',
        img_path ='/Data/crowd_counting/test/img_file/')

trainset1, trainset2 = torch.utils.data.random_split(trainset, [1400,  1050])
train_db, test_db = torch.utils.data.random_split(testset, [140, 300])
concat_data = torch.utils.data.ConcatDataset([trainset1, train_db])

trainloader = torch.utils.data.DataLoader(
    dataset=concat_data,
    batch_size=BATCH_SIZE,
    shuffle=False)

# devloader = torch.utils.data.DataLoader(
#     dataset=dev_db,
#     batch_size=BATCH_SIZE,
#     shuffle=False, )


testloader = torch.utils.data.DataLoader(
        dataset=test_db,
        batch_size=BATCH_SIZE,
        shuffle=False)


net = ResNet50().to(device)
# UNet = UNetModel()

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(UNet.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#optimizer = optim.RMSprop(UNet.parameters(), lr=LR, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

# 训练
if __name__ == "__main__":
    print("Start Training, ResNet50!")  # 定义遍历数据集的次数
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        # if (epoch+1) <= 5:
        #     LR = 0.1*(epoch+1)/5
        if (epoch+1) % 30 == 0:
            LR /= 10
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            # 准备数据
            length = len(trainloader)
            inputs, labels = data
            labels = labels.squeeze(1)
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * float(correct) / total))
        writer.add_scalar('Train/loss', sum_loss / (i + 1), epoch+1)
        writer.add_scalar('Train/accuracy', 100. * float(correct) / total, epoch+1)


        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            testcorrect = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                labels = labels.squeeze(1)
                labels = labels.long()
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                testcorrect += (predicted == labels).cpu().sum()

            print('测试分类准确率为：%.3f%%' % (100 * float(testcorrect) / total))
            acc = 100. * float(testcorrect) / total
            print('Saving model......')
            torch.save(net.state_dict(), '%s/ResNet50_%03d.pth' % (args.outf, epoch + 1))
            writer.add_scalar('Test/accuracy', acc, epoch + 1)

    print("Training Finished, TotalEPOCH=%d" % EPOCH)

