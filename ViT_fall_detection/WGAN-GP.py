import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import date, datetime
import logging
import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataroot = "./data"
batch_size = 6
num_epochs = 500
lr = 0.0002


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=0, bias=False)
        self.batchN1 = nn.LayerNorm([64, 48, 28])
        self.LeakyReLU1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64 * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN2 = nn.LayerNorm([64 * 2, 24, 14])
        self.LeakyReLU2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64 * 2, out_channels=64 * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN3 = nn.LayerNorm([64 * 4, 12, 7])
        self.LeakyReLU3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64 * 4, out_channels=1, kernel_size=(12, 7),  bias=False)
        # self.batchN4 = nn.LayerNorm([64 * 8, 2, 20])
        # self.LeakyReLU4 = nn.LeakyReLU(0.2, inplace=True)
        # self.conv5 = nn.Conv2d(in_channels=64 * 8, out_channels=1, kernel_size=(2, 20), bias=False)

    def forward(self, x):
        x = self.LeakyReLU1(self.batchN1(self.conv1(x)))
        x = self.LeakyReLU2(self.batchN2(self.conv2(x)))
        x = self.LeakyReLU3(self.batchN3(self.conv3(x)))
        # x = self.LeakyReLU4(self.batchN4(self.conv4(x)))
        x = self.conv4(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ConvT1 = nn.ConvTranspose2d(in_channels=100, out_channels=64 * 4, kernel_size=(12, 7),
                                         bias=False)  # 这里的in_channels是和初始的随机数有关
        self.batchN1 = nn.BatchNorm2d(64 * 4)
        self.relu1 = nn.ReLU()
        self.ConvT2 = nn.ConvTranspose2d(in_channels=64 * 4, out_channels=64 * 2, kernel_size=4, stride=2, padding=1,
                                         bias=False)  # 这里的in_channels是和初始的随机数有关
        self.batchN2 = nn.BatchNorm2d(64 * 2)
        self.relu2 = nn.ReLU()
        self.ConvT3 = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=64, kernel_size=4, stride=2, padding=1,
                                         bias=False)  # 这里的in_channels是和初始的随机数有关
        self.batchN3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.ConvT4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=0,
                                         bias=False)  # 这里的in_channels是和初始的随机数有关
        # self.batchN4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.ReLU()
        # self.ConvT5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=(3, 2), padding=(2, 7),
        #                                  bias=False)
        self.tanh = nn.Tanh()  # 激活函数

    def forward(self, x):
        x = self.relu1(self.batchN1(self.ConvT1(x)))
        x = self.relu2(self.batchN2(self.ConvT2(x)))
        x = self.relu3(self.batchN3(self.ConvT3(x)))
        # x = self.relu4(self.batchN4(self.ConvT4(x)))
        x = self.ConvT4(x)
        x = self.tanh(x)
        return x.view(-1, 3, 90, 50)




dataset = datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


D = Discriminator().to(device)
G = Generator().to(device)
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

W_D = []

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        # 训练Discriminator
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        noise = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)

        # 首先计算真实的图片的loss, d_loss_real
        outputs = D(imgs)
        d_loss_real = -torch.mean(outputs)
        # 接着计算假的图片的loss, d_loss_fake
        fake_images = G(noise)
        outputs = D(fake_images)
        d_loss_fake = torch.mean(outputs)
        # 接着计算penalty region 的loss, d_loss_penalty
        # 生成penalty region
        alpha = torch.rand((batch_size, 1, 1, 1)).to(device)
        x_hat = alpha * imgs.data + (1 - alpha) * fake_images.data
        x_hat.requires_grad = True
        # 将中间的值进行分类
        pred_hat = D(x_hat)
        # 计算梯度
        gradient = torch.autograd.grad(outputs=pred_hat, inputs=x_hat,
                                       grad_outputs=torch.ones(pred_hat.size()).to(device),
                                       create_graph=False, retain_graph=False)
        penalty_lambda = 10  # 梯度惩罚系数
        gradient_penalty = penalty_lambda * (
                (gradient[0].view(gradient[0].size()[0], -1).norm(p=2, dim=1) - 1) ** 2).mean()
        d_loss = d_loss_real + d_loss_fake + gradient_penalty
        Wasserstein_D = d_loss_fake - d_loss_real
        g_optimizer.zero_grad()  # 两个优化器梯度都要清0
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 训练Generator
        normal_noise = Variable(torch.randn(batch_size, 100, 1, 1)).normal_(0, 1).to(device)
        fake_images = G(normal_noise)  # 生成假的图片
        outputs = D(fake_images)  # 放入辨别器
        g_loss = -torch.mean(outputs)  # 希望生成器生成的图片判别器可以判别为真
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        test_noise = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
        batches_done = epoch * len(dataloader) + i
        if batches_done % 500 == 0 or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            gen_imgs = G(test_noise)
            gen_imgs = gen_imgs.view(-1, 3, 90, 50)
            gen_imgs = gen_imgs.mul(0.5).add(0.5)
            save_image(gen_imgs.data[:8], "./result/WGAN-GP1/%d.png" % batches_done, nrow=1, normalize=True)

        W_D.append(Wasserstein_D.cpu().data.numpy())


plt.title('Wasserstein Distance')
plt.xlabel('Itteration')
plt.xlim(1, num_epochs * len(dataloader))
plt.ylabel('Wasserstein Distance')
plt.plot(W_D)
plt.show()
plt.savefig('./result/WGAN-GP1/Wasserstein Distance.jpg')
