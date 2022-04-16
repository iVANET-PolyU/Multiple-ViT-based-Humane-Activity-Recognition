import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = double_conv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #print('sizes', x1.size(), x2.size(), diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = double_conv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.activation = nn.Sigmoid()
        #self.conv = nn.Conv2d(16,8,kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        x1 = self.inc(x)
        #print('inc output shape:\t', x1.shape)
        x2 = self.down1(x1)
        #print('down1 output shape:\t', x2.shape)
        x3 = self.down2(x2)
        #print('down2 output shape:\t', x3.shape)
        x4 = self.down3(x3)
        #print('down3 output shape:\t', x4.shape)
        x5 = self.down4(x4)
        #print('down4 output shape:\t', x5.shape)
        x = self.up1(x5, x4)
        #print('up1 output shape:\t', x.shape)
        x = self.up2(x, x3)
        #print('up2 output shape:\t', x.shape)
        x = self.up3(x, x2)
        #print('up3 output shape:\t', x.shape)
        x = self.up4(x, x1)
        #print('up4 output shape:\t', x.shape)
        x = self.activation(x)
        x = self.avgpool(x)
        #print('avgpool output shape:\t', x.shape)
        out = self.fc(x)
        #print('fc output shape:\t', out.shape)
        return out

def UNetModel():
    return UNet(n_channels=1, n_classes=3)

# if __name__ == "__main__":
#     net = UNet(n_channels = 1, n_classes=3)
#     X = torch.rand((1, 1, 30, 300))
#     X = net(X)