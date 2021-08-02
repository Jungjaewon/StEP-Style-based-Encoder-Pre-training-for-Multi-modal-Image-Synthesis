import torch.nn as nn

from block import ResidualBlock as ResBlock
from block import DoubleConv
from block import Up


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channel=3, spec_norm=False, LR=0.02):
        super(Discriminator, self).__init__()

        self.main = list()
        self.main.append(DoubleConv(in_channel, 16, spec_norm=spec_norm, stride=2, LR=LR)) # 256 -> 128
        self.main.append(DoubleConv(16, 32, spec_norm=spec_norm, stride=2, LR=LR)) # 128 -> 64
        self.main.append(DoubleConv(32, 64, spec_norm=spec_norm, stride=2, LR=LR)) # 64 -> 32
        self.main.append(DoubleConv(64, 128, spec_norm=spec_norm, stride=2, LR=LR)) # 32 -> 16
        self.main.append(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


class UNet(nn.Module):
    def __init__(self, n_channels=513, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128, stride=2)
        self.down2 = DoubleConv(128, 256, stride=2)
        self.down3 = DoubleConv(256, 512, stride=2)
        factor = 2 if bilinear else 1
        self.down4 = DoubleConv(512, 1024 // factor, stride=2)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.tanh(self.outc(x))
        return logits
