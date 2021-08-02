import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm


class ResidualBlock(nn.Module):
    """Residual Block with some normalization. Conv : W = (W - F + 2P) /S + 1 / TransPosed : W = (Win - 1) * S - 2P + F + OutP"""
    def __init__(self, dim_in, dim_out, mid_dim=None, spec_norm=False, LR=0.02, stride=1):
        super(ResidualBlock, self).__init__()

        if not mid_dim:
            mid_dim = dim_out

        if spec_norm:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, mid_dim, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(mid_dim, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True),
                SpectralNorm(nn.Conv2d(mid_dim, dim_out, kernel_size=3, stride=stride, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True))
        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(mid_dim, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True),
                nn.Conv2d(mid_dim, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True))

        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        return self.main(x) + self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, dim_in, dim_out, mid_dim=None, LR=0.02, spec_norm=False, stride=1):
        super().__init__()
        if not mid_dim:
            mid_dim = dim_out

        if spec_norm:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, mid_dim, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(mid_dim, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True),
                SpectralNorm(nn.Conv2d(mid_dim, dim_out, kernel_size=3, stride=stride, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True))
        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(mid_dim, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True),
                nn.Conv2d(mid_dim, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=True))

    def forward(self, x):
        return self.main(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)