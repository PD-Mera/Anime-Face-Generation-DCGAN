from torch import nn
import torch
import torch.nn.functional as F


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding,
            bias = bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.generation_sequence = nn.Sequential(
            # N x Z x 1 x 1 -> N x 512 x 4 x 4
            GenBlock(in_channels, 512, 4, 1, 0),
            # N x 512 x 4 x 4 -> N x 256 x 8 x 8
            GenBlock(512, 256, 4, 2, 1),
            # N x 256 x 8 x 8 -> N x 128 x 16 x 16
            GenBlock(256, 128, 4, 2, 1),
            # N x 128 x 16 x 16 -> N x 64 x 32 x 32
            GenBlock(128, 64, 4, 2, 1),
        )
        
        # N x 64 x 32 x 32 -> N x 3 x 64 x 64
        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.generation_sequence(x)
        x = self.output_block(x)
        return x


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding,
            bias = bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.discrimination_sequence = nn.Sequential(
            DisBlock(64, 128, 4, 2, 1),
            DisBlock(128, 256, 4, 2, 1),
            DisBlock(256, 512, 4, 2, 1)
        )
        self.output_block = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.discrimination_sequence(x)
        x = self.output_block(x)
        return x