import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint as cp


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool, dropout: float):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool, dropout: float):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias, dropout)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool, dropout: float):
        super(DecoderBlock, self).__init__()
        self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            DoubleConv2d(in_channels, out_channels, kernel_size, bias, dropout)
        )

    def forward(self, x1, x2):
        x = self.trans(x1)
        x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool, activation):
        super(OutputBlock, self).__init__()
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)]
        layer.append(activation)
        self.conv = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter: int, kernel_size: int, bias: bool, dropout: float, activation: nn, checkpoint: bool):
        super(UNet, self).__init__()
        self.checkpoint = checkpoint
        self.e1 = EncoderBlock(in_channels, filter, kernel_size, bias, dropout)
        self.e2 = EncoderBlock(filter, filter*2, kernel_size, bias, dropout)
        self.e3 = EncoderBlock(filter*2, filter*4, kernel_size, bias, dropout)
        self.e4 = EncoderBlock(filter*4, filter*8, kernel_size, bias, dropout)
        self.neck = DoubleConv2d(filter*8, filter*16, kernel_size, bias, dropout)
        self.d4 = DecoderBlock(filter*16, filter*8, kernel_size, bias, dropout)
        self.d3 = DecoderBlock(filter*8, filter*4, kernel_size, bias, dropout)
        self.d2 = DecoderBlock(filter*4, filter*2, kernel_size, bias, dropout)
        self.d1 = DecoderBlock(filter*2, filter, kernel_size, bias, dropout)
        self.out = OutputBlock(filter, out_channels, bias, activation)

    def init_weights(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def cp_forward(self, x):
        e1, p1 = cp(self.e1, x)
        e2, p2 = cp(self.e2, p1)
        e3, p3 = cp(self.e3, p2)
        e4, p4 = cp(self.e4, p3)
        neck = self.neck(p4)
        d4 = cp(self.d4, neck, e4)
        d3 = cp(self.d3, d4, e3)
        d2 = cp(self.d2, d3, e2)
        d1 = cp(self.d1, d2, e1)
        out = self.out(d1)
        return out

    def forward(self, x):
        if self.checkpoint == True:
            return self.cp_forward(x)
        else:
            e1, p1 = self.e1(x)
            e2, p2 = self.e2(p1)
            e3, p3 = self.e3(p2)
            e4, p4 = self.e4(p3)
            neck = self.neck(p4)
            d4 = self.d4(neck, e4)
            d3 = self.d3(d4, e3)
            d2 = self.d2(d3, e2)
            d1 = self.d1(d2, e1)
            out = self.out(d1)
            return out

