import torch.nn as nn
from torch import Tensor
import torch


class DoubleConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            bias: bool,
    ) -> None:
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            bias: bool,
            dropout: float
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        p = self.pool(x)
        p = self.drop(p)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            bias: bool,
            dropout: float
    ) -> None:
        super(DecoderBlock, self).__init__()
        self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.trans(x1)
        x = self.drop(x)
        x = self.conv(torch.cat([x2, x], dim=1))
        return x


class OutputBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool
    ) -> None:
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            filter: int,
            kernel_size: int,
            bias: bool,
            dropout: float,
            init_weights: bool,
    ) -> nn.Module:
        super(UNet, self).__init__()
        # UNet structure
        self.e1 = EncoderBlock(in_channels, filter, kernel_size, bias, dropout)
        self.e2 = EncoderBlock(filter, filter*2, kernel_size, bias, dropout)
        self.e3 = EncoderBlock(filter*2, filter*4, kernel_size, bias, dropout)
        self.e4 = EncoderBlock(filter*4, filter*8, kernel_size, bias, dropout)
        self.neck = DoubleConv2d(filter*8, filter*16, kernel_size, bias)
        self.d4 = DecoderBlock(filter*16, filter*8, kernel_size, bias, dropout)
        self.d3 = DecoderBlock(filter*8, filter*4, kernel_size, bias, dropout)
        self.d2 = DecoderBlock(filter*4, filter*2, kernel_size, bias, dropout)
        self.d1 = DecoderBlock(filter*2, filter, kernel_size, bias, dropout)
        self.out = OutputBlock(filter, out_channels, bias)

        if init_weights == True:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    if m.weight is not None:
                        nn.init.normal_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
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

