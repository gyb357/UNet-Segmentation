from typing import Optional, Callable, Tuple
import torch.nn as nn
from utils import operate
from torch import Tensor
import torch


def normalize_layer(normalize: Optional[Callable[..., nn.Module]] = None) -> nn.Module:
    return operate(normalize is None, nn.BatchNorm2d, normalize)


class DoubleConv2d(nn.Module):
    stride: int = 1
    padding: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DoubleConv2d, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, self.stride, self.padding, bias=bias),
            normalize_layer(normalize)(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, self.stride, self.padding, bias=bias),
            normalize_layer(normalize)(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
    pool_kernel_size: int = 2
    pool_stride: int = 2

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias, normalize)
        self.pool = nn.MaxPool2d(self.pool_kernel_size, self.pool_stride)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv(x)
        p = self.pool(x)
        p = self.drop(p)
        return x, p


class DecoderBlock(nn.Module):
    trans_kernel_size: int = 2
    trans_stride: int = 2

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
            up_in_channels: Optional[int] = None,
            up_out_channels: Optional[int] = None
    ) -> None:
        super(DecoderBlock, self).__init__()
        if up_in_channels is None:
            up_in_channels = in_channels
        if up_out_channels is None:
            up_out_channels = out_channels

        self.trans = nn.ConvTranspose2d(up_in_channels, up_out_channels, self.trans_kernel_size, self.trans_stride, bias=bias)
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias, normalize)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.trans(x1)
        x = self.conv(torch.cat([x2, x], dim=1))
        x = self.drop(x)
        return x

