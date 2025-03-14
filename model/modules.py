import torch.nn as nn
import torch
from typing import Optional, Callable, Tuple
from torch import Tensor


class DoubleConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DoubleConv2d, self).__init__()
        self.normalize = normalize or nn.BatchNorm2d
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            self.normalize(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            self.normalize(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv2d(in_channels, out_channels, bias, normalize)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv(x)
        p = self.pool(x)
        p = self.drop(p)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(
            self,
            up_in_channels: int,
            up_out_channels: int,
            in_channels: int,
            out_channels: int,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(DecoderBlock, self).__init__()
        self.trans = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2, bias=bias)
        self.conv = DoubleConv2d(in_channels, out_channels, bias, normalize)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.trans(x1)
        x = self.conv(torch.cat([x, x2], dim=1))
        x = self.drop(x)
        return x


class OutputBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_classes: int,
            backbone: str,
            bias: bool = False
    ) -> None:
        super(OutputBlock, self).__init__()
        if backbone:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=bias)
            )
        else: self.layers = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

