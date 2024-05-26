from typing import Optional, Callable
import torch.nn as nn
from torch import Tensor
from resnet import resnet18, resnet34, resnet50
import torch


def norm_layer(norm: Optional[Callable[..., nn.Module]]):
    if norm is None:
        return nn.BatchNorm2d
    else: return norm


class DoubleConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            bias: bool = False,
            norm: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DoubleConv2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias),
            norm_layer(norm)(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias),
            norm_layer(norm)(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            bias: bool = False,
            norm: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias, norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        p = self.pool(x)
        p = self.drop(p)
        return x, p


class BackBoneEncoderBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            backbone: str,
            pretrained: bool,
            freeze_grad: bool
    ) -> None:
        super(BackBoneEncoderBlock, self).__init__()
        backbones = {
            'resnet18': (resnet18, [64, 128, 256, 512]),
            'resnet34': (resnet34, [64, 128, 256, 512]),
            'resnet50': (resnet50, [64, 256, 512, 1024])
        }
        if backbone not in backbones:
            raise ValueError('Backbone must be resnet18, resnet34, or resnet50.')
        
        model, self.filters = backbones[backbone]
        model = model(channels, pretrained=pretrained)

        if freeze_grad:
            for param in model.parameters():
                param.requires_grad = False

        self.input_layer = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu
        )
        self.pool = model.pool

        self.encoder1 = model.layer1
        self.encoder2 = model.layer2
        self.encoder3 = model.layer3
        self.encoder4 = model.layer4

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.input_layer(x)
        p = self.pool(x1)
        x2 = self.encoder1(p)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        return x1, p, x2, x3, x4, x5


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            bias: bool = False,
            norm: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
            up_in_channels: int = None,
            up_out_channels: int = None
    ) -> None:
        super(DecoderBlock, self).__init__()
        if up_in_channels is None:
            up_in_channels = in_channels
        if up_out_channels is None:
            up_out_channels = out_channels

        self.trans = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2, bias=bias)
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias, norm)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.trans(x1)
        x = self.conv(torch.cat([x2, x], dim=1))
        x = self.drop(x)
        return x

