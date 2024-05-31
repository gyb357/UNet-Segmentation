from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from typing import Optional, Callable, List
import torch.nn as nn
from torch import Tensor
import torch


backbones = {
    'resnet18': (resnet18, [64, 128, 256, 512]),
    'resnet34': (resnet34, [64, 128, 256, 512]),
    'resnet50': (resnet50, [64, 256, 512, 1024, 2048]),
    'resnet101': (resnet101, [64, 256, 512, 1024, 2048]),
    'resnet152': (resnet152, [64, 256, 512, 1024, 2048])
}


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


class EncoderBlocks(nn.Module):
    def __init__(
            self,
            channels: int,
            filters: List[int],
            backbone: Optional[str] = None,
            pretrained: bool = False,
            freeze_grad: bool = False,
            kernel_size: int = 3,
            bias: bool = False,
            norm: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(EncoderBlocks, self).__init__()
        self.backbone = backbone

        if backbone is None:
            model, filters = backbones[backbone]


class DecoderBlocks(nn.Module):
    def __init__(
            self,
            filters: List[int],
            backbone: Optional[str] = None,
            kernel_size: int = 3,
            bias: bool = False,
            norm: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
    ) -> None:
        super(DecoderBlocks, self).__init__()
        self.backbone = backbone
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        decoder = []
        for i in range(len(filters) - 1):
            decoder.append(DecoderBlock(filters[-1 - i], filters[-2 - i], kernel_size, bias, norm, dropout))

        if backbone is not None:
            decoder.append(DecoderBlock(filters[1], filters[0], kernel_size, bias, norm, dropout, filters[0], filters[0]))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, e: List[Tensor], c: Tensor) -> Tensor:
        if self.backbone is not None:
            c = self.pool(c)

        d = c
        for i, decoder in enumerate(self.decoder):
            d = decoder(d, e[-i - 1])
        return d


class UNet(nn.Module):
    def __init__(
            self,
            channels: int,
            num_classes: int,
            backbone: Optional[str] = None,
            pretrained: bool = False,
            freeze_grad: bool = False,
            kernel_size: int = 3,
            bias: bool = False,
            norm: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(UNet, self).__init__()
        self.filters = [64, 128, 256, 512, 1024]

    def forward(self, x):
        pass


model = UNet(3, 1)
print(model)

