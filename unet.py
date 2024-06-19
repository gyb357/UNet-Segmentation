from typing import Optional, Callable, List, Tuple
import torch.nn as nn
from utils import operate
from torch import Tensor
import torch
from resnet import resnet


# resnet50 or higher is under development


def norm_layer(norm: Optional[Callable[..., nn.Module]] = None) -> nn.Module:
    return operate(norm is None, nn.BatchNorm2d, norm)


class DoubleConv2d(nn.Module):
    stride: int = 1
    padding: int = 1

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
            nn.Conv2d(in_channels, out_channels, kernel_size, self.stride, self.padding, bias=bias),
            norm_layer(norm)(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, self.stride, self.padding, bias=bias),
            norm_layer(norm)(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class EncoderBlock(nn.Module):
    pool_kernel_size: int = 2
    pool_stride: int = 2

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
            norm: Optional[Callable[..., nn.Module]] = None,
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
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias, norm)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.trans(x1)
        x = self.conv(torch.cat([x2, x], dim=1))
        x = self.drop(x)
        return x


class EncoderBlocks(nn.Module):
    backbone_layers: List[str] = ['layer1', 'layer2', 'layer3', 'layer4']

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
        self.filters = filters
        self.backbone = backbone

        if backbone is None:
            self.encoder = nn.ModuleList()
            in_channels = channels
            for out_channels in filters[:-1]:
                self.encoder.append(EncoderBlock(in_channels, out_channels, kernel_size, bias, norm, dropout))
                in_channels = out_channels
        else:
            self.encoder, self.filters = resnet(backbone, channels, pretrained=pretrained)

            for param in self.encoder.parameters():
                param.requires_grad = not freeze_grad

            self.inputs = nn.Sequential(
                self.encoder.conv1,
                self.encoder.bn1,
                self.encoder.relu
            )
            self.pool = self.encoder.pool

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        x_out = []
        e_out = x

        if self.backbone is None:
            p = x
            for encoder in self.encoder:
                x, p = encoder(p)
                x_out.append(x)
                e_out = p
        else:
            x = self.inputs(x)
            p = self.pool(x)
            x_out.append(x)
            for name, module in self.encoder.named_children():
                if name in self.backbone_layers:
                    p = module(p)
                    x_out.append(p)
                    e_out = p
        return x_out, e_out


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
        self.decoder = nn.ModuleList()

        for i in range(len(filters) - 1):
            self.decoder.append(DecoderBlock(filters[-1 - i], filters[-2 - i], kernel_size, bias, norm, dropout))

        if backbone is not None:
            self.decoder.append(DecoderBlock(filters[1], filters[0], kernel_size, bias, norm, dropout, filters[0], filters[0]))

    def forward(self, x_out: List[Tensor], c: Tensor) -> Tensor:
        d = self.pool(c)
        print(d.shape)

        for i, decoder in enumerate(self.decoder):
            x = x_out[-1 - i]
            if self.backbone is None:
                x = self.pool(x)
                print(x.shape)

            d = decoder(d, x)
        return d


class UNet(nn.Module):
    filters: List[int] = [64, 128, 256, 512, 1024]
    trans_kernel_size: int = 2
    trans_stride: int = 2

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
            dropout: float = 0.0,
            init_weights: bool = True
    ) -> None:
        super(UNet, self).__init__()
        self.encoder = EncoderBlocks(channels, self.filters, backbone, pretrained, freeze_grad, kernel_size, bias, norm, dropout)
        self.filters = self.encoder.filters

        self.center = DoubleConv2d(self.filters[-2], self.filters[-1], kernel_size, bias, norm)
        self.decoder = DecoderBlocks(self.filters, backbone, kernel_size, bias, norm, dropout)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.filters[0], self.filters[0], self.trans_kernel_size, self.trans_stride),
            nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        )

        if init_weights:
            init_target = operate(backbone is None, self.modules(), [self.center, self.decoder, self.out])

            for module in init_target:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x_out, e_out = self.encoder(x)
        c = self.center(e_out)
        d = self.decoder(x_out, c)
        return self.out(d)


class EnsembleUNet(nn.Module):
    def __init__(self, unet_models: List[UNet]) -> None:
        super(EnsembleUNet, self).__init__()
        self.unet_models = nn.ModuleList(unet_models)

    def forward(self, x: Tensor) -> Tensor:
        out = [model(x) for model in self.unet_models]
        out = torch.mean(torch.stack(out), dim=0)
        return out


model = UNet(3, 1, 'resnet50', True)
print(model)
inp = torch.rand((1, 3, 320, 320))
out = model(inp)
print(out.shape)