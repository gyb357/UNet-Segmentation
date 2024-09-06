from typing import Optional, Callable, Tuple, List
import torch.nn as nn
from utils import operate, operate_elif
from torch import Tensor
import torch
from resnet import resnet


UNET_FILTERS = {
    None: (64, 128, 256, 512, 1024),
    'resnet18': (64, 128, 256, 512, 1024),
    'resnet34': (64, 128, 256, 512, 1024),
    'resnet50': (64, 256, 512, 1024, 2048),
    'resnet101': (64, 256, 512, 1024, 2048),
    'resnet152': (64, 256, 512, 1024, 2048)
}


def normalize_layer(normalize: Optional[Callable[..., nn.Module]] = None) -> nn.Module:
    return operate(normalize is None, nn.BatchNorm2d, normalize)


class DoubleConv2d(nn.Module):
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
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias),
            normalize_layer(normalize)(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias),
            normalize_layer(normalize)(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
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

        self.trans = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2, bias=bias)
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, bias, normalize)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.trans(x1)
        x = self.conv(torch.cat([x2, x], dim=1))
        x = self.drop(x)
        return x


class OutBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_classes: int
    ) -> None:
        super(OutBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(
            self,
            channels: int,
            num_classes: int,
            backbone_name: Optional[str] = None,
            pretrained: bool = False,
            freeze_grad: bool = False,
            kernel_size: int = 3,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
            init_weights: bool = False
    ) -> None:
        super(UNet, self).__init__()
        # Filters
        self.filters = UNET_FILTERS[backbone_name]
        self.k = 1 # Filter coefficient

        self.backbone_state = operate_elif(
            backbone_name in ['resnet18', 'resnet34'], 's',               # Shallow
            backbone_name in ['resnet50', 'resnet101', 'resnet152'], 'd', # Deep
            None
        )
        if self.backbone_state == 'd':
            self.k = 2

        # Encoder blocks (with backbone)
        if self.backbone_state:
            self.backbone = resnet(backbone_name, channels, pretrained=pretrained)

            for param in self.backbone.parameters():
                param.requires_grad = not freeze_grad

            self.encoder_input = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu
            )
            self.encoder = nn.ModuleList()
            self.pool = self.backbone.pool

            for name, module in self.backbone.named_children():
                if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    self.encoder.append(module)

        # Encoder blocks (without backbone)
        else:
            self.encoder = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            in_channels = channels
            for out_channels in self.filters[:-1]:
                self.encoder.append(
                    EncoderBlock(in_channels, out_channels, kernel_size, bias, normalize, dropout)
                )
                in_channels = out_channels

        # Center blocks
        self.center = DoubleConv2d(self.filters[-2]*self.k, self.filters[-1]*self.k, kernel_size, bias, normalize)

        # Decoder blocks
        self.decoder = nn.ModuleList()
        filters_len = len(self.filters) - 1
        k_ = 1
        
        for i in range(filters_len):
            if self.backbone_state and i == filters_len - 1:
                k_ = 2
                
            self.decoder.append(
                DecoderBlock(self.filters[-1 - i]*self.k, self.filters[-2 - i]*self.k*k_, kernel_size, bias, normalize)
            )
        if self.backbone_state == 's': self.decoder.append(DecoderBlock(128, 64, kernel_size, bias, normalize, dropout, 64, 64))
        if self.backbone_state == 'd': self.decoder.append(DecoderBlock(192, 64, kernel_size, bias, normalize, dropout, 256, 128))

        # Out blocks
        self.out = OutBlock(self.filters[0], self.filters[0], num_classes)

        # Initialize weights
        if init_weights:
            init_targets = operate(self.backbone_state is None, self.modules(), [self.center, self.decoder, self.out])

            for module in init_targets:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x_out = []

        # Encoder
        if self.backbone_state:
            x = self.encoder_input(x)
            p = self.pool(x)
            x_out.append(x)
        else:
            p = x

        for encoder in self.encoder:
            if self.backbone_state:
                p = encoder(p)
                x_out.append(p)
            else:
                x, p = encoder(p)
                x_out.append(x)
            e_out = p

        # Center
        c = self.center(e_out)

        # Decoder
        d = self.pool(c)
        for i, decoder in enumerate(self.decoder):
            x = x_out[-1 - i]
            if self.backbone_state is None:
                x = self.pool(x)

            d = decoder(d, x)

        # Output
        return self.out(d)


class EnsembleUNet(nn.Module):
    def __init__(self, unet: List[UNet]) -> None:
        super(EnsembleUNet, self).__init__()
        self.unet = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> Tensor:
        out = [model(x) for model in self.unet]
        out = torch.mean(torch.stack(out), dim=0)
        return out

