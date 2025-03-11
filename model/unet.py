from typing import Optional, Callable, List
from modules import DoubleConv2d, EncoderBlock, DecoderBlock, OutBlock
from torch import Tensor
from resnet import resnet
import torch.nn as nn
import torch


UNET_FILTERS = {
    None: (64, 128, 256, 512, 1024),
    'resnet18': (64, 128, 256, 512, 1024),
    'resnet34': (64, 128, 256, 512, 1024),
    'resnet50': (64, 256, 512, 1024, 2048),
    'resnet101': (64, 256, 512, 1024, 2048),
    'resnet152': (64, 256, 512, 1024, 2048)
}


class UNet(nn.Module):
    def __init__(
            self,
            channels: int,
            num_classes: int,
            backbone_name: Optional[str] = None,
            pretrained: bool = False,
            freeze_grad: bool = False,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
            init_weights: bool = False
    ) -> None:
        super(UNet, self).__init__()
        self.backbone_name = backbone_name

        # filters
        filters = UNET_FILTERS[backbone_name]

        # filter coefficient
        k = 1 if backbone_name in ['resnet18', 'resnet34'] else 2
        
        # encoder blocks (with backbone)
        if backbone_name:
            backbone = resnet(backbone_name, channels, pretrained=pretrained)

            # gradient freeze
            for param in backbone.parameters():
                param.requires_grad = not freeze_grad

            self.encoder_input = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu
            )
            self.encoder = nn.ModuleList()
            self.maxpool = backbone.maxpool

            for name, module in backbone.named_children():
                if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    self.encoder.append(module)

        # encoder blocks (without backbone)
        else:
            self.encoder = nn.ModuleList()
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            in_channels = channels

            for out_channels in filters[:-1]:
                self.encoder.append(EncoderBlock(in_channels, out_channels, bias, normalize, dropout))
                in_channels = out_channels

        # center blocks
        self.center = DoubleConv2d(filters[-2]*k, filters[-1]*k, bias, normalize)

        # decoder blocks
        self.decoder = nn.ModuleList()

        for i in range(len(filters) - 1):
            in_channels = filters[-1 - i]*k
            out_channels = filters[-2 - i]*k

            if k == 2 and i == len(filters) - 2:
                out_channels *= 2

            self.decoder.append(DecoderBlock(in_channels, out_channels, bias, normalize))

        if backbone_name:
            if k == 1:
                self.decoder.append(DecoderBlock(128, 64, bias, normalize, dropout, 64, 64))
            elif k == 2:
                self.decoder.append(DecoderBlock(192, 64, bias, normalize, dropout, 256, 128))

        # out blocks
        self.out = OutBlock(filters[0], filters[0], num_classes)

        # initialize weights
        if init_weights:
            init_targets = self.modules() if backbone_name else [self.center, self.decoder, self.out]

            for module in init_targets:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x_out = []

        # encoder input (with backbone)
        if self.backbone_name:
            x = self.encoder_input(x)
            p = self.maxpool(x)
            x_out.append(x)
        # encoder input (without backbone)
        else:
            p = x

        # encoder blocks
        for encoder in self.encoder:
            if self.backbone_name:
                p = encoder(p)
                x_out.append(p)
            else:
                x, p = encoder(p)
                x_out.append(x)
            e_out = p

        # center blocks
        c = self.center(e_out)

        # decoder blocks
        d = self.maxpool(c)

        for i, decoder in enumerate(self.decoder):
            x = x_out[-1 - i]
            if self.backbone_name is None:
                x = self.maxpool(x)

            d = decoder(d, x)

        # out blocks
        return self.out(d)

