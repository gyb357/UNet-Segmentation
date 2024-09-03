import torch.nn as nn
from typing import List, Optional, Callable
from unet_module import EncoderBlock, DoubleConv2d, DecoderBlock
from torch import Tensor
import torch
from resnet import resnet


class UNet(nn.Module):
    filters: List[int] = [64, 128, 256, 512, 1024]

    def __init__(
            self,
            channels: int,
            num_classes: int,
            kernel_size: int = 3,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
    ) -> None:
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = channels

        for out_channels in self.filters[:-1]:
            self.encoder.append(
                EncoderBlock(in_channels, out_channels, kernel_size, bias, normalize, dropout)
            )
            in_channels = out_channels

        # Center
        self.center = DoubleConv2d(self.filters[-2], self.filters[-1], kernel_size, bias, normalize)

        # Decoder
        self.decoder = nn.ModuleList()

        for i in range(len(self.filters) - 1):
            self.decoder.append(
                DecoderBlock(self.filters[-1 - i], self.filters[-2 - i], kernel_size, bias, normalize, dropout)
            )

        # Out
        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.filters[0], self.filters[0], kernel_size=2, stride=2),
            nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_out = []
        e_out = x

        # Encoder
        p = x
        for encoder in self.encoder:
            x, p = encoder(p)
            x_out.append(x)
            e_out = p
            print(p.shape)

        # Center
        c = self.center(e_out)
        d = self.pool(c)
        print(d.shape)

        for i, decoder in enumerate(self.decoder):
            x = x_out[-1 - i]
            x = self.pool(x)
            d = decoder(d, x)
            print(d.shape)

        # Out
        out = self.out(d)
        return out


class SUNet(nn.Module):
    filters: List[int] = [64, 128, 256, 512, 1024]
    
    def __init__(
            self,
            channels: int,
            num_classes: int,
            name: str,
            pretrained: bool,
            freeze_grad: bool = False,
            kernel_size: int = 3,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(SUNet, self).__init__()
        self.backbone = resnet(name, channels, pretrained=pretrained)

        for param in self.backbone.parameters():
            param.requires_grad = not freeze_grad
        
        # Encoder
        self.encoder = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
        )
        self.pool = self.backbone.pool

        # Center
        self.center = DoubleConv2d(self.filters[-2], self.filters[-1], kernel_size, bias, normalize)

        # Decoder
        self.decoder = nn.ModuleList()

        for i in range(len(self.filters) - 1):
            self.decoder.append(
                DecoderBlock(self.filters[-1 - i], self.filters[-2 - i], kernel_size, bias, normalize, dropout)
            )
        self.decoder.append(DecoderBlock(self.filters[1], self.filters[0], kernel_size, bias, normalize, dropout, self.filters[0], self.filters[0]))


        # Out
        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.filters[0], self.filters[0], kernel_size=2, stride=2),
            nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_out = []
        e_out = x

        # Encoder
        x = self.encoder(x)
        p = self.pool(x)
        x_out.append(x)
        print(x.shape)

        for name, module in self.backbone.named_children():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                p = module(p)
                x_out.append(p)
                e_out = p
                print(p.shape)

        # Center
        c = self.center(e_out)
        d = self.pool(c)
        print(d.shape)

        for i, decoder in enumerate(self.decoder):
            x = x_out[-1 - i]
            d = decoder(d, x)
            print(d.shape)

        # Out
        out = self.out(d)
        return out


class DUNet(nn.Module):
    filters: List[int] = [64, 256, 512, 1024, 2048]

    def __init__(
            self,
            channels: int,
            num_classes: int,
            name: str,
            pretrained: bool,
            freeze_grad: bool = False,
            kernel_size: int = 3,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(DUNet, self).__init__()
        self.backbone = resnet(name, channels, pretrained=pretrained)

        for param in self.backbone.parameters():
            param.requires_grad = not freeze_grad
        
        # Encoder
        self.encoder = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
        )
        self.pool = self.backbone.pool

        # Center
        self.center = DoubleConv2d(self.filters[-1], self.filters[-1]*2, kernel_size, bias, normalize)

        # Decoder
        self.decoder = nn.ModuleList()

        for i in range(len(self.filters) - 2):
            self.decoder.append(
                DecoderBlock(self.filters[-1 - i]*2, self.filters[-2 - i]*2, kernel_size, bias, normalize, dropout)
            )
        self.decoder.append(DecoderBlock(self.filters[2], self.filters[1], kernel_size, bias, normalize, dropout, self.filters[2], self.filters[1]))
        self.decoder.append(DecoderBlock(self.filters[0], self.filters[0], kernel_size, bias, normalize, dropout, self.filters[0], self.filters[0]))

        # Out
        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.filters[0], self.filters[0], kernel_size=2, stride=2),
            nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_out = []
        e_out = x

        # Encoder
        x = self.encoder(x)
        p = self.pool(x)
        x_out.append(x)
        print(x.shape)

        for name, module in self.backbone.named_children():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                p = module(p)
                x_out.append(p)
                e_out = p
                print(p.shape)

        # Center
        c = self.center(e_out)
        d = self.pool(c)
        print(d.shape)

        for i, decoder in enumerate(self.decoder):
            x = x_out[-1 - i]
            d = decoder(d, x)
            print(d.shape)

        # Out
        out = self.out(d)
        return out


# model = UNet(3, 1)
model = DUNet(3, 1, 'resnet50', True)
print(model)
inp = torch.rand((1, 3, 320, 320))
out = model(inp)
print(out.shape)

