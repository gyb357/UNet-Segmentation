from typing import Optional, Callable
import torch.nn as nn
from module import DoubleConv2d, EncoderBlock, BackBoneEncoderBlock, DecoderBlock
from torch import Tensor


class UNet(nn.Module):
    in_channels: int = 64
    filters = [64, 128, 256, 512]

    def __init__(
            self,
            channels: int,
            num_classes: int,
            backbone: str = None,
            pretrained: bool = False,
            freeze_grad: bool = False,
            kernel_size: int = 3,
            bias: bool = False,
            norm: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        super(UNet, self).__init__()
        if backbone is None:
            encoder = []
            encoder.append(EncoderBlock(channels, self.filters[0], kernel_size, bias, norm, dropout))

            for i in range(len(self.filters) - 1):
                encoder.append(EncoderBlock(self.filters[i], self.filters[i + 1], kernel_size, bias, norm, dropout))
            self.encoder = nn.Sequential(*encoder)
        else:
            self.encoder = BackBoneEncoderBlock(channels, backbone, pretrained, freeze_grad)
            self.filters = self.encoder.filters


        self.center = DoubleConv2d(self.filters[3]*2, self.filters[3]*2, kernel_size, bias, norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        decoder = []
        decoder1 = DecoderBlock(self.filters[3]*2, self.filters[3], kernel_size, bias, norm, dropout)
        decoder2 = DecoderBlock(self.filters[3], self.filters[2], kernel_size, bias, norm, dropout)
        decoder3 = DecoderBlock(self.filters[2], self.filters[1], kernel_size, bias, norm, dropout)
        decoder.append(decoder1)
        decoder.append(decoder2)
        decoder.append(decoder3)

        if backbone in ['resnet18', 'resnet34']:
            decoder4 = DecoderBlock(self.filters[1], self.filters[0], kernel_size, bias, norm, dropout)
            decoder5 = DecoderBlock(self.filters[1], self.filters[0], kernel_size, bias, norm, dropout, self.filters[0], self.filters[0])
            decoder.append(decoder4)
            decoder.append(decoder5)
        if backbone == 'resnet50':
            decoder4 = DecoderBlock(self.filters[0]*3, self.filters[0]*2, kernel_size, bias, norm, dropout, self.filters[1], self.filters[0]*2)
            decoder5 = DecoderBlock(self.filters[0] + 3, self.filters[0], kernel_size, bias, norm, dropout, self.filters[0]*2, self.filters[0])
            decoder.append(decoder4)
            decoder.append(decoder5)
        self.decoder = nn.Sequential(*decoder)


        self.out = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)



    def forward(self, x: Tensor) -> Tensor:
        # non backbone forward
        # e1, p1 = self.encoder[0](x)
        # e2, p2 = self.encoder[1](p1)
        # e3, p3 = self.encoder[2](p2)
        # e4, p4 = self.encoder[3](p3)
        # c = self.center(p4)
        # d4 = self.decoder[0](c, e4)
        # d3 = self.decoder[1](d4, e3)
        # d2 = self.decoder[2](d3, e2)
        # d1 = self.decoder[3](d2, e1)
        # out = self.out(d1)
        # return out

        # resnet18, resnet34 forward
        # e1, _, e2, e3, e4, e5 = self.encoder(x)
        # c = self.center(e5)
        # c = self.pool(c)
        # d5 = self.decoder[0](c, e5)
        # d4 = self.decoder[1](d5, e4)
        # d3 = self.decoder[2](d4, e3)
        # d2 = self.decoder[3](d3, e2)
        # d1 = self.decoder[4](d2, e1)
        # out = self.out(d1)
        # return out

        # resnet50 forward
        e1, _, e2, e3, e4, e5 = self.encoder(x)
        c = self.center(e5)
        d5 = self.decoder[0](c, e4)
        d4 = self.decoder[1](d5, e3)
        d3 = self.decoder[2](d4, e2)
        d2 = self.decoder[3](d3, e1)
        d1 = self.decoder[4](d2, x)
        out = self.out(d1)
        return out
    

model = UNet(
    channels=3,
    num_classes=1,
    backbone='resnet50',
    pretrained=True
).cuda()
# print(model)
inp = torch.rand((16, 3, 320, 320)).cuda()
out = model(inp)
print(out)