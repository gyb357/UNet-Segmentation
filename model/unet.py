import torch.nn as nn
from typing import Optional, Callable
from utils import ternary_op_elif
from resnet import resnet
from modules import EncoderBlock, DoubleConv2d, DecoderBlock, OutputBlock
from torch import Tensor


# UNet's decoder layers filter configuration
UNET_CONFIGS = {
    # Conv filters
    'convolution':
    {
        None: [1024, 512, 256, 128, 64],
        'shal': [768, 384, 192, 128, 64],
        'deep': [1536, 768, 384, 128, 64],
    },
    # Transpose filters
    'transpose':
    {
        None: [1024, 512, 256, 128, 64],
        'shal': [512, 512, 256, 128, 64],
        'deep': [2048, 512, 256, 128, 64, 32],
    }
}


class UNet(nn.Module):
    def __init__(
            self,
            channels: int,
            num_classes: int,
            backbone: Optional[str] = None,
            pretrained: Optional[str] = None,
            freeze_backbone: bool = False,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
            init_weights: bool = False,
    ) -> None:
        super(UNet, self).__init__()

        # Attributes
        self.backbone = backbone
        # Depth of the backbone
        self.backbone_depth = ternary_op_elif(
            backbone in ['resnet18', 'resnet34'], 'shal',
            backbone in ['resnet50', 'resnet101', 'resnet152'], 'deep',
            None
        )
        # Set the decoder filters from the configuration
        self.decoder_conv_filters = UNET_CONFIGS['convolution'][self.backbone_depth]
        self.decoder_trans_filters = UNET_CONFIGS['transpose'][self.backbone_depth]

        # Encoder layers (with backbone)
        if backbone:
            encoder_layers = list(resnet(backbone, pretrained, channels).children())

            self.e1 = nn.Sequential(*encoder_layers[:3])
            self.e2 = nn.Sequential(*encoder_layers[3:5])
            self.e3 = encoder_layers[5]
            self.e4 = encoder_layers[6]
            self.e5 = encoder_layers[7]

            # Freeze the backbone
            if freeze_backbone:
                for p in self.parameters():
                    p.requires_grad = False

        # Encoder layers (without backbone)
        else:
            self.e1 = EncoderBlock(channels, 64, bias, normalize, dropout)
            self.e2 = EncoderBlock(64, 128, bias, normalize, dropout)
            self.e3 = EncoderBlock(128, 256, bias, normalize, dropout)
            self.e4 = EncoderBlock(256, 512, bias, normalize, dropout)

            # Center layer
            self.e5 = DoubleConv2d(512, 1024, bias, normalize)

        # Decoder layers
        self.d1 = DecoderBlock(self.decoder_conv_filters[0], 512, bias, normalize, dropout, self.decoder_trans_filters[0], 512)
        self.d2 = DecoderBlock(self.decoder_conv_filters[1], 256, bias, normalize, dropout, self.decoder_trans_filters[1], 256)
        self.d3 = DecoderBlock(self.decoder_conv_filters[2], 128, bias, normalize, dropout, self.decoder_trans_filters[2], 128)
        self.d4 = DecoderBlock(self.decoder_conv_filters[3], 64, bias, normalize, dropout, self.decoder_trans_filters[3], 64)

        # Output layer
        self.out = OutputBlock(self.decoder_conv_filters[4], num_classes, backbone)

        # Initialize weights
        if init_weights:
            self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            # Convolution layers
            if self.backbone:
                if not isinstance(m, (EncoderBlock)) and isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            # BatchNorm layers
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        if self.backbone:
            e1 = self.e1(x)
            e2 = self.e2(e1)
            e3 = self.e3(e2)
            e4 = self.e4(e3)
            e5 = self.e5(e4)
        else:
            e1, p1 = self.e1(x)
            e2, p2 = self.e2(p1)
            e3, p3 = self.e3(p2)
            e4, p4 = self.e4(p3)

            # Center layer
            e5 = self.e5(p4)

        # Decoder
        d1 = self.d1(e5, e4)
        d2 = self.d2(d1, e3)
        d3 = self.d3(d2, e2)
        d4 = self.d4(d3, e1)

        # Output
        return self.out(d4)

