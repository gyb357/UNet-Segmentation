from . import *


_UNET3PLUS_CONFIGS = {
    'd4': (64, 128, 256, 512, 1024),
    'd3': (64, 128, 256, 64, 1024),
    'd2': (64, 128, 64, 64, 1024),
    'd1': (64, 64, 64, 64, 1024)
}


class UNet3Plus(nn.Module):
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
            cgm: bool = False
    ) -> None:
        """
        Args:
            channels (int): Number of input channels
            num_classes (int): Number of output classes
            backbone (str): Backbone architecture for encoder
            pretrained (str): Pretrained model path
            freeze_backbone (bool): Whether to freeze backbone weights
            bias (bool): Whether to use bias in convolutional layers
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability
            init_weights (bool): Whether to initialize weights
            cgm (bool): Whether to use CGM
        """

        super(UNet3Plus, self).__init__()
        # Attributes
        self.backbone = backbone
        self.cgm = cgm

        # Encoder layers (with backbone)
        if backbone:
            encoder_layers = list(resnet(backbone, pretrained, channels).children())

            self.e1 = nn.Sequential(*encoder_layers[:3])
            self.e2 = nn.Sequential(*encoder_layers[3:5])
            self.e3 = encoder_layers[5]
            self.e4 = encoder_layers[6]
            self.e5 = encoder_layers[7]

            # Freeze the backbone if requested
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
        self.d1 = DecoderBlock3Plus(
            in_channels_list=_UNET3PLUS_CONFIGS['d1'],
            mid_channels=64,
            normalize=normalize,
            bias=bias,
            dropout=dropout
        )
        self.d2 = DecoderBlock3Plus(
            in_channels_list=_UNET3PLUS_CONFIGS['d2'],
            mid_channels=64,
            normalize=normalize,
            bias=bias,
            dropout=dropout
        )
        self.d3 = DecoderBlock3Plus(
            in_channels_list=_UNET3PLUS_CONFIGS['d3'],
            mid_channels=64,
            normalize=normalize,
            bias=bias,
            dropout=dropout
        )
        self.d4 = DecoderBlock3Plus(
            in_channels_list=_UNET3PLUS_CONFIGS['d4'],
            mid_channels=64,
            normalize=normalize,
            bias=bias,
            dropout=dropout
        )

        # Output layer
        self.out = OutputBlock(64, 64, num_classes, backbone, bias)

        if cgm:
            self.cgm_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Conv2d(1024, 2, kernel_size=1, bias=bias),
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                nn.Sigmoid()
            )

        # Initialize weights
        if init_weights:
            self._init_weights()

    def _init_weights(self) -> None:
        if self.backbone:
            # If backbone is provided, initialize only the decoder layers
            encoder_modules = set()
            encoder_layers = [self.e1, self.e2, self.e3, self.e4, self.e5]
            for layer in encoder_layers:
                encoder_modules.update(layer.modules())

            for m in self.modules():
                if m in encoder_modules:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            # If backbone is not provided, initialize all layers
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

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
            e5 = self.e5(p4)

        # CGM
        if self.cgm:
            cls = self.cgm_head(e5).argmax(1).view(-1, 1, 1, 1).float()

        # Decoder fusion passes
        target4 = e4.shape[-2:]
        d1 = self.d1((e1, e2, e3, e4, e5), target4)

        target3 = e3.shape[-2:]
        d2 = self.d2((e1, e2, e3, d1, e5), target3)

        target2 = e2.shape[-2:]
        d3 = self.d3((e1, e2, d2, d1, e5), target2)

        target1 = e1.shape[-2:]
        d4 = self.d4((e1, d3, d2, d1, e5), target1)

        # CGM weighting
        if self.cgm:
            d4 = d4 * cls
        return self.out(d4)

