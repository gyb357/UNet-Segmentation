from . import *


# Configuration of the decoder layers
UNET_CONFIGS = {
    'shallow': {
        'conv_filters': [768, 384, 192, 128, 64],
        'trans_filters': [512, 512, 256, 128, 64]
    },
    'deep': {
        'conv_filters': [1536, 768, 384, 128, 64],
        'trans_filters': [2048, 512, 256, 128, 64]
    },
    'default': {
        'conv_filters': [1024, 512, 256, 128, 64],
        'trans_filters': [1024, 512, 256, 128, 64]
    }
}


class UNet(nn.Module):
    """
    UNet implementation for image segmentation.
    This model can use either a ResNet backbone as encoder or a standard UNet encoder.

    Structure
    ---------
    https://arxiv.org/abs/1505.04597
    """
    
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
        """

        super(UNet, self).__init__()
        # Attributes
        self.backbone = backbone

        # Get configuration based on backbone depth
        self.backbone_depth = ternary_op_elif(
            backbone in ['resnet18', 'resnet34'], 'shallow',
            backbone in ['resnet50', 'resnet101', 'resnet152'], 'deep',
            'default'
        )
        config = UNET_CONFIGS[self.backbone_depth]
        self.decoder_conv_filters = config['conv_filters']
        self.decoder_trans_filters = config['trans_filters']

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
        self.d1 = DecoderBlock(self.decoder_trans_filters[0], 512, self.decoder_conv_filters[0], 512, bias, normalize, dropout)
        self.d2 = DecoderBlock(self.decoder_trans_filters[1], 256, self.decoder_conv_filters[1], 256, bias, normalize, dropout)
        self.d3 = DecoderBlock(self.decoder_trans_filters[2], 128, self.decoder_conv_filters[2], 128, bias, normalize, dropout)
        self.d4 = DecoderBlock(self.decoder_trans_filters[3], 64, self.decoder_conv_filters[3], 64, bias, normalize, dropout)
        
        # Output layer
        self.out = OutputBlock(self.decoder_conv_filters[4], 32, num_classes, backbone)

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

    def _get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

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
        return self.out(d4)

