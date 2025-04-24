from . import *


# Configuration of the decoder layers
_UNET3PLUS_CONFIGS = {
    'd1': (64, 128, 256, 512, 1024),
    'd2': (64, 128, 256, 64, 1024),
    'd3': (64, 128, 64, 64, 1024),
    'd4': (64, 64, 64, 64, 1024)
}


class UNet3Plus(nn.Module):
    """
    UNet3+ implementation for image segmentation
    Supports ResNet backbone or standard UNet encoder,
    optional deep supervision and classification-guided module (CGM)

    https://arxiv.org/abs/2004.08790
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
            deep_supervision: bool = False,
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
            deep_supervision (bool): Whether to use deep supervision
            cgm (bool): Whether to use CGM
        """

        super(UNet3Plus, self).__init__()
        # Attributes
        self.backbone = backbone
        self.deep_supervision = deep_supervision
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
                for p in itertools.chain(
                    self.e1.parameters(), self.e2.parameters(),
                    self.e3.parameters(), self.e4.parameters(), self.e5.parameters()
                ):
                    p.requires_grad = False

        # Encoder layers (without backbone)
        else:
            self.e1 = EncoderBlock(channels, 64, bias, normalize, dropout)
            self.e2 = EncoderBlock(64, 128, bias, normalize, dropout)
            self.e3 = EncoderBlock(128, 256, bias, normalize, dropout)
            self.e4 = EncoderBlock(256, 512, bias, normalize, dropout)

            # Center layer
            self.e5 = DoubleConv2d(512, 1024, bias, normalize)
        
        # Decoder
        self.d1 = DecoderBlock3Plus(_UNET3PLUS_CONFIGS['d1'], mid_channels=64, bias=bias, normalize=normalize, dropout=dropout)
        self.d2 = DecoderBlock3Plus(_UNET3PLUS_CONFIGS['d2'], mid_channels=64, bias=bias, normalize=normalize, dropout=dropout)
        self.d3 = DecoderBlock3Plus(_UNET3PLUS_CONFIGS['d3'], mid_channels=64, bias=bias, normalize=normalize, dropout=dropout)
        self.d4 = DecoderBlock3Plus(_UNET3PLUS_CONFIGS['d4'], mid_channels=64, bias=bias, normalize=normalize, dropout=dropout)

        # Output layer
        self.out = OutputBlock(64, 64, num_classes, backbone, bias)

        # Deep supervision
        if self.deep_supervision:
            self.aux1 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux2 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux3 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux4 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)

        # Classification-Guided Module head
        if self.cgm:
            self.cgm_head = CGMHead(1024, bias)

        # Initialize weights
        if init_weights:
            self._init_weights()

    def _init_weights(self) -> None:
        encoder_modules = []
        if self.backbone:
            encoder_modules = [self.e1, self.e2, self.e3, self.e4, self.e5]
        for m in self.modules():
            if self.backbone and any(m in mod.modules() for mod in encoder_modules):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
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

        # CGM: soft gating
        if self.cgm:
            probas = self.cgm_head(e5)
            fg = probas[:, 1].view(-1, 1, 1, 1) # foreground probability

        # Decoder
        sz4 = e4.shape[-2:]
        d1 = self.d1((e1, e2, e3, e4, e5), sz4)
        sz3 = e3.shape[-2:]
        d2 = self.d2((e1, e2, e3, d1, e5), sz3)
        sz2 = e2.shape[-2:]
        d3 = self.d3((e1, e2, d2, d1, e5), sz2)
        sz1 = e1.shape[-2:]
        d4 = self.d4((e1, d3, d2, d1, e5), sz1)

        # CGM gating
        if self.cgm:
            d4 = d4 * fg

        # Output layer forward pass
        out = self.out(d4)

        # Deep supervision outputs
        if self.deep_supervision and self.training:
            size = x.shape[2:]
            aux1 = F.interpolate(self.aux1(d1), size=size, mode='bilinear', align_corners=True)
            aux2 = F.interpolate(self.aux2(d2), size=size, mode='bilinear', align_corners=True)
            aux3 = F.interpolate(self.aux3(d3), size=size, mode='bilinear', align_corners=True)
            aux4 = F.interpolate(self.aux4(d4), size=size, mode='bilinear', align_corners=True)
            return out, aux1, aux2, aux3, aux4
        return out

