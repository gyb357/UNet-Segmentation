from . import *


# Configuration of the encoder layers
_UNET2PLUS_CONFIGS = {
    'shallow': (64, 64, 128, 256, 512),    # resnet18,34
    'deep':    (64, 256, 512, 1024, 2048), # resnet50,101,152
    'default': (64, 128, 256, 512, 1024)
}


class UNet2Plus(nn.Module):
    """
    UNet2+ implementation for image segmentation
    Supports ResNet backbone or standard UNet encoder,
    optional deep supervision and classification-guided module (CGM)

    https://arxiv.org/abs/1807.10165
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
            backbone (str): Backbone architecture for encoder ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained (str): Pretrained model path
            freeze_backbone (bool): Whether to freeze backbone weights
            bias (bool): Whether to use bias in convolutional layers
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability
            init_weights (bool): Whether to initialize weights
            deep_supervision (bool): Whether to use deep supervision
            cgm (bool): Whether to use CGM(Classification-Guided Module)
        """

        super(UNet2Plus, self).__init__()
        # Attributes
        self.backbone = backbone
        self.deep_supervision = deep_supervision
        self.cgm = cgm

        # Select encoder config based on backbone depth
        base = ternary_op_elif(
            backbone in ['resnet18', 'resnet34'], 'shallow',
            backbone in ['resnet50', 'resnet101', 'resnet152'], 'deep',
            'default'
        )
        e1, e2, e3, e4, e5 = _UNET2PLUS_CONFIGS[base]

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
            self.e1 = EncoderBlock(channels, e1, bias, normalize, dropout)
            self.e2 = EncoderBlock(e1, e2, bias, normalize, dropout)
            self.e3 = EncoderBlock(e2, e3, bias, normalize, dropout)
            self.e4 = EncoderBlock(e3, e4, bias, normalize, dropout)

            # Center layer
            self.e5 = DoubleConv2d(e4, e5, bias, normalize)

        # Decoder layers
        self.d01 = DecoderBlock2Plus([e1, e2], 64, bias, normalize, dropout)             # 
        self.d11 = DecoderBlock2Plus([e2, e3], 64, bias, normalize, dropout)
        self.d21 = DecoderBlock2Plus([e3, e4], 64, bias, normalize, dropout)
        self.d31 = DecoderBlock2Plus([e4, e5], 64, bias, normalize, dropout)
        self.d02 = DecoderBlock2Plus([e1, 64, 64], 64, bias, normalize, dropout)         # 
        self.d12 = DecoderBlock2Plus([e2, 64, 64], 64, bias, normalize, dropout)
        self.d22 = DecoderBlock2Plus([e3, 64, 64], 64, bias, normalize, dropout)
        self.d03 = DecoderBlock2Plus([e1, 64, 64, 64], 64, bias, normalize, dropout)     # 
        self.d13 = DecoderBlock2Plus([e2, 64, 64, 64], 64, bias, normalize, dropout)
        self.d04 = DecoderBlock2Plus([e1, 64, 64, 64, 64], 64, bias, normalize, dropout) # 

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
        # Encoder forward pass
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

        # Apply CGM gating
        if self.cgm:
            probas = self.cgm_head(e5)
            fg = probas[:, 1].view(-1, 1, 1, 1)

        # Decoder forward
        sz = e1.shape[2:]

        d01 = self.d01([e1, e2], sz)
        d11 = self.d11([e2, e3], sz)
        d21 = self.d21([e3, e4], sz)
        d31 = self.d31([e4, e5], sz)
        d02 = self.d02([e1, d01, d11], sz)
        d12 = self.d12([e2, d11, d21], sz)
        d22 = self.d22([e3, d21, d31], sz)
        d03 = self.d03([e1, d01, d02, d12], sz)
        d13 = self.d13([e2, d11, d12, d22], sz)
        d04 = self.d04([e1, d01, d02, d03, d13], sz)

        if self.cgm:
            d04 = d04 * fg

        # Output layer forward pass
        out = self.out(d04)

        # Deep supervision outputs
        if self.deep_supervision and self.training:
            size = x.shape[2:]
            aux1 = F.interpolate(self.aux1(d01), size=size, mode='bilinear', align_corners=True)
            aux2 = F.interpolate(self.aux2(d02), size=size, mode='bilinear', align_corners=True)
            aux3 = F.interpolate(self.aux3(d03), size=size, mode='bilinear', align_corners=True)
            aux4 = F.interpolate(self.aux4(d04), size=size, mode='bilinear', align_corners=True)
            return out, aux1, aux2, aux3, aux4
        return out

