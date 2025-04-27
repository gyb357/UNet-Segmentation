from . import *


# Configuration of the decoder layers
_UNET_CONFIGS = {
    'shallow': {'conv_filters': [768, 384, 192, 128, 64], 'trans_filters': [512, 512, 256, 128, 64]},
    'deep':    {'conv_filters': [1536, 768, 384, 128, 64], 'trans_filters': [2048, 512, 256, 128, 64]},
    'default': {'conv_filters': [1024, 512, 256, 128, 64], 'trans_filters': [1024, 512, 256, 128, 64]}
}
_UNET_PLUS_CONFIGS = {
    'shallow': (64, 64, 128, 256, 512),    # resnet 18, 34
    'deep':    (64, 256, 512, 1024, 2048), # resnet 50, 101, 152
    'default': (64, 128, 256, 512, 1024)
}


class UNet(nn.Module):
    """
    UNet implementation for image segmentation
    Supports ResNet backbone or standard UNet encoder,
    optional deep supervision and classification-guided module (CGM)
    
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
            normalize: Type[nn.Module] = nn.BatchNorm2d,
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
            pretrained (str): Pretrained model path (./model/pretrained/...)
            freeze_backbone (bool): Whether to freeze backbone weights (default: `False`)
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability (default: `0.0`)
            init_weights (bool): Whether to initialize weights (default: `False`)
            deep_supervision (bool): Whether to use deep supervision (default: `False`)
            cgm (bool): Whether to use CGM(Classification-Guided Module) (default: `False`)
        """

        super(UNet, self).__init__()
        # Attributes
        self.backbone = backbone
        self.deep_supervision = deep_supervision
        self.cgm = cgm

        # Determine depth
        self.backbone_depth = ternary_operation_elif(
            backbone in ['resnet18', 'resnet34'], 'shallow',
            backbone in ['resnet50', 'resnet101', 'resnet152'], 'deep',
            'default'
        )
        config = _UNET_CONFIGS[self.backbone_depth]
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

        # Decoder layers
        self.d1 = DecoderBlock(self.decoder_trans_filters[0], 512, self.decoder_conv_filters[0], 512, bias, normalize, dropout)
        self.d2 = DecoderBlock(self.decoder_trans_filters[1], 256, self.decoder_conv_filters[1], 256, bias, normalize, dropout)
        self.d3 = DecoderBlock(self.decoder_trans_filters[2], 128, self.decoder_conv_filters[2], 128, bias, normalize, dropout)
        self.d4 = DecoderBlock(self.decoder_trans_filters[3], 64, self.decoder_conv_filters[3], 64, bias, normalize, dropout)
        
        # Output layer
        self.out = OutputBlock(self.decoder_conv_filters[4], 32, num_classes, backbone)

        # Deep supervision
        if deep_supervision:
            self.aux_convs = nn.ModuleList([
                nn.Conv2d(self.decoder_conv_filters[i], num_classes, kernel_size=1, bias=bias)
                for i in range(4)
            ])

        # Classification-Guided Module head
        if cgm:
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

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
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

        # Decoder
        d1 = self.d1(e5, e4)
        d2 = self.d2(d1, e3)
        d3 = self.d3(d2, e2)
        d4 = self.d4(d3, e1)

        # CGM gating
        if self.cgm:
            probas = self.cgm_head(e5)
            fg = probas[:, 1].view(-1, 1, 1, 1)
            d4 = d4 * fg

        # Output
        out = self.out(d4)

        # Deep supervision
        if self.deep_supervision and self.training:
            size = x.shape[2:]
            auxs = []
            for i, feat in enumerate([d1, d2, d3, d4]):
                aux = F.interpolate(
                    self.aux_convs[i](feat),
                    size=size, mode='bilinear', align_corners=False
                )
                auxs.append(aux)
            return (out, *auxs)
        return out


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
            normalize: Type[nn.Module] = nn.BatchNorm2d,
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
            pretrained (str): Pretrained model path (./model/pretrained/...)
            freeze_backbone (bool): Whether to freeze backbone weights (default: `False`)
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability (default: `0.0`)
            init_weights (bool): Whether to initialize weights (default: `False`)
            deep_supervision (bool): Whether to use deep supervision (default: `False`)
            cgm (bool): Whether to use CGM(Classification-Guided Module) (default: `False`)
        """

        super(UNet, self).__init__()
        # Attributes
        self.backbone = backbone
        self.deep_supervision = deep_supervision
        self.cgm = cgm

        # Determine depth
        self.backbone_depth = ternary_operation_elif(
            backbone in ['resnet18', 'resnet34'], 'shallow',
            backbone in ['resnet50', 'resnet101', 'resnet152'], 'deep',
            'default'
        )
        e1, e2, e3, e4, e5 = _UNET_PLUS_CONFIGS[self.backbone_depth]

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

        # Decoder layers
        self.d01 = DecoderBlockPlus((e1, e2), 64, bias, normalize, dropout)
        self.d11 = DecoderBlockPlus((e2, e3), 64, bias, normalize, dropout)
        self.d21 = DecoderBlockPlus((e3, e4), 64, bias, normalize, dropout)
        self.d31 = DecoderBlockPlus((e4, e5), 64, bias, normalize, dropout)
        self.d02 = DecoderBlockPlus((e1, 64, 64), 64, bias, normalize, dropout)
        self.d12 = DecoderBlockPlus((e2, 64, 64), 64, bias, normalize, dropout)
        self.d22 = DecoderBlockPlus((e3, 64, 64), 64, bias, normalize, dropout)
        self.d03 = DecoderBlockPlus((e1, 64, 64, 64), 64, bias, normalize, dropout)
        self.d13 = DecoderBlockPlus((e2, 64, 64, 64), 64, bias, normalize, dropout)
        self.d04 = DecoderBlockPlus((e1, 64, 64, 64, 64), 64, bias, normalize, dropout)

        # Output layer
        self.out = OutputBlock(64, 64, num_classes, backbone, bias)

        # Deep supervision
        if deep_supervision:
            self.aux1 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux2 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux3 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux4 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)

        # Classification-Guided Module head
        if cgm:
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

        # CGM gating
        if self.cgm:
            probas = self.cgm_head(e5)
            fg = probas[:, 1].view(-1, 1, 1, 1)

        # Compute target sizes per level
        s1 = e1.shape[2:]
        s2 = e2.shape[2:]
        s3 = e3.shape[2:]
        s4 = e4.shape[2:]

        # Nested decoding
        d01 = self.d01((e1, e2), s1)
        d11 = self.d11((e2, e3), s2)
        d21 = self.d21((e3, e4), s3)
        d31 = self.d31((e4, e5), s4)
        d02 = self.d02((e1, d01, d11), s1)
        d12 = self.d12((e2, d11, d21), s2)
        d22 = self.d22((e3, d21, d31), s3)
        d03 = self.d03((e1, d01, d02, d12), s1)
        d13 = self.d13((e2, d11, d12, d22), s2)
        d04 = self.d04((e1, d01, d02, d03, d13), s1)

        # CGM gating
        if self.cgm:
            d04 = d04 * fg

        # Output
        out = self.out(d04)

        # Deep supervision
        if self.deep_supervision and self.training:
            size = x.shape[2:]
            aux1 = F.interpolate(self.aux1(d01), size=size, mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux2(d02), size=size, mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux3(d03), size=size, mode='bilinear', align_corners=False)
            aux4 = F.interpolate(self.aux4(d04), size=size, mode='bilinear', align_corners=False)
            return out, aux1, aux2, aux3, aux4
        return out


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
            normalize: Type[nn.Module] = nn.BatchNorm2d,
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
            pretrained (str): Pretrained model path (./model/pretrained/...)
            freeze_backbone (bool): Whether to freeze backbone weights (default: `False`)
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability (default: `0.0`)
            init_weights (bool): Whether to initialize weights (default: `False`)
            deep_supervision (bool): Whether to use deep supervision (default: `False`)
            cgm (bool): Whether to use CGM(Classification-Guided Module) (default: `False`)
        """

        super(UNet3Plus, self).__init__()
        # Attributes
        self.backbone = backbone
        self.deep_supervision = deep_supervision
        self.cgm = cgm

        # Determine depth
        self.backbone_depth = ternary_operation_elif(
            backbone in ['resnet18', 'resnet34'], 'shallow',
            backbone in ['resnet50', 'resnet101', 'resnet152'], 'deep',
            'default'
        )
        e1, e2, e3, e4, e5 = _UNET_PLUS_CONFIGS[self.backbone_depth]

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

        # Decoder layers
        self.d1 = DecoderBlockPlus((e1, e2, e3, e4, e5), 64, bias, normalize, dropout)
        self.d2 = DecoderBlockPlus((e1, e2, e3, 64, e5), 64, bias, normalize, dropout)
        self.d3 = DecoderBlockPlus((e1, e2, 64, 64, e5), 64, bias, normalize, dropout)
        self.d4 = DecoderBlockPlus((e1, 64, 64, 64, e5), 64, bias, normalize, dropout)

        # Output layer
        self.out = OutputBlock(64, 64, num_classes, backbone, bias)

        # Deep supervision
        if deep_supervision:
            self.aux1 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux2 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux3 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)
            self.aux4 = nn.Conv2d(64, num_classes, kernel_size=1, bias=bias)

        # Classification-Guided Module head
        if cgm:
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

        # CGM gating
        if self.cgm:
            probas = self.cgm_head(e5)
            fg = probas[:, 1].view(-1, 1, 1, 1)

        # Compute target sizes per level
        s1 = e1.shape[2:]
        s2 = e2.shape[2:]
        s3 = e3.shape[2:]
        s4 = e4.shape[2:]

        # Nested decoding
        d1 = self.d1((e1, e2, e3, e4, e5), s4)
        d2 = self.d2((e1, e2, e3, d1, e5), s3)
        d3 = self.d3((e1, e2, d2, d1, e5), s2)
        d4 = self.d4((e1, d3, d2, d1, e5), s1)

        # CGM gating
        if self.cgm:
            d4 = d4 * fg

        # Output
        out = self.out(d4)

        # Deep supervision
        if self.deep_supervision and self.training:
            size = x.shape[2:]
            aux1 = F.interpolate(self.aux1(d1), size=size, mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux2(d2), size=size, mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux3(d3), size=size, mode='bilinear', align_corners=False)
            aux4 = F.interpolate(self.aux4(d4), size=size, mode='bilinear', align_corners=False)
            return out, aux1, aux2, aux3, aux4
        return out


class EnsembleUNet(nn.Module):
    """
    Ensemble wrapper for segmentation models.
    Supports averaging logits or probabilities from multiple sub-models.
    """

    def __init__(
            self,
            models: List[nn.Module],
            channels: int,
            num_classes: int,
            backbone: Optional[str] = None,
            pretrained: Optional[str] = None,
            freeze_backbone: bool = False,
            bias: bool = False,
            normalize: Type[nn.Module] = nn.BatchNorm2d,
            dropout: float = 0.0,
            init_weights: bool = False,
            deep_supervision: bool = False,
            cgm: bool = False
    ) -> None:
        """
        Args:
            models (List[nn.Module]): List of sub-models
            channels (int): Number of input channels
            num_classes (int): Number of output classes
            backbone (str): Backbone architecture for encoder ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained (str): Pretrained model path (./model/pretrained/...)
            freeze_backbone (bool): Whether to freeze backbone weights (default: `False`)
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability (default: `0.0`)
            init_weights (bool): Whether to initialize weights (default: `False`)
            deep_supervision (bool): Whether to use deep supervision (default: `False`)
            cgm (bool): Whether to use CGM(Classification-Guided Module) (default: `False`)
        """

        super(EnsembleUNet, self).__init__()
        model_list = get_model_list(models)

        # Create a list of sub-models
        self.sub_models = nn.ModuleList([
            model(
                channels,
                num_classes,
                backbone,
                pretrained,
                freeze_backbone,
                bias,
                normalize,
                dropout,
                init_weights,
                deep_supervision,
                cgm
            )
            for model in model_list
        ])

    def _get_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = [m(x) for m in self.sub_models]
        first = outputs[0]

        if isinstance(first, (list, tuple)):
            level_outputs = list(zip(*outputs))
            avg_levels = [
                torch.stack(level, dim=0).mean(dim=0)
                for level in level_outputs
            ]
            return avg_levels
        return torch.stack(outputs, dim=0).mean(dim=0)

