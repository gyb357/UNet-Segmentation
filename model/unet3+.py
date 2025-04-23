from . import *


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
            init_weights: bool = False
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

        super(UNet3Plus, self).__init__()
        # Attributes
        self.backbone = backbone

        # Encoder layers
        self.e1 = EncoderBlock(channels, 64, bias, normalize, dropout)
        self.e2 = EncoderBlock(64, 128, bias, normalize, dropout)
        self.e3 = EncoderBlock(128, 256, bias, normalize, dropout)
        self.e4 = EncoderBlock(256, 512, bias, normalize, dropout)

        # Center layer
        self.e5 = DoubleConv2d(512, 1024, bias, normalize)

        # Decoder layers
        self.d1 = DecoderBlock
