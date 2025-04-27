from . import *


class DoubleConv2d(nn.Module):
    """
    Double convolutional block with normalize and ReLU activation
    
    Structure
    ---------
     | ↓ Conv2d (3x3)
     | ↓ Normalize
     | ↓ ReLU
     | ↓ Conv2d (3x3)
     | ↓ Normalize
     | ↓ ReLU
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = False,
            normalize: Optional[Type[nn.Module]] = None
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
        """

        super(DoubleConv2d, self).__init__()
        normalize = normalize if normalize is not None else nn.BatchNorm2d
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            normalize(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            normalize(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
    """
    Encoder block with double convolutional layer and max pooling
    
    Structure
    ---------
     | ↓ DoubleConv2d
     | ↓ MaxPool2d (2x2)
     | ↓ Dropout
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = False,
            normalize: Optional[Type[nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability (default: `0.0`)
        """

        super(EncoderBlock, self).__init__()
        normalize = normalize if normalize is not None else nn.BatchNorm2d
        self.conv = DoubleConv2d(in_channels, out_channels, bias, normalize)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv(x)
        p = self.pool(x)
        p = self.drop(p)
        return x, p


class DecoderBlock(nn.Module):
    """
    Decoder block with transposed convolutional layer and double convolutional layer
    Concatenates the input from the encoder block

    Structure
    ---------
     | ↓ ConvTranspose2d (2x2)
     | ↓ DoubleConv2d
     | ↓ Dropout
    """

    def __init__(
            self,
            up_in_channels: int,
            up_out_channels: int,
            in_channels: int,
            out_channels: int,
            bias: bool = False,
            normalize: Optional[Type[nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        """
        Args:
            up_in_channels (int): Number of input channels for transposed convolutional layer
            up_out_channels (int): Number of output channels for transposed convolutional layer
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability (default: `0.0`)
        """

        super(DecoderBlock, self).__init__()
        normalize = normalize if normalize is not None else nn.BatchNorm2d
        self.trans = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2, bias=bias)
        self.conv = DoubleConv2d(in_channels, out_channels, bias, normalize)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.trans(x1)

        if x.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = self.conv(torch.cat([x, x2], dim=1))
        x = self.drop(x)
        return x


class DecoderBlockPlus(nn.Module):
    """
    Decoder block with UNet2+ and UNet3+ source features

    Structure
    ---------
     | ↓ Conv2d (1x1)
     | ↓ DoubleConv2d
     | ↓ Dropout
    """

    def __init__(
            self,
            in_channels_list: Tuple[int, ...],
            mid_channels: int,
            bias: bool = False,
            normalize: Optional[Type[nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        """
        Args:
            in_channels_list (tuple): List of input channels for each source feature
            mid_channels (int): Number of middle channels
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability (default: `0.0`)
        """

        super(DecoderBlockPlus, self).__init__()
        normalize = normalize if normalize is not None else nn.BatchNorm2d
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=bias)
            for in_channels in in_channels_list
        ])
        self.fusion = DoubleConv2d(mid_channels * len(in_channels_list), mid_channels, bias, normalize)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rescale(x: Tensor, size: Tuple[int, int]) -> Tensor:
        h, w = x.shape[2:]
        th, tw = size

        # Downsample with adaptive max pooling if larger
        if h > th and w > tw:
            return F.adaptive_max_pool2d(x, (th, tw))
        # Upsample with interpolation if smaller
        if h < th or w < tw:
            return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x
    
    def forward(self, x: Tuple[Tensor, ...], size: Tuple[int, int]) -> Tensor:
        aligned = [conv(self._rescale(f, size)) for conv, f in zip(self.conv, x)]
        fused = self.fusion(torch.cat(aligned, dim=1))
        fused = self.dropout(fused)
        return fused


class CGMHead(nn.Module):
    """
    Classification-Guided Module (CGM) head for classification tasks
    Applies a 1x1 convolution followed by adaptive average pooling and softmax activation

    Structure
    ---------
     | ↓ Conv2d (1x1)
     | ↓ AdaptiveAvgPool2d (1x1)
     | ↓ Flatten
     | ↓ Softmax
    """

    def __init__(
            self,
            in_channels: int,
            bias: bool = False
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
        """

        super(CGMHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=bias),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class OutputBlock(nn.Module):
    """
    Output block with transposed convolutional layer and convolutional layer
    If backbone is not provided, the block only contains a convolutional layer
    
    Structure
    ---------
     | ↓ ConvTranspose2d (2x2)
     | ↓ Conv2d (1x1)
    (with backbone)

     | ↓ Conv2d (1x1)
    (without backbone)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_classes: int,
            backbone: Optional[str] = None,
            bias: bool = False
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_classes (int): Number of output classes
            backbone (str): Whether to use as a backbone (default: `None`)
            bias (bool): Whether to use bias in convolutional layers (default: `False`)
        """

        super(OutputBlock, self).__init__()
        if backbone:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=bias)
            )
        else: self.layers = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

