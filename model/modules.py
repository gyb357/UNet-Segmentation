from . import *


class DoubleConv2d(nn.Module):
    """
    Double convolutional block with normalization and ReLU activation.
    
    Structure
    ---------
     | ↓ Conv2d (3x3)
     | ↓ Normalization
     | ↓ ReLU
     | ↓ Conv2d (3x3)
     | ↓ Normalization
     | ↓ ReLU
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bias (bool): Whether to use bias in convolutional layers
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
        """

        super(DoubleConv2d, self).__init__()
        self.normalize = normalize or nn.BatchNorm2d
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            self.normalize(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            self.normalize(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
    """
    Encoder block with double convolutional layer and max pooling.
    
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
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bias (bool): Whether to use bias in convolutional layers
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability
        """

        super(EncoderBlock, self).__init__()
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
    Decoder block with transposed convolutional layer and double convolutional layer.
    Concatenates the input from the encoder block.

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
            normalize: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0
    ) -> None:
        """
        Args:
            up_in_channels (int): Number of input channels for transposed convolutional layer
            up_out_channels (int): Number of output channels for transposed convolutional layer
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bias (bool): Whether to use bias in convolutional layers
            normalize (nn.Module): Normalization layer to use (default: `nn.BatchNorm2d`)
            dropout (float): Dropout probability
        """

        super(DecoderBlock, self).__init__()
        self.trans = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2, bias=bias)
        self.conv = DoubleConv2d(in_channels, out_channels, bias, normalize)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.trans(x1)

        assert x.shape[2:] == x2.shape[2:], f"Skip connection size {x2.shape[2:]} doesn't match upsampled size {x1.shape[2:]}."

        x = self.conv(torch.cat([x, x2], dim=1))
        x = self.drop(x)
        return x


class OutputBlock(nn.Module):
    """
    Output block with transposed convolutional layer and convolutional layer.
    If backbone is not provided, the block only contains a convolutional layer.
    
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
            backbone (str): Whether to use as a backbone
            bias (bool): Whether to use bias in convolutional layers
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

