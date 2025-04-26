from . import *
from unet import UNet
from unet2plus import UNet2Plus
from unet3plus import UNet3Plus


class EnsembleUNet(nn.Module):
    """
    Ensemble wrapper for segmentation models.
    Supports averaging logits or probabilities from multiple sub-models.
    """

    def __init__(
            self,
            model_names: List[Union[UNet, UNet2Plus, UNet3Plus]],
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
            model_names (list): List of model names
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

        super(EnsembleUNet, self).__init__()
        # Attributes
        self.sub_models = nn.ModuleList()

        if model_names not in [UNet, UNet2Plus, UNet3Plus]:
            raise ValueError(f"Invalid model name: {model_names}")
        
        self.sub_models = nn.ModuleList([
            model(
                channels=channels,
                num_classes=num_classes,
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                bias=bias,
                normalize=normalize,
                dropout=dropout,
                init_weights=init_weights,
                deep_supervision=deep_supervision,
                cgm=cgm
            )
            for model in model_names
        ])

    def _get_parameters(self) -> int:
        return sum(p.numel() for m in self.sub_models for p in m.parameters())
            
    def forward(self, x):
        # Collect logits from each model
        outputs = [m(x) for m in self.sub_models]
        # If deep supervision, only take main output
        if isinstance(outputs[0], tuple):
            outputs = [o[0] for o in outputs]
        # Stack and average
        stacked = torch.stack(outputs, dim=0)
        return torch.mean(stacked, dim=0)

