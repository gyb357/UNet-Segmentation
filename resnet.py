import torch.nn as nn
from typing import Optional, Callable, List, Type, Union, Tuple
from utils import operate
from torch import Tensor
import torch


def conv1x1_layer(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = False
) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def conv3x3_layer(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = False
) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)


def normalize_layer(normalize: Optional[Callable[..., nn.Module]] = None) -> nn.Module:
    return operate(normalize is None, nn.BatchNorm2d, normalize)


class Bottleneck2Conv(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            downsample: Optional[nn.Module] = None
    ) -> None:
        super(Bottleneck2Conv, self).__init__()
        self.conv1 = conv3x3_layer(in_channels, out_channels, stride, bias)
        self.bn1 = normalize_layer(normalize)(out_channels)
        self.conv2 = conv3x3_layer(out_channels, out_channels, bias=bias)
        self.bn2 = normalize_layer(normalize)(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            out += self.downsample(x)

        out = self.relu(out)
        return out


class Bottleneck3Conv(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            downsample: Optional[nn.Module] = None
    ) -> None:
        super(Bottleneck3Conv, self).__init__()
        self.conv1 = conv1x1_layer(in_channels, out_channels, bias=bias)
        self.bn1 = normalize_layer(normalize)(out_channels)
        self.conv2 = conv3x3_layer(out_channels, out_channels, stride, bias)
        self.bn2 = normalize_layer(normalize)(out_channels)
        self.conv3 = conv1x1_layer(out_channels, out_channels*self.expansion)
        self.bn3 = normalize_layer(normalize)(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            out += self.downsample(x)

        out = self.relu(out)
        return out


class ResNet(nn.Module):
    in_channels: int = 64
    filters: List[int] = [64, 128, 256, 512]
    strides: List[int] = [1, 2, 2, 2]

    def __init__(
            self,
            bottleneck: Type[Union[Bottleneck2Conv, Bottleneck3Conv]],
            layers: List[int],
            channels: int = 3,
            num_classes: int = 1000,
            bias: bool = False,
            normalize: Optional[Callable[..., nn.Module]] = None,
            init_weights: bool = False,
            zero_init_residual: bool = False
    ) -> None:
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=bias)
        self.bn1 = normalize_layer(normalize)(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.bottleneck_layer(bottleneck, self.filters[0], self.strides[0], bias, normalize, layers[0])
        self.layer2 = self.bottleneck_layer(bottleneck, self.filters[1], self.strides[1], bias, normalize, layers[1])
        self.layer3 = self.bottleneck_layer(bottleneck, self.filters[2], self.strides[2], bias, normalize, layers[2])
        self.layer4 = self.bottleneck_layer(bottleneck, self.filters[3], self.strides[3], bias, normalize, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.filters[3]*bottleneck.expansion, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3Conv) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Bottleneck2Conv) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def bottleneck_layer(
            self,
            bottleneck: Type[Union[Bottleneck2Conv, Bottleneck3Conv]],
            out_channels: int,
            stride: int,
            bias: bool,
            normalize: Optional[Callable[..., nn.Module]],
            layer: int
    ) -> nn.Sequential:
        expansion = bottleneck.expansion
        
        if stride != 1 or self.in_channels != out_channels*expansion:
            downsample = nn.Sequential(
                conv1x1_layer(self.in_channels, out_channels*expansion, stride, bias),
                normalize_layer(normalize)(out_channels*expansion)
            )
        else: downsample = None

        layers = []
        layers.append(bottleneck(self.in_channels, out_channels, stride, bias, normalize, downsample))

        self.in_channels = out_channels*expansion
        for _ in range(1, layer):
            layers.append(bottleneck(self.in_channels, out_channels, stride=1, bias=bias, normalize=normalize))
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


RESNET_CONFIGS = {
    'resnet18': (Bottleneck2Conv, [2, 2, 2, 2], [64, 128, 256, 512, 1024]),
    'resnet34': (Bottleneck2Conv, [3, 4, 6, 3], [64, 128, 256, 512, 1024]),
    'resnet50': (Bottleneck3Conv, [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
    'resnet101': (Bottleneck3Conv, [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
    'resnet152': (Bottleneck3Conv, [3, 8, 36, 3], [64, 256, 512, 1024, 2048])
}

IMAGENET_1K_WEIGHTS = {
    'resnet18': 'model/pretrained/resnet18-f37072fd.pth',
    'resnet34': 'model/pretrained/resnet34-b627a593.pth',
    'resnet50': 'model/pretrained/resnet50-11ad3fa6.pth',
    'resnet101': 'model/pretrained/resnet101-cd907fc2.pth',
    'resnet152': 'model/pretrained/resnet152-f82ba261.pth'
}


def resnet(
        name: str,
        channels: int = 3,
        num_classes: int = 1000,
        bias: bool = False,
        normalize: Optional[Callable[..., nn.Module]] = None,
        init_weights: bool = False,
        zero_init_residual: bool = False,
        pretrained: bool = False
) -> Tuple[ResNet, List[int]]:
    bottleneck, layers, filters = RESNET_CONFIGS[name]
    resnet = ResNet(bottleneck, layers, channels, num_classes, bias, normalize, init_weights, zero_init_residual)

    if pretrained:
        resnet.load_state_dict(torch.load(IMAGENET_1K_WEIGHTS[name]))
    return resnet, filters

