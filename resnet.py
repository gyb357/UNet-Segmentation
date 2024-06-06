import torch.nn as nn
from typing import Optional, Callable, Type, Union, List, Tuple
from utils import operate
from torch import Tensor
import torch


# Download link
# https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html

# resnet18 v1: https://download.pytorch.org/models/resnet18-f37072fd.pth
# resnet34 v1: https://download.pytorch.org/models/resnet34-b627a593.pth
# resnet50 v2: https://download.pytorch.org/models/resnet50-11ad3fa6.pth
# resnet101 v2: https://download.pytorch.org/models/resnet101-cd907fc2.pth
# resnet152 v2: https://download.pytorch.org/models/resnet152-f82ba261.pth


IMAGENET_1K_WEIGHTS = {
    'resnet18': 'model/pretrained/resnet18-f37072fd.pth',
    'resnet34': 'model/pretrained/resnet34-b627a593.pth',
    'resnet50': 'model/pretrained/resnet50-11ad3fa6.pth',
    'resnet101': 'model/pretrained/resnet101-cd907fc2.pth',
    'resnet152': 'model/pretrained/resnet152-f82ba261.pth'
}


def conv1x1(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = False
) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


def conv3x3(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bias: bool = False
) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)


def norm_layer(norm: Optional[Callable[..., nn.Module]]) -> nn.Module:
    return operate(norm is None, nn.BatchNorm2d, norm)


class Bottleneck2Conv(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            bias: bool = False,
            norm: Optional[Callable[..., nn.Module]] = None,
            downsample: Optional[nn.Module] = None
    ) -> None:
        super(Bottleneck2Conv, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride, bias)
        self.bn1 = norm_layer(norm)(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, bias=bias)
        self.bn2 = norm_layer(norm)(out_channels)
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
            norm: Optional[Callable[..., nn.Module]] = None,
            downsample: Optional[nn.Module] = None
    ) -> None:
        super(Bottleneck3Conv, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels, bias=bias)
        self.bn1 = norm_layer(norm)(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride, bias)
        self.bn2 = norm_layer(norm)(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels*self.expansion)
        self.bn3 = norm_layer(norm)(out_channels*self.expansion)
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
            norm: Optional[Callable[..., nn.Module]] = None,
            init_weights: bool = True
    ) -> None:
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=bias)
        self.bn1 = norm_layer(norm)(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.bottleneck_layer(bottleneck, self.filters[0], self.strides[0], bias, norm, layers[0])
        self.layer2 = self.bottleneck_layer(bottleneck, self.filters[1], self.strides[1], bias, norm, layers[1])
        self.layer3 = self.bottleneck_layer(bottleneck, self.filters[2], self.strides[2], bias, norm, layers[2])
        self.layer4 = self.bottleneck_layer(bottleneck, self.filters[3], self.strides[3], bias, norm, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.filters[3]*bottleneck.expansion, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def bottleneck_layer(
            self,
            bottleneck: Type[Union[Bottleneck2Conv, Bottleneck3Conv]],
            out_channels: int,
            stride: int,
            bias: bool,
            norm: Optional[Callable[..., nn.Module]],
            layer: int
    ) -> nn.Sequential:
        if stride != 1 or self.in_channels != out_channels*bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels*bottleneck.expansion, stride, bias),
                norm_layer(norm)(out_channels*bottleneck.expansion)
            )
        else: downsample = None

        layers = []
        layers.append(bottleneck(self.in_channels, out_channels, stride, bias, norm, downsample))

        self.in_channels = out_channels*bottleneck.expansion
        for _ in range(1, layer):
            layers.append(bottleneck(self.in_channels, out_channels, stride=1, bias=bias, norm=norm))
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


def resnet(
        name: str,
        channels: int = 3,
        num_classes: int = 1000,
        bias: bool = False,
        norm: Optional[Callable[..., nn.Module]] = None,
        init_weights: bool = True,
        pretrained: bool = False
) -> Tuple[ResNet, List[int]]:
    bottleneck, layers, filters = RESNET_CONFIGS[name]
    model = ResNet(bottleneck, layers, channels, num_classes, bias, norm, init_weights)

    if pretrained:
        model.load_state_dict(torch.load(IMAGENET_1K_WEIGHTS[name]))
    return model, filters

