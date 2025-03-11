from typing import Optional, Type, Union, List
from torch import Tensor
import torch.nn as nn
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


class Bottleneck2Conv(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            bias: bool = False,
            downsample: Optional[nn.Module] = None
    ) -> None:
        super(Bottleneck2Conv, self).__init__()
        self.conv1 = conv3x3_layer(in_channels, out_channels, stride, bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3_layer(out_channels, out_channels, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
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
            downsample: Optional[nn.Module] = None
    ) -> None:
        super(Bottleneck3Conv, self).__init__()
        self.conv1 = conv1x1_layer(in_channels, out_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3_layer(out_channels, out_channels, stride, bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1_layer(out_channels, out_channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
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
            init_weights: bool = False,
            zero_init_residual: bool = False
    ) -> None:
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=bias)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.bottleneck_layer(bottleneck, self.filters[0], self.strides[0], bias, layers[0])
        self.layer2 = self.bottleneck_layer(bottleneck, self.filters[1], self.strides[1], bias, layers[1])
        self.layer3 = self.bottleneck_layer(bottleneck, self.filters[2], self.strides[2], bias, layers[2])
        self.layer4 = self.bottleneck_layer(bottleneck, self.filters[3], self.strides[3], bias, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.filters[3]*bottleneck.expansion, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3Conv):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Bottleneck2Conv):
                    nn.init.constant_(m.bn2.weight, 0)

    def bottleneck_layer(
            self,
            bottleneck: Type[Union[Bottleneck2Conv, Bottleneck3Conv]],
            out_channels: int,
            stride: int,
            bias: bool,
            layer: int
    ) -> nn.Sequential:
        expansion = bottleneck.expansion
        
        if stride != 1 or out_channels*expansion != self.in_channels:
            downsample = nn.Sequential(
                conv1x1_layer(self.in_channels, out_channels*expansion, stride, bias),
                nn.BatchNorm2d(out_channels*expansion)
            )
        else: downsample = None

        layers = []
        layers.append(bottleneck(self.in_channels, out_channels, stride, bias, downsample))

        self.in_channels = out_channels*expansion
        for _ in range(1, layer):
            layers.append(bottleneck(self.in_channels, out_channels, stride=1, bias=bias))
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


RESNET_CONFIGS = {
    'resnet18': (Bottleneck2Conv, [2, 2, 2, 2]),
    'resnet34': (Bottleneck2Conv, [3, 4, 6, 3]),
    'resnet50': (Bottleneck3Conv, [3, 4, 6, 3]),
    'resnet101': (Bottleneck3Conv, [3, 4, 23, 3]),
    'resnet152': (Bottleneck3Conv, [3, 8, 36, 3])
}


def resnet(
        name: str = 'resnet18',
        channels: int = 3,
        num_classes: int = 1000,
        bias: bool = False,
        init_weights: bool = False,
        zero_init_residual: bool = False,
        pretrained: Optional[str] = None
) -> ResNet:
    if name not in RESNET_CONFIGS:
        raise ValueError(f'Invalid ResNet name {name}. Available options are {list(RESNET_CONFIGS.keys())}.')
    
    bottleneck, layers = RESNET_CONFIGS[name]
    resnet = ResNet(bottleneck, layers, channels, num_classes, bias, init_weights, zero_init_residual)

    if pretrained:
        try:
            resnet.load_state_dict(torch.load(pretrained))
        except Exception as e:
            raise RuntimeError(f'Failed to load pretrained weights: {e}')
    return resnet

