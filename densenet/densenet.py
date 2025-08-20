import math
import torch
from torch import nn

from utils import train_model_cifar10

DenseNetTypeArch = {
    121: [6, 12, 24, 16],
    169: [6, 12, 32, 32],
    201: [6, 12, 48, 32],
    264: [6, 12, 64, 48],
}


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # BN-ReLU-Conv
        inter_channels = out_channels * 4

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(num_features=inter_channels)
        self.conv2 = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x

        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.bn2(x))

        return torch.concat([input, x], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        super().__init__()

        self.layers = self._make_layers(num_layers, in_channels, growth_rate)

    def _make_layers(self, num_layers, in_channels, growth_rate) -> nn.Sequential:
        layers = []

        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.avgpool(x)

        return x


class DenseNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        classes: int,
        depth: int,
        growth_rate: int = 12,
        reduction_rate: float = 0.5,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * growth_rate,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.bn = nn.BatchNorm2d(num_features=2 * growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        try:
            num_layers = DenseNetTypeArch[depth]
        except KeyError:
            raise Exception("invalid arch type")
        self.in_channels = 2 * growth_rate

        self.blocks = self._make_blocks(
            num_layers, self.in_channels, growth_rate, reduction_rate
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=self.in_channels, out_features=classes)

    def _make_blocks(
        self,
        num_layers: list[int],
        in_channels: int,
        growth_rate: int,
        reduction_rate: float,
    ):
        blocks = []

        for i, x in enumerate(num_layers):
            out_channels = in_channels + x * growth_rate
            blocks.append(
                DenseBlock(
                    num_layers=x, in_channels=in_channels, growth_rate=growth_rate
                )
            )
            in_channels = out_channels

            if i != len(num_layers) - 1:
                out_channels = math.floor(in_channels * reduction_rate)
                blocks.append(
                    TransitionBlock(in_channels=in_channels, out_channels=out_channels)
                )
                in_channels = out_channels

        self.in_channels = in_channels

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.maxpool(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


train_model_cifar10(
    lambda: DenseNet(
        in_channels=3, classes=10, depth=121, growth_rate=12, reduction_rate=0.5
    ),
    5,
)
