import torch
from torch import nn

from utils import train_model_cifar10


class ResNeXtBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        base_out_channels: int,
        cardinality: int = 32,
        width: int = 4,
        stride: int = 1,
    ):
        super().__init__()

        D = cardinality * width

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=D, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=D)

        self.conv2 = nn.Conv2d(
            in_channels=D,
            out_channels=D,
            kernel_size=3,
            stride=stride,
            groups=cardinality,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=D)

        self.conv3 = nn.Conv2d(
            in_channels=D,
            out_channels=base_out_channels * self.expansion,
            kernel_size=1,
        )
        self.bn3 = nn.BatchNorm2d(num_features=base_out_channels * self.expansion)

        if stride != 1 or in_channels != base_out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=base_out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(num_features=base_out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        identity = self.shortcut(identity)
        x += identity
        x = self.relu(x)

        return x


class ResNeXt50(nn.Module):
    def __init__(self, block, classes: int, in_channels: int = 3):
        super().__init__()

        self.block = block

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = self._make_blocks(
            [3, 4, 6, 3],
            base_in_channels=64,
            base_out_channels=64,
            base_group_width=4,
            cardinality=32,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=classes)

    def _make_blocks(
        self,
        arch: list[int],
        base_in_channels: int,
        base_out_channels: int,
        base_group_width: int,
        cardinality: int,
    ):
        layers = []
        in_channels = base_in_channels

        for i, x in enumerate(arch):
            blocks = []
            out_channels = base_out_channels * (2**i)
            width = base_group_width * (2**i)
            stride = 2 if i != 0 else 1

            blocks.append(
                self.block(
                    in_channels=in_channels,
                    base_out_channels=out_channels,
                    cardinality=cardinality,
                    width=width,
                    stride=stride,
                )
            )
            in_channels = out_channels * self.block.expansion

            for _ in range(1, x):
                blocks.append(
                    self.block(
                        in_channels=in_channels,
                        base_out_channels=out_channels,
                        cardinality=cardinality,
                        width=width,
                        stride=1,
                    )
                )

            layers.extend(blocks)

        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


train_model_cifar10(lambda: ResNeXt50(block=ResNeXtBlock, classes=10, in_channels=3), 3)
