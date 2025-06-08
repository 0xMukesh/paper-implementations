import torch
from torch import nn

ResNetArch = {
    "18": [2, 2, 2, 2],
    "34": [3, 4, 6, 3],
    "50": [3, 4, 6, 3],
    "101": [3, 4, 23, 3],
    "152": [3, 8, 36, 3]
}

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_conv1x1: bool = False, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(num_features=out_channels)
        ) if use_conv1x1 else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        identity = self.shortcut(identity)

        x += identity
        x = self.relu(x)

        return x

class BottleneckBlock(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=stride),
            nn.BatchNorm2d(num_features=out_channels * self.expansion)
        ) if in_channels != out_channels * self.expansion else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        identity = self.shortcut(identity)

        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, type: int, in_channels: int = 3):
        super().__init__()

        self.conv_in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_layers = self._make_layers(type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        final_channels = 512 * BottleneckBlock.expansion if type >= 50 else 512
        self.fc = nn.Linear(in_features=final_channels, out_features=1000)
    
        self.relu = nn.ReLU(inplace=True)

    def _make_layers(self, type: int):
        arch = ResNetArch[str(type)]
        use_bottleneck = type >= 50

        layers = []
        in_channels = self.conv_in_channels

        for i, num_blocks in enumerate(arch):
            stride = 1 if i == 0 else 2
            blocks = []

            out_channels = 64 * (2 ** i) 

            for j in range(num_blocks):
                if use_bottleneck:
                    blocks.append(BottleneckBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride if j == 0 else 1
                    ))
                    in_channels = out_channels * BottleneckBlock.expansion
                else:
                    blocks.append(ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        use_conv1x1=(in_channels != out_channels or stride != 1),
                        stride=stride if j == 0 else 1
                    ))
                    in_channels = out_channels

            layers.append(nn.Sequential(*blocks))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
