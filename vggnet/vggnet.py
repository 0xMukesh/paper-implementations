import torch
from torch import nn

from utils import train_model_cifar10

VGG_TYPES = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGGNet(nn.Module):
    def __init__(self, arch: list[int | str], classes: int, in_channels: int = 3):
        super().__init__()

        self.in_channels = in_channels

        self.conv_layers = self._create_conv_layers(arch)
        self.fc1 = nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=classes)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def _create_conv_layers(self, arch: list[int | str]):
        in_channels = self.in_channels

        layers = []

        for x in arch:
            if isinstance(x, int):
                out_channels = x

                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                layers.append(nn.ReLU())

                in_channels = out_channels
            elif isinstance(x, str) and x == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise Exception("invalid arch type")

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, -1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x


train_model_cifar10(
    lambda: VGGNet(arch=VGG_TYPES["VGG19"], classes=10, in_channels=3), 5
)
