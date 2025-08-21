import torch
from torch import nn
import torchvision.transforms.functional as TF
from typing import List


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features: List[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()

        self._downs = nn.ModuleList()
        self._ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self._downs.append(
                DoubleConv(in_channels=in_channels, out_channels=feature)
            )
            in_channels = feature

        self._bottleneck = DoubleConv(
            in_channels=features[-1], out_channels=features[-1] * 2
        )

        for feature in reversed(features):
            self._ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self._ups.append(DoubleConv(in_channels=feature * 2, out_channels=feature))

        self._final_conv = nn.Conv2d(
            in_channels=features[0], out_channels=num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: List[torch.Tensor] = []

        for down in self._downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self._bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self._ups), 2):
            x = self._ups[i](x)
            skip_connection = skip_connections[i // 2]

            if skip_connection.shape != x.shape:
                h, w = skip_connection.shape[2:]
                x = TF.resize(x, [h, w])

            x = torch.cat((skip_connection, x), dim=1)
            x = self._ups[i + 1](x)

        return self._final_conv(x)
