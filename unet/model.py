import torch
from torch import nn
import torchvision.transforms.functional as VF
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
        features: List[int] = [
            64,
            128,
            256,
            512,
        ],
    ) -> None:
        super().__init__()

        self._downs = nn.ModuleList()
        self._ups = nn.ModuleList()

        self.in_channels = in_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self._downs.append(
                DoubleConv(in_channels=in_channels, out_channels=feature)
            )
            in_channels = feature

        for feature in reversed(features):
            self._ups.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )
            self._ups.append(
                nn.Conv2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=3,
                    padding=1,
                )
            )
            self._ups.append(DoubleConv(in_channels=feature * 2, out_channels=feature))

        self._bottleneck = DoubleConv(
            in_channels=features[-1], out_channels=features[-1] * 2
        )

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

        for i in range(0, len(skip_connections)):
            skip_connection = skip_connections[i]

            x = self._ups[3 * i](x)
            x = self._ups[3 * i + 1](x)

            if skip_connection.shape != x.shape:
                h, w = x.shape[-2:]
                skip_connection = VF.resize(skip_connection, [h, w])

            x = torch.cat((x, skip_connection), dim=1)
            x = self._ups[3 * i + 2](x)

        x = self._final_conv(x)

        return x


if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=1)
    x = torch.randn((1, 1, 572, 572))
    x = model.forward(x)
    print(x.shape)
