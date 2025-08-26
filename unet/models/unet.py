import torch
from torch import nn
import torchvision.transforms.functional as TF
from typing import List


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
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
        x = self.model(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features: List[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder
        self.encoder = nn.ModuleList()
        for up in features:
            self.encoder.append(DoubleConv(in_channels=in_channels, out_channels=up))
            in_channels = up

        # bottleneck
        self.bottleneck = DoubleConv(
            in_channels=features[-1], out_channels=features[-1] * 2
        )

        # decoder
        self.decoder = nn.ModuleList()
        for down in features[::-1]:
            self.decoder.append(
                nn.ConvTranspose2d(
                    in_channels=down * 2, out_channels=down, kernel_size=2, stride=2
                )
            )
            self.decoder.append(DoubleConv(in_channels=down * 2, out_channels=down))

        self.final_conv = nn.Conv2d(
            in_channels=features[0], out_channels=num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: List[torch.Tensor] = []

        # encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        for i, skip_connection in enumerate(skip_connections[::-1]):
            conv_transpose = self.decoder[2 * i]
            double_conv = self.decoder[2 * i + 1]

            # upscale image
            x = conv_transpose(x)

            # resize skip connection
            if skip_connection.shape != x.shape:
                h, w = x.shape[2:]
                skip_connection = TF.resize(skip_connection, [h, w])

            # combine skip connection + upscale image
            x = torch.cat((x, skip_connection), dim=1)

            # apply double conv
            x = double_conv(x)

        x = self.final_conv(x)

        return x
