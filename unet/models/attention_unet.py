import torch
from torch import nn
import torch.nn.functional as F
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
        return self.model(x)


class AttentionGate(nn.Module):
    def __init__(
        self,
        g_channels: int,
        x_channels: int,
    ) -> None:
        super().__init__()

        self.wg = nn.Sequential(
            nn.Conv2d(
                in_channels=g_channels,
                out_channels=x_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=x_channels),
        )

        self.wx = nn.Sequential(
            nn.Conv2d(
                in_channels=x_channels,
                out_channels=x_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=x_channels),
        )

        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = self.wg(g)
        x1 = self.wx(x)

        if g.shape[:2] != x.shape[:2] or g.shape[:3] != x.shape[:3]:
            g = F.interpolate(
                g, (x.shape[:2], x.shape[:3]), mode="bilinear", align_corners=False
            )

        out = self.relu(g + x1)
        out = self.phi(out)
        out = x * out

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.double_conv = DoubleConv(
            in_channels=in_channels, out_channels=out_channels
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skip = self.double_conv(x)
        out = self.pool(skip)

        return [out, skip]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int) -> None:
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )
        self.att_gate = AttentionGate(g_channels=out_channels, x_channels=skip_channels)
        self.double_conv = DoubleConv(
            in_channels=out_channels + skip_channels, out_channels=out_channels
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        skip = self.att_gate(x, skip)
        out = torch.cat([x, skip], dim=1)
        out = self.double_conv(out)

        return out


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.features = [64, 128, 256, 512]
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for feature in self.features:
            self.encoders.append(
                EncoderBlock(in_channels=in_channels, out_channels=feature)
            )
            in_channels = feature

        for feature in reversed(self.features):
            self.decoders.append(
                DecoderBlock(
                    in_channels=feature * 2, out_channels=feature, skip_channels=feature
                )
            )

        self.bottleneck = DoubleConv(
            in_channels=self.features[-1], out_channels=self.features[-1] * 2
        )

        self.final_conv = nn.Conv2d(
            in_channels=self.features[0], out_channels=num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i + 1)]
            x = decoder(x, skip)

        return self.final_conv(x)
