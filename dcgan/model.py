import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self, z_dim: int, g_channels: int, out_channels: int, num_blocks: int = 4
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            self._make_block_chain(z_dim, g_channels, num_blocks),
            nn.ConvTranspose2d(
                in_channels=g_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        return net

    def _make_block_chain(
        self, z_dim: int, g_channels: int, num_blocks: int
    ) -> nn.Sequential:
        net = nn.Sequential()
        in_channels = z_dim

        for i in range(num_blocks):
            multiplier = 2 ** (num_blocks - i - 1)
            stride = 2 if i != 0 else 1
            padding = 1 if i != 0 else 0

            net.add_module(
                f"block_{i}",
                self._block(in_channels, g_channels * multiplier, 4, stride, padding),
            )

            in_channels = g_channels * multiplier

        return net


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, d_channels: int, num_blocks: int = 4) -> None:
        super().__init__()

        self.net = nn.Sequential(
            self._make_block_chain(in_channels, d_channels, num_blocks),
            nn.Conv2d(
                in_channels=d_channels * 2 ** (num_blocks - 1),
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_batchnorm: bool = True,
        relu_slope: float = 0.2,
    ):
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        )

        if use_batchnorm:
            net.add_module("batchnorm", nn.BatchNorm2d(out_channels))

        net.add_module("leakyrelu", nn.LeakyReLU(relu_slope, inplace=True))

        return net

    def _make_block_chain(self, in_channels: int, d_channels: int, num_blocks: int):
        net = nn.Sequential()

        for i in range(num_blocks):
            multiplier = 2**i
            stride = 2 if i != num_blocks - 1 else 1
            padding = 1 if i != num_blocks - 1 else 0

            net.add_module(
                f"block_{i}",
                self._block(
                    in_channels,
                    d_channels * multiplier,
                    4,
                    stride,
                    padding,
                    i != 0,
                ),
            )

            in_channels = d_channels * multiplier

        return net
