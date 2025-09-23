from typing import Tuple
import torch
from torch import nn
from einops import rearrange


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        g_channels: int,
        out_channels: int,
        num_classes: int,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=z_dim)
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

    def forward(self, x: torch.Tensor, labels) -> torch.Tensor:
        embedding = self.embed(labels)
        embedding = rearrange(embedding, "b c -> b c 1 1")
        x = torch.cat([x, embedding], dim=1)

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
        in_channels = z_dim + z_dim

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


class Critic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_channels: int,
        num_classes: int,
        img_size: int | Tuple[int, int],
        num_blocks: int = 4,
    ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_height = img_size
            self.img_width = img_size
        else:
            self.img_height = img_size[0]
            self.img_width = img_size[1]

        self.embedding_dim = self.img_height * self.img_width

        self.embed = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=self.embedding_dim,
        )
        self.net = nn.Sequential(
            self._make_block_chain(in_channels, d_channels, num_blocks),
            nn.Conv2d(
                in_channels=d_channels * 2 ** (num_blocks - 1),
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor, labels) -> torch.Tensor:
        embedding = self.embed(labels)
        embedding = rearrange(
            embedding, "b (h w) -> b 1 h w", h=self.img_height, w=self.img_width
        )
        
        x = torch.cat([x, embedding], dim=1)

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
        in_channels = in_channels + 1

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


def test():
    z_dim = 50
    batch_size = 64
    img_size = (64, 64)
    num_classes = 10
    img_channels = 1
    num_features = 128

    img_h, img_w = img_size

    noise = torch.randn((batch_size, z_dim, 1, 1))
    img = torch.randn((batch_size, 1, img_h, img_w))
    labels = torch.randint(0, num_classes, (batch_size,))

    gen = Generator(
        z_dim=z_dim,
        g_channels=num_features,
        out_channels=img_channels,
        num_classes=num_classes,
    )
    critic = Critic(
        in_channels=img_channels,
        d_channels=num_features,
        img_size=img_size,
        num_classes=num_classes,
    )

    print(f"shape of G(noise) = {gen(noise, labels).shape}")
    print(f"shape of C(img) = {critic(img, labels).shape}")
    print(f"shape of C(G(noise)) = {critic(gen(noise, labels), labels).shape}")


if __name__ == "__main__":
    test()
