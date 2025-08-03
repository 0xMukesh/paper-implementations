import torch
from torch import nn
from typing import List, Tuple, Union

from yolo.constants import NUM_BBOXES_PER_SPLIT, NUM_CLASSES, SPLIT_SIZE

ConvLayerConfig = Tuple[int, int, int, int]
RepeatBlockConfig = List[Union[ConvLayerConfig, int]]
ArchConfigItem = Union[ConvLayerConfig, str, RepeatBlockConfig]
ArchConfig = List[ArchConfigItem]

ARCH_CONFIG: ArchConfig = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, arch_config: ArchConfig, in_channels=3):
        super().__init__()

        self.in_channels = in_channels
        self.arch_config = arch_config

        self.split_size = SPLIT_SIZE
        self.num_bboxes = NUM_BBOXES_PER_SPLIT
        self.num_classes = NUM_CLASSES

        self.conv_layers = self._create_conv_layers()
        self.fc_layers = self._create_fc_layers()

    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        for cfg in self.arch_config:
            if isinstance(cfg, tuple) and all(isinstance(x, int) for x in cfg):
                # conv layer
                kernel_size, out_channels, stride, padding = cfg
                layers.append(
                    Conv2dBlock(in_channels, out_channels, kernel_size, stride, padding)
                )
                in_channels = out_channels
            elif isinstance(cfg, str) and cfg.lower() == "m":
                # max pool
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(cfg, list):
                # repeated block
                *conv_layers, num_repeat = cfg
                if not isinstance(num_repeat, int):
                    raise ValueError("invalid config")

                for _ in range(num_repeat):
                    for cfg in conv_layers:
                        if not isinstance(cfg, tuple) or len(cfg) != 4:
                            raise ValueError("invalid config")

                        kernel_size, out_channels, stride, padding = cfg
                        layers.append(
                            Conv2dBlock(
                                in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                            )
                        )
                        in_channels = out_channels

        return nn.Sequential(*layers)

    def _create_fc_layers(self):
        return nn.Sequential(
            nn.Linear(50176, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(
                4096,
                self.split_size
                * self.split_size
                * (self.num_classes + 5 * self.num_bboxes),
            ),
        )

    def forward(self, x) -> torch.Tensor:
        out = self.conv_layers(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc_layers(out)

        return out
