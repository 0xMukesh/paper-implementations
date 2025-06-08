import torch
from torch import nn

from utils import train_model_cifar10

class NiN(nn.Module):
    def __init__(self, classes: int, in_channels: int = 3):
        super().__init__()

        self.in_channels = in_channels

        self.block1 = self._make_block(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.block2 = self._make_block(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.block3 = self._make_block(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.block4 = self._make_block(in_channels=384, out_channels=classes, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        x = self.avgpool(x)

        return torch.flatten(x, 1)

train_model_cifar10(lambda: NiN(classes=10, in_channels=3), 5)
