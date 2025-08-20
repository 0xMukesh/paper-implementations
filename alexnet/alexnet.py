import torch
from torch import nn

from utils import train_model_cifar10


class AlexNet(nn.Module):
    def __init__(self, classes: int, in_channels: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=1,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, padding=1
        )

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=256 * 5 * 5, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=classes)

        self.relu = nn.ReLU(inplace=True)
        self.lrn = nn.LocalResponseNorm(size=5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


train_model_cifar10(lambda: AlexNet(classes=10, in_channels=3), 5)
