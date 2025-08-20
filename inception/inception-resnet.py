import torch
from torch import nn

from utils import train_model_cifar10


class Stem(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=3, stride=2
        )
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(num_features=80)

        self.conv5 = nn.Conv2d(in_channels=80, out_channels=192, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(num_features=192)

        self.conv6 = nn.Conv2d(
            in_channels=192, out_channels=256, kernel_size=3, stride=2
        )
        self.bn6 = nn.BatchNorm2d(num_features=256)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))

        return x


class InceptionResNetA(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1),
            nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv_1x1 = nn.Conv2d(in_channels=32 * 3, out_channels=256, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)

        x = torch.concat([b1, b2, b3], 1)
        x = self.conv_1x1(x)
        x = identity + 0.2 * x

        x = self.relu(x)

        return x


class InceptionResNetB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(1, 7), padding=(0, 3)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(7, 1), padding=(3, 0)
            ),
            nn.ReLU(),
        )

        self.conv_1x1 = nn.Conv2d(in_channels=128 * 2, out_channels=896, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        b1 = self.b1(x)
        b2 = self.b2(x)

        x = torch.concat([b1, b2], 1)
        x = self.conv_1x1(x)
        x = identity + 0.2 * x

        x = self.relu(x)
        return x


class InceptionResNetC(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=192, kernel_size=1),
            nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=192, out_channels=192, kernel_size=(1, 3), padding=(0, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=192, out_channels=192, kernel_size=(3, 1), padding=(1, 0)
            ),
            nn.ReLU(),
        )

        self.conv_1x1 = nn.Conv2d(in_channels=192 * 2, out_channels=1792, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        b1 = self.b1(x)
        b2 = self.b2(x)
        x = torch.concat([b1, b2], 1)
        x = self.conv_1x1(x)
        x = identity + 0.2 * x

        x = self.relu(x)
        return x


class ReductionA(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=384, kernel_size=3, stride=2
            ),
            nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)

        return torch.concat([b1, b2, b3], 1)


class ReductionB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        return torch.concat([b1, b2, b3, b4], 1)


class InceptionResNetV1(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()

        self.stem = Stem(in_channels)
        self.incep_a = nn.Sequential(*[InceptionResNetA(256) for _ in range(5)])
        self.red_a = ReductionA(256)
        self.incep_b = nn.Sequential(*[InceptionResNetB(896) for _ in range(10)])
        self.red_b = ReductionB(896)
        self.incep_c = nn.Sequential(*[InceptionResNetC(1792) for _ in range(5)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1792, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incep_a(x)
        x = self.red_a(x)
        x = self.incep_b(x)
        x = self.red_b(x)
        x = self.incep_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


train_model_cifar10(lambda: InceptionResNetV1(num_classes=10, in_channels=3), 10)
