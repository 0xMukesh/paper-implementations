import torch
from torch import nn

from utils import train_model_cifar10

class InceptionV1Block(nn.Module):
    def __init__(self, arch: list[int], in_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=arch[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=arch[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=arch[1], out_channels=arch[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=arch[3], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=arch[3], out_channels=arch[4], kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=arch[5], kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        return torch.concat([b1, b2, b3, b4], 1)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels: int, classes: int):
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, padding="same")
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)

        return x

class GoogLeNet(nn.Module):
    def __init__(self, classes: int, in_channels: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)

        self.incep_3a = InceptionV1Block([64, 96, 128, 16, 32, 32], 192)
        self.incep_3b = InceptionV1Block([128, 128, 192, 32, 96, 64], 256)

        self.incep_4a = InceptionV1Block([192, 96, 208, 16, 48, 64], 480)
        self.incep_4b = InceptionV1Block([160, 112, 224, 24, 64, 64], 512)
        self.incep_4c = InceptionV1Block([128, 128, 256, 24, 64, 64], 512)
        self.incep_4d = InceptionV1Block([112, 144, 288, 32, 64, 64], 512)
        self.incep_4e = InceptionV1Block([256, 160, 320, 32, 128, 128], 528)

        self.incep_5a = InceptionV1Block([256, 160, 320, 32, 128, 128], 832)
        self.incep_5b = InceptionV1Block([384, 192, 384, 48, 128, 128], 832)

        self.aux1 = AuxiliaryClassifier(in_channels=512, classes=classes)
        self.aux2 = AuxiliaryClassifier(in_channels=528, classes=classes)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(in_features=1024, out_features=1000)

        self.lrn = nn.LocalResponseNorm(size=5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = []

        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.lrn(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrn(x)

        x = self.maxpool(x)

        x = self.incep_3a(x)
        x = self.incep_3b(x)

        x = self.maxpool(x)

        x = self.incep_4a(x)

        if self.training:
            outputs.append(self.aux1(x))
        else:
            outputs.append(None)

        x = self.incep_4b(x)
        x = self.incep_4c(x)
        x = self.incep_4d(x)

        if self.training:
            outputs.append(self.aux2(x))
        else:
            outputs.append(None)
        
        x = self.incep_4e(x)

        x = self.maxpool(x)

        x = self.incep_5a(x)
        x = self.incep_5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        outputs.append(x)

        return outputs        

train_model_cifar10(lambda: GoogLeNet(classes=10, in_channels=3), 5)
