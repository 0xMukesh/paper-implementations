import torch
from torch import nn

from utils import train_model_cifar10

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding: int | tuple[int, int] = 0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class InceptionA(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=96, kernel_size=1),
        )
        self.b2 = ConvBNReLU(in_channels=in_channels, out_channels=96, kernel_size=1)
        self.b3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=96, kernel_size=1),
            ConvBNReLU(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        )
        self.b4 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=64, kernel_size=1),
            ConvBNReLU(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        return torch.concat([b1, b2, b3, b4], 1)

class InceptionB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        )
        self.b2 = ConvBNReLU(in_channels=in_channels, out_channels=384, kernel_size=1)
        self.b3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=192, kernel_size=1),
            ConvBNReLU(in_channels=192, out_channels=224, kernel_size=(7, 1), padding=(3, 0)),
            ConvBNReLU(in_channels=224, out_channels=256, kernel_size=(1, 7), padding=(0, 3))
        )
        self.b4 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=192, kernel_size=1),
            ConvBNReLU(in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
            ConvBNReLU(in_channels=192, out_channels=224, kernel_size=(7, 1), padding=(3, 0)),
            ConvBNReLU(in_channels=224, out_channels=224, kernel_size=(1, 7), padding=(0, 3)),
            ConvBNReLU(in_channels=224, out_channels=256, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        return torch.concat([b1, b2, b3, b4], 1)

class InceptionC(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=256, kernel_size=1)
        )

        self.b2 = ConvBNReLU(in_channels=in_channels, out_channels=256, kernel_size=1)

        self.b3_1 = ConvBNReLU(in_channels=in_channels, out_channels=384, kernel_size=1)
        self.b3_a = ConvBNReLU(in_channels=384, out_channels=256, kernel_size=(1, 3), padding=(0, 1))
        self.b3_b = ConvBNReLU(in_channels=384, out_channels=256, kernel_size=(3, 1), padding=(1, 0))

        self.b4_1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=384, kernel_size=1),
            ConvBNReLU(in_channels=384, out_channels=448, kernel_size=(1, 3), padding=(0, 1)),
            ConvBNReLU(in_channels=448, out_channels=512, kernel_size=(3, 1), padding=(1, 0))
        )
        self.b4_a = ConvBNReLU(in_channels=512, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
        self.b4_b = ConvBNReLU(in_channels=512, out_channels=256, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)

        b3_1 = self.b3_1(x)
        b3_a = self.b3_a(b3_1)
        b3_b = self.b3_b(b3_1)
        b3 = torch.concat([b3_a, b3_b], 1)

        b4_1 = self.b4_1(x)
        b4_a = self.b4_a(b4_1)
        b4_b = self.b4_b(b4_1)
        b4 = torch.concat([b4_a, b4_b], 1)

        return torch.concat([b1, b2, b3, b4], 1)

class ReductionA(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.b2 = ConvBNReLU(in_channels=in_channels, out_channels=384, kernel_size=3, stride=2)
        self.b3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=192, kernel_size=1),
            ConvBNReLU(in_channels=192, out_channels=224, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=224, out_channels=256, kernel_size=3, stride=2)
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
            ConvBNReLU(in_channels=in_channels, out_channels=192, kernel_size=1),
            ConvBNReLU(in_channels=192, out_channels=192, kernel_size=3, stride=2)
        )
        self.b3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=256, kernel_size=1),
            ConvBNReLU(in_channels=256, out_channels=256, kernel_size=(1, 7), padding=(0, 3)),
            ConvBNReLU(in_channels=256, out_channels=320, kernel_size=(7, 1), padding=(3, 0)),
            ConvBNReLU(in_channels=320, out_channels=320, kernel_size=3, stride=2)
        )


    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)

        return torch.concat([b1, b2, b3], 1)

class Stem(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.brancha_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.brancha_2 = ConvBNReLU(in_channels=64, out_channels=96, kernel_size=3, stride=2)

        self.branchb_1 = nn.Sequential(
            ConvBNReLU(in_channels=160, out_channels=64, kernel_size=1),
            ConvBNReLU(in_channels=64, out_channels=96, kernel_size=3)
        )
        self.branchb_2 = nn.Sequential(
            ConvBNReLU(in_channels=160, out_channels=64, kernel_size=1),
            ConvBNReLU(in_channels=64, out_channels=64, kernel_size=(7, 1), padding=(3, 0)),
            ConvBNReLU(in_channels=64, out_channels=64, kernel_size=(1, 7), padding=(0, 3)),
            ConvBNReLU(in_channels=64, out_channels=96, kernel_size=3, stride=2)
        ) 

        self.branchc_1 = ConvBNReLU(in_channels=192, out_channels=192, kernel_size=3)
        self.branchc_2 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        ba_1 = self.brancha_1(x)
        ba_2 = self.brancha_2(x)
        x = torch.concat([ba_1, ba_2], 1)

        bb_1 = self.branchb_1(x)
        bb_2 = self.branchb_2(x)
        x = torch.concat([bb_1, bb_2], 1)

        bc_1 = self.branchc_1(x)
        bc_2 = self.branchc_2(x)
        x = torch.concat([bc_1, bc_2], 1)

        return x

class InceptionV4(nn.Module):
    def __init__(self, classes, in_channels: int = 3):
        super().__init__()

        self.stem = Stem(in_channels)
        
        self.incep_a = nn.Sequential(*([InceptionA(in_channels=384)] * 4))
        self.red_a = ReductionA(in_channels=384)
        
        self.incep_b = nn.Sequential(*([InceptionB(in_channels=1024)] * 7))
        self.red_b = ReductionB(in_channels=1024)
        
        self.incep_c = nn.Sequential(*([InceptionC(in_channels=1536)] * 3))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(in_features=1536, out_features=classes)
    
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

train_model_cifar10(lambda: InceptionV4(classes=10, in_channels=3), 10)
