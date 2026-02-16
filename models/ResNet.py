import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_downsample=False):
        super().__init__()
        self.is_downsample = is_downsample

        stride = 2 if is_downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=stride,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if is_downsample:
            self.downsample = nn.Sequential( 
                # we downsample the input X, and map: in_channels -> out_channels
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=(1, 1),
                    stride=2,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, X):
        F = self.relu(self.bn1(self.conv1(X)))
        F = self.bn2(self.conv2(F))

        if self.is_downsample:
            X = self.downsample(X)

        return self.relu(X + F)


class ResNet34(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()

        # input: 224 x 224 x 3

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=2, 
            padding=(3, 3),
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=(1, 1))

        self.layer1 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64, is_downsample=False),
            ResBlock(in_channels=64, out_channels=64, is_downsample=False),
            ResBlock(in_channels=64, out_channels=64, is_downsample=False)
        )

        self.layer2 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=128, is_downsample=True),
            ResBlock(in_channels=128, out_channels=128, is_downsample=False),
            ResBlock(in_channels=128, out_channels=128, is_downsample=False),
            ResBlock(in_channels=128, out_channels=128, is_downsample=False),
        )

        self.layer3 = nn.Sequential(
            ResBlock(in_channels=128, out_channels=256, is_downsample=True),
            ResBlock(in_channels=256, out_channels=256, is_downsample=False),
            ResBlock(in_channels=256, out_channels=256, is_downsample=False),
            ResBlock(in_channels=256, out_channels=256, is_downsample=False),
            ResBlock(in_channels=256, out_channels=256, is_downsample=False),
            ResBlock(in_channels=256, out_channels=256, is_downsample=False),
        )

        self.layer4 = nn.Sequential(
            ResBlock(in_channels=256, out_channels=512, is_downsample=True),
            ResBlock(in_channels=512, out_channels=512, is_downsample=False),
            ResBlock(in_channels=512, out_channels=512, is_downsample=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(in_features=512, out_features=out_features)

    def forward(self, X):
        X = self.maxpool(self.relu(self.bn1(self.conv1(X))))

        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)

        X = self.flatten(self.avgpool(X))

        return self.fc(X)