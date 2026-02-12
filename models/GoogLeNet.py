import torch
from torch import nn
import torch.nn.functional as F 

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, bias=False):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)

class Inception(nn.Module):
    def __init__(
            self, in_channels, out_channels_br1, out_channels_br2_red, out_channels_br2, 
            out_channels_br3_red, out_channels_br3, out_channels_br4_proj
        ):
        super().__init__()

        self.branch1 = BasicConv2d(
            in_channels=in_channels,
            out_channels=out_channels_br1,
            kernel_size=(1, 1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(
                in_channels=in_channels, 
                out_channels=out_channels_br2_red, 
                kernel_size=(1, 1)
            ),
            BasicConv2d(
                in_channels=out_channels_br2_red, 
                out_channels=out_channels_br2, 
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(
                in_channels=in_channels, 
                out_channels=out_channels_br3_red, 
                kernel_size=(1, 1)
            ),
            BasicConv2d(
                in_channels=out_channels_br3_red, 
                out_channels=out_channels_br3, 
                kernel_size=(5, 5),
                padding=(2, 2)
            )
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=1, ceil_mode=True),
            BasicConv2d(
                in_channels=in_channels, 
                out_channels=out_channels_br4_proj, 
                kernel_size=(1, 1),
            )
        )

    def forward(self, X):
        return torch.cat([
            self.branch1(X),
            self.branch2(X),
            self.branch3(X),
            self.branch4(X)
        ], dim=1)
    
class AuxilaryClassifier(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5, 5), stride=3),
            BasicConv2d(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=(1, 1)
            ),
            nn.Flatten(), # Output: 2048
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=1024, out_features=out_features)
        )
    
    def forward(self, X):
        return self.classifier(X)

class GoogLeNet(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
    
        # input - 224 x 224 x 3
        self.conv1 = BasicConv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7),
            padding=(3, 3),
            stride=2
        ) #output - 112 x 112 x 64

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True) #output - 56 x 56 x 64

        self.conv2 = BasicConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=1
        ) #output - 56 x 56 x 64

        self.conv3 = BasicConv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1)
        ) #output - 56 x 56 x 192

        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True) #output - 28 x 28 x 192

        self.inception3a = Inception(
            in_channels=192,
            out_channels_br1=64,
            out_channels_br2_red=96,
            out_channels_br2=128,
            out_channels_br3_red=16,
            out_channels_br3=32,
            out_channels_br4_proj=32
        ) #output - 28 x 28 x 256

        self.inception3b = Inception(
            in_channels=256,
            out_channels_br1=128,
            out_channels_br2_red=128,
            out_channels_br2=192,
            out_channels_br3_red=32,
            out_channels_br3=96,
            out_channels_br4_proj=64
        ) #output - 28 x 28 x 480

        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True) #output - 14 x 14 x 480

        self.inception4a = Inception(
            in_channels=480,
            out_channels_br1=192,
            out_channels_br2_red=96,
            out_channels_br2=208,
            out_channels_br3_red=16,
            out_channels_br3=48,
            out_channels_br4_proj=64
        ) # Output: 14 x 14 x 512

        self.aux1 = AuxilaryClassifier(
            in_channels=512,
            out_features=out_features
        )

        self.inception4b = Inception(
            in_channels=512,
            out_channels_br1=160,
            out_channels_br2_red=112,
            out_channels_br2=224,
            out_channels_br3_red=24,
            out_channels_br3=64,
            out_channels_br4_proj=64
        ) # Output: 14 x 14 x 512

        self.inception4c = Inception(
            in_channels=512,
            out_channels_br1=128,
            out_channels_br2_red=128,
            out_channels_br2=256,
            out_channels_br3_red=24,
            out_channels_br3=64,
            out_channels_br4_proj=64
        ) # Output: 14 x 14 x 512

        self.inception4d = Inception(
            in_channels=512,
            out_channels_br1=112,
            out_channels_br2_red=144,
            out_channels_br2=288,
            out_channels_br3_red=32,
            out_channels_br3=64,
            out_channels_br4_proj=64
        ) # Output: 14 x 14 x 528

        self.aux2 = AuxilaryClassifier(
            in_channels=528,
            out_features=out_features
        )

        self.inception4e = Inception(
            in_channels=528,
            out_channels_br1=256,
            out_channels_br2_red=160,
            out_channels_br2=320,
            out_channels_br3_red=32,
            out_channels_br3=128,
            out_channels_br4_proj=128
        ) # Output: 14 x 14 x 832

        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True) #output - 7 x 7 x 832

        self.inception5a = Inception(
            in_channels=832,
            out_channels_br1=256,
            out_channels_br2_red=160,
            out_channels_br2=320,
            out_channels_br3_red=32,
            out_channels_br3=128,
            out_channels_br4_proj=128
        ) # Output: 7 x 7 x 832

        self.inception5b = Inception(
            in_channels=832,
            out_channels_br1=384,
            out_channels_br2_red=192,
            out_channels_br2=384,
            out_channels_br3_red=48,
            out_channels_br3=128,
            out_channels_br4_proj=128
        ) # Output: 7 x 7 x 1024

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # Output: 1 x 1 x 1024

        self.dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(in_features=1024, out_features=out_features)

    def forward(self, X):
        # Stem
        X = self.maxpool1(self.conv1(X))
        X = self.maxpool2(self.conv3(self.conv2(X)))
        
        # Stage 3
        X = self.inception3a(X) 
        X = self.inception3b(X) 
        X = self.maxpool3(X)

        # Stage 4
        X = self.inception4a(X)
        aux1_out = self.aux1(X) if self.training else None # Only run aux in training

        X = self.inception4b(X)
        X = self.inception4c(X)
        X = self.inception4d(X)
        aux2_out = self.aux2(X) if self.training else None

        X = self.inception4e(X)
        X = self.maxpool4(X)

        # Stage 5
        X = self.inception5a(X)
        X = self.inception5b(X)

        # Classifier
        X = self.avgpool(X)
        X = torch.flatten(X, 1) 
        X = self.dropout(X)
        logits = self.fc(X)

        if self.training:
            return logits, aux1_out, aux2_out
        else:
            return (logits, None)