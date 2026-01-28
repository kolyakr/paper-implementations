import torch
from torch import nn
from custom_layers import ChannelsSelectedConv2d
class LeNet(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self.selected_in_channels = [
            [0, 1, 2], 
            [1, 2, 3], 
            [2, 3, 4], 
            [3, 4, 5], 
            [4, 5, 0], 
            [5, 0, 1],
            [0, 1, 2, 3], 
            [1, 2, 3, 4], 
            [2, 3, 4, 5], 
            [3, 4, 5, 0], 
            [4, 5, 0, 1], 
            [5, 0, 1, 2],
            [0, 1, 3, 4], 
            [1, 2, 4, 5], 
            [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5]
        ]

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ChannelsSelectedConv2d(selected_in_channels=self.selected_in_channels, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=out_features)
        )

    def forward(self, X):
        return self.net(X)
    
   
    
