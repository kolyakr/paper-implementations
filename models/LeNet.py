import torch
from torch import nn
from custom_layers import ChannelsSelectedConv2d
from collections import OrderedDict
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
    
   
class LeNetVisualization(LeNet):
    def __init__(self, out_features=10):
        super().__init__(out_features)
    
    def forward(self, X):
        feature_maps = [X]
        current_input = X
        
        for layer in self.net:
            current_input = layer(current_input)
            feature_maps.append(current_input)
        
        return current_input, feature_maps
    

class LeNetV2(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            ('flatten', nn.Flatten()),
            
            ('fc1', nn.Linear(in_features=400, out_features=120)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(in_features=120, out_features=84)),
            ('relu4', nn.ReLU()),
            ('fc3', nn.Linear(in_features=84, out_features=out_features))
        ]))

    def forward(self, X):
        return self.net(X)