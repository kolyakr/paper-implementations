import torch
from torch import nn
from custom import ChannelsSelectedConv2d
from custom.BatchNorm import BatchNorm, BatchNormFixed, BatchNormMean, BatchNormVar
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
        return (self.net(X), None)
    
   
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
    def __init__(self, out_features, in_channels=1):
        super().__init__()

        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0)),
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
    

class LeNetV3(nn.Module):
    def __init__(self, out_features, in_channels=1):
        super().__init__()

        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=5, stride=1, padding=0)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            ('conv2', nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            ('flatten', nn.Flatten()),
            
            ('fc1', nn.Linear(in_features=864, out_features=200)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(in_features=200, out_features=100)),
            ('relu4', nn.ReLU()),
            ('fc3', nn.Linear(in_features=100, out_features=out_features))
        ]))

    def forward(self, X):
        return self.net(X)
    
#-----------------------------------------------
class LeNetBaseline(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        # input - 1 x 32 x 32
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0 # output: 6 x 28 x 28
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # output: 6 x 14 x 14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), # output: 16 x 10 x 10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # output: 16 x 5 x 5
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=out_features)
        )

    def forward(self, X):
        return (self.net(X), None)

class LeNetFullBN(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5, bias=False), 
            BatchNorm(6, num_dims=4),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(6, 16, 5, bias=False),
            BatchNorm(16, num_dims=4),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(400, 120, bias=False),
            BatchNorm(120, num_dims=2),
            nn.Tanh(),
            
            nn.Linear(120, 84, bias=False),
            BatchNorm(84, num_dims=2),
            nn.Tanh(),
            
            nn.Linear(84, out_features)
        )
    def forward(self, X): return (self.net(X), None)

class LeNetMeanOnly(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5, bias=False),
            BatchNormMean(6, num_dims=4),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, bias=False),
            BatchNormMean(16, num_dims=4),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(400, 120, bias=False),
            BatchNormMean(120, num_dims=2),
            nn.Tanh(),
            nn.Linear(120, 84, bias=False),
            BatchNormMean(84, num_dims=2),
            nn.Tanh(),
            nn.Linear(84, out_features)
        )
    def forward(self, X): return (self.net(X), None)
class LeNetVarOnly(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5, bias=False),
            BatchNormVar(6, num_dims=4),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, bias=False),
            BatchNormVar(16, num_dims=4),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(400, 120, bias=False),
            BatchNormVar(120, num_dims=2),
            nn.Tanh(),
            nn.Linear(120, 84, bias=False),
            BatchNormVar(84, num_dims=2),
            nn.Tanh(),
            nn.Linear(84, out_features)
        )
    def forward(self, X): return (self.net(X), None)

class LeNetDropout(nn.Module):
    def __init__(self, out_features, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(400, 120),
            nn.Tanh(),
            nn.Dropout(p),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, out_features)
        )
    def forward(self, X): return (self.net(X), None)

class LeNetBNDropout(nn.Module):
    def __init__(self, out_features, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5, bias=False), 
            BatchNorm(6, num_dims=4),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(6, 16, 5, bias=False),
            BatchNorm(16, num_dims=4),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            
            nn.Flatten(),
            nn.Dropout(p),

            nn.Linear(400, 120, bias=False),
            BatchNorm(120, num_dims=2),
            nn.Tanh(),
            nn.Dropout(p),
            
            nn.Linear(120, 84, bias=False),
            BatchNorm(84, num_dims=2),
            nn.Tanh(),
            
            nn.Linear(84, out_features)
        )
    def forward(self, X): return (self.net(X), None)

class LeNetFixedBN(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5, bias=False), BatchNormFixed(6, 4), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, bias=False), BatchNormFixed(16, 4), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(400, 120, bias=False), BatchNormFixed(120, 2), nn.Tanh(),
            nn.Linear(120, 84, bias=False), BatchNormFixed(84, 2), nn.Tanh(),
            nn.Linear(84, out_features)
        )
    def forward(self, X): return (self.net(X), None)