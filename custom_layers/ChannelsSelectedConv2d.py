import torch
from torch import nn

class ChannelsSelectedConv2d(nn.Module):
    def __init__(self, selected_in_channels, kernel_size):
        super().__init__()
        self.selected_in_channels = selected_in_channels

        self.layers = nn.ModuleList([
                nn.LazyConv2d(out_channels=1, kernel_size=kernel_size, stride=1, padding=0) 
                for _ in self.selected_in_channels
        ])

    def forward(self, X):
        output = []
        for in_channels, conv2d in zip(
            self.selected_in_channels, self.layers
        ):
            output.append(conv2d(X[:, in_channels, :, :]))
        
        return torch.cat(output, dim=1)