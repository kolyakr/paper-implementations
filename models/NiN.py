import torch
from torch import nn

def NiN_block(in_channels, out_channels, n_MLP=2, kernel_size=(3, 3), stride=1, padding=0):
    block = [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding
        ),
        nn.ReLU(inplace=True)
    ]

    for _ in range(n_MLP):
        block.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1)),
            nn.ReLU(inplace=True)
        ])
    
    return block


class NiN(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()

        self.features = nn.Sequential(
            *NiN_block(in_channels=in_channels, out_channels=96, n_MLP=2, kernel_size=(11, 11), stride=4),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(p=0.5),
            *NiN_block(in_channels=96, out_channels=256, n_MLP=2, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(p=0.5),
            *NiN_block(in_channels=256, out_channels=384, n_MLP=2, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            *NiN_block(in_channels=384, out_channels=out_features, n_MLP=2, kernel_size=(3, 3), padding=1),
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
    
    def forward(self, X):
        features = self.features(X)
        outputs = self.global_avg_pool(features)

        return outputs, features