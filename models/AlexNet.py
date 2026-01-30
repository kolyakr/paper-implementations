import torch
from torch import nn
from collections import OrderedDict

BIAS_ZERO_LAYERS = [
    'conv1',
    'conv3'
]

def layers_init(module, name):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0, std=0.01)

        if name in BIAS_ZERO_LAYERS:
            nn.init.constant_(module.bias, 0)
        else:
            nn.init.constant_(module.bias, 1)

class AlexNet(nn.Module):

    def __init__(self, in_channels, out_features):
        super().__init__()

        self.net = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4)),
                ('relu1', nn.ReLU()),
                ('lrn1', nn.LocalResponseNorm(size=5, k=2, alpha=0.0001, beta=0.75)),
                ('max_pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)),
                ('relu2', nn.ReLU()),
                ('lrn2', nn.LocalResponseNorm(size=5, k=2, alpha=0.0001, beta=0.75)),
                ('max_pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)),
                ('relu3', nn.ReLU()),
                ('conv4', nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)),
                ('relu4', nn.ReLU()),
                ('conv5', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)),
                ('relu5', nn.ReLU()),
                ('max_pool3', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('flat', nn.Flatten()),
                ('fc1', nn.Linear(in_features=9216, out_features=4096)),
                ('relu6', nn.ReLU()),
                ('fc2', nn.Linear(in_features=4096, out_features=4096)),
                ('relu7', nn.ReLU()),
                ('fc3', nn.Linear(in_features=4096, out_features=out_features))
            ])
        )

        for name, module in self.net.named_children():
            layers_init(module, name)

    def forward(self, X):
        return self.net(X)

        