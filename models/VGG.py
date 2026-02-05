import torch
from torch import nn

def get_vgg_features(config, in_channels, batch_norm=False):
    features = []
    curr_in_channels = in_channels

    for param in config:
        if isinstance(param, int):
            features.append(nn.Conv2d(
                in_channels=curr_in_channels, out_channels=param, kernel_size=3, stride=1, padding=1
            ))

            if batch_norm:
                features.append(nn.BatchNorm2d(param, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

            features.append(nn.ReLU(inplace=True))

            curr_in_channels = param
        else:
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*features)

class VGG(nn.Module):
    def __init__(self, in_channels, out_features, config, batch_norm=True):
        super().__init__()

        self.features = get_vgg_features(config, in_channels, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, out_features)
        )

        self._initialize_weights()

    def forward(self, X):
        feat = self.avgpool(self.features(X))
        flat_feat = torch.flatten(feat, 1)
        y_hat = self.classifier(flat_feat)
        return (y_hat, flat_feat)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)