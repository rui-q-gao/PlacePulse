"""
A module for the siamese nn implementation for PlacePulse.

Author: Rui Gao
Date: Jan 22, 2020
"""

import torch
from torchvision.models.vgg import VGG
from torch import nn
import torchvision.models as models


class SNNet(VGG):
    def __init__(self):
        super(SNNet, self).__init__(make_layers(cfgs['D'], batch_norm=True))
        # self.ranker = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 1),
        # )


    def forward(self, data):
        res = []
        for i in range(2):
            x = data[i]
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            res.append(x)

        x = res[1] - res[0]
        x = self.classifier(x)
        return x


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,
          'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
          512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
          512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# model = SNNet()
# model.load_state_dict(models.vgg16_bn(pretrained=True).state_dict())
