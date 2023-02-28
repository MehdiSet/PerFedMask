#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .models import ScalableModule




def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        in_channels = 3
        num_classes = args.num_classes
        
        hidden_size = 64
        
        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )
        
        self.linear = nn.Linear(hidden_size*2*2, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        logits = self.linear(features)
        
        return logits
    
    def extract_features(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        
        return features
        
'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNetCifar(ScalableModule):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1

    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    input_shape = [None, 3, 32, 32]
    
    def __init__(self, num_classes=10, track_running_stats=False, width_scale=1.,rescale_init=False, 
                 share_affine=False, rescale_layer=False, bn_type='bn',):
        super(MobileNetCifar, self).__init__(width_scale=width_scale, rescale_init=rescale_init,
                                     rescale_layer=rescale_layer)
        
        self.cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
        
        if width_scale != 1.:
            temp_cfg=[]
            for x in self.cfg:
                I1 = int(x * width_scale) if isinstance(x, int) else int(x[0] * width_scale)
                I2 = 1 if isinstance(x, int) else x[1]
                if I2==1:
                    temp_cfg.append(I1)
                else:
                    temp_cfg.append((I1, I2))
                    
            self.cfg = temp_cfg
            
        
        
        self.conv1 = nn.Conv2d(3, int(32 * width_scale), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * width_scale), track_running_stats=False)
        self.layers = self._make_layers(in_planes=int(32 *width_scale))
        self.linear = nn.Linear(int(width_scale*1024), num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        return out