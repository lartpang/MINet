# -*- coding: utf-8 -*-
from torch import nn

from backbone.origin.resnet import resnet50
from backbone.origin.vgg import vgg16_bn


def Backbone_ResNet50_in3():
    net = resnet50(pretrained=True)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


def Backbone_VGG16_in3():
    net = vgg16_bn(pretrained=True, progress=True)
    div_1 = nn.Sequential(*list(net.children())[0][0:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16
