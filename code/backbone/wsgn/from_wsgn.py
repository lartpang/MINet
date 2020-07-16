import torch
from torch import nn

from backbone.wsgn.resnet import l_resnet50
from backbone.wsgn.resnext import l_resnext50


def Backbone_ResNet50_in3():
    net = l_resnet50(pretrained=True)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


def Backbone_ResNeXt50_in3():
    net = l_resnext50(pretrained=True)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


if __name__ == "__main__":
    div_list = Backbone_ResNet50_in3()
    in_data = torch.randn((4, 3, 320, 320))
    for div_func in div_list:
        in_data = div_func(in_data)
        print(in_data.size())
