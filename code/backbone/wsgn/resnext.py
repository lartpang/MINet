from __future__ import division

from torch.utils import model_zoo

"""
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016).
Aggregated residual transformations for deep neural networks.
arXiv preprint arXiv:1611.05431.
import from https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
"""
import math
import torch.nn as nn
from backbone.wsgn import customized_func as L

__all__ = ["l_resnext50", "l_resnext101"]

model_urls = {
    "resnext50": "http://cs.jhu.edu/~syqiao/WeightStandardization/X-50-GN-WS.pth.tar",
    "resnext101": "http://cs.jhu.edu/~syqiao/WeightStandardization/X-101-GN-WS.pth.tar",
}


class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """

    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = L.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = L.BatchNorm2d(D * C)
        self.conv2 = L.Conv2d(
            D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False
        )
        self.bn2 = L.BatchNorm2d(D * C)
        self.conv3 = L.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = L.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, baseWidth, cardinality, layers, num_classes):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = L.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = L.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                L.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                L.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def l_resnext50(pretrained, baseWidth, cardinality):
    """
    Construct ResNeXt-50.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 6, 3], 1000)
    if pretrained:
        L.load_pretrained_params(model, model_url=model_urls["resnext50"])
    return model


def l_resnext101(pretrained, baseWidth, cardinality):
    """
    Construct ResNeXt-101.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 23, 3], 1000)
    if pretrained:
        L.load_pretrained_params(model, model_url=model_urls["resnext101"])
    return model
