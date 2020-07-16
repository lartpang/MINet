# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 下午9:15
# @Author  : Lart Pang
# @FileName: BaseOps.py
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn.functional as F


def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


def upsample_cat(*xs):
    y = xs[-1]
    out = []
    for x in xs[:-1]:
        out.append(F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False))
    return torch.cat([*out, y], dim=1)


def upsample_reduce(b, a):
    """
    上采样所有特征到最后一个特征的尺度以及前一个特征的通道数
    """
    _, C, _, _ = b.size()
    N, _, H, W = a.size()

    b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
    a = a.reshape(N, -1, C, H, W).mean(1)

    return b + a


def shuffle_channels(x, groups):
    """
    Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]
    一共C个channel要分成g组混合的channel，先把C reshape成(g, C/g)的形状，
    然后转置成(C/g, g)最后平坦成C组channel
    """
    N, C, H, W = x.size()
    x = x.reshape(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4)
    return x.reshape(N, C, H, W)


if __name__ == "__main__":
    a = torch.rand(3, 4, 10, 10)
    b = torch.rand(3, 2, 5, 5)
    print(upsample_reduce(b, a).size())
