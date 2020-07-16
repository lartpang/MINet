# -*- coding: utf-8 -*-
# @Time    : 2020/3/28
# @Author  : Lart Pang
# @FileName: CEL.py
# @GitHub  : https://github.com/lartpang

from torch import nn


class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)
