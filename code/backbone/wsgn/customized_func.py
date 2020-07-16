import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.utils import model_zoo


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)


def load_pretrained_params(model, model_url: str):
    pretrained_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if list(pretrained_dict.keys())[0].startswith("module."):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
