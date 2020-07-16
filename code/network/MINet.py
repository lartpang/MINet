import torch
import torch.nn as nn

from module.BaseBlocks import BasicConv2d
from utils.tensor_ops import cus_sample, upsample_add
from backbone.origin.from_origin import Backbone_ResNet50_in3, Backbone_VGG16_in3
from module.MyModule import AIM, SIM


class MINet_VGG16(nn.Module):
    def __init__(self):
        super(MINet_VGG16, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()

        self.trans = AIM((64, 128, 256, 512, 512), (32, 64, 64, 64, 64))

        self.sim16 = SIM(64, 32)
        self.sim8 = SIM(64, 32)
        self.sim4 = SIM(64, 32)
        self.sim2 = SIM(64, 32)
        self.sim1 = SIM(32, 16)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data):
        in_data_1 = self.encoder1(in_data)
        in_data_2 = self.encoder2(in_data_1)
        in_data_4 = self.encoder4(in_data_2)
        in_data_8 = self.encoder8(in_data_4)
        in_data_16 = self.encoder16(in_data_8)

        in_data_1, in_data_2, in_data_4, in_data_8, in_data_16 = self.trans(
            in_data_1, in_data_2, in_data_4, in_data_8, in_data_16
        )

        out_data_16 = self.upconv16(self.sim16(in_data_16) + in_data_16)  # 1024

        out_data_8 = self.upsample_add(out_data_16, in_data_8)
        out_data_8 = self.upconv8(self.sim8(out_data_8) + out_data_8)  # 512

        out_data_4 = self.upsample_add(out_data_8, in_data_4)
        out_data_4 = self.upconv4(self.sim4(out_data_4) + out_data_4)  # 256

        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        out_data_2 = self.upconv2(self.sim2(out_data_2) + out_data_2)  # 64

        out_data_1 = self.upsample_add(out_data_2, in_data_1)
        out_data_1 = self.upconv1(self.sim1(out_data_1) + out_data_1)  # 32

        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()


class MINet_Res50(nn.Module):
    def __init__(self):
        super(MINet_Res50, self).__init__()
        self.div_2, self.div_4, self.div_8, self.div_16, self.div_32 = Backbone_ResNet50_in3()

        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.trans = AIM(iC_list=(64, 256, 512, 1024, 2048), oC_list=(64, 64, 64, 64, 64))

        self.sim32 = SIM(64, 32)
        self.sim16 = SIM(64, 32)
        self.sim8 = SIM(64, 32)
        self.sim4 = SIM(64, 32)
        self.sim2 = SIM(64, 32)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data):
        in_data_2 = self.div_2(in_data)
        in_data_4 = self.div_4(in_data_2)
        in_data_8 = self.div_8(in_data_4)
        in_data_16 = self.div_16(in_data_8)
        in_data_32 = self.div_32(in_data_16)

        in_data_2, in_data_4, in_data_8, in_data_16, in_data_32 = self.trans(
            in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        )

        out_data_32 = self.upconv32(self.sim32(in_data_32) + in_data_32)  # 1024

        out_data_16 = self.upsample_add(out_data_32, in_data_16)  # 1024
        out_data_16 = self.upconv16(self.sim16(out_data_16) + out_data_16)

        out_data_8 = self.upsample_add(out_data_16, in_data_8)
        out_data_8 = self.upconv8(self.sim8(out_data_8) + out_data_8)  # 512

        out_data_4 = self.upsample_add(out_data_8, in_data_4)
        out_data_4 = self.upconv4(self.sim4(out_data_4) + out_data_4)  # 256

        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        out_data_2 = self.upconv2(self.sim2(out_data_2) + out_data_2)  # 64

        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        out_data = self.classifier(out_data_1)

        return out_data.sigmoid()


if __name__ == "__main__":
    in_data = torch.randn((1, 3, 288, 288))
    net = MINet_VGG16()
    print(net(in_data).size())
