import torch
import torch.nn as nn
from module.BaseBlocks import BasicConv2d
from utils.tensor_ops import cus_sample, upsample_add
from backbone.origin.from_origin import Backbone_ResNet50_in3, Backbone_VGG16_in3
from module.MyLightModule import LightAIM, SIM
from torch.utils.checkpoint import checkpoint


class LightMINet_VGG16(nn.Module):
    def __init__(self):
        super(LightMINet_VGG16, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()

        self.trans = LightAIM((64, 128, 256, 512, 512), (32, 64, 64, 64, 64))

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


class CPLightMINet_VGG16(nn.Module):
    def __init__(self):
        super(CPLightMINet_VGG16, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()

        self.trans = LightAIM((64, 128, 256, 512, 512), (32, 64, 64, 64, 64))

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
        in_data_1 = checkpoint(self.checkpoint_en_1, in_data, self.dummy_tensor)
        in_data_2 = checkpoint(self.checkpoint_en_2, in_data_1)
        in_data_4 = checkpoint(self.checkpoint_en_4, in_data_2)
        in_data_8 = checkpoint(self.checkpoint_en_8, in_data_4)
        in_data_16 = checkpoint(self.checkpoint_en_16, in_data_8)

        in_data_1, in_data_2, in_data_4, in_data_8, in_data_16 = checkpoint(
            self.checkpoint_trans, in_data_1, in_data_2, in_data_4, in_data_8, in_data_16
        )
        out_data_16 = checkpoint(self.checkpoint_de_16, in_data_16)
        out_data_8 = checkpoint(self.checkpoint_de_8, in_data_8, out_data_16)
        out_data_4 = checkpoint(self.checkpoint_de_4, in_data_4, out_data_8)
        out_data_2 = checkpoint(self.checkpoint_de_2, in_data_2, out_data_4)
        out_data_1 = checkpoint(self.checkpoint_de_1, in_data_1, out_data_2)
        out_data = self.classifier(out_data_1)
        return out_data.sigmoid()

    def checkpoint_en_1(self, in_data, dummy_arg=None):
        # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
        assert dummy_arg is not None
        in_data_1 = self.encoder1(in_data)
        return in_data_1

    def checkpoint_en_2(self, in_data_1):
        in_data_2 = self.encoder2(in_data_1)
        return in_data_2

    def checkpoint_en_4(self, in_data_2):
        in_data_4 = self.encoder4(in_data_2)
        return in_data_4

    def checkpoint_en_8(self, in_data_4):
        in_data_8 = self.encoder8(in_data_4)
        return in_data_8

    def checkpoint_en_16(self, in_data_8):
        in_data_16 = self.encoder16(in_data_8)
        return in_data_16

    def checkpoint_trans(self, in_data_1, in_data_2, in_data_4, in_data_8, in_data_16):
        in_data_1, in_data_2, in_data_4, in_data_8, in_data_16 = self.trans(
            in_data_1, in_data_2, in_data_4, in_data_8, in_data_16
        )
        # Here we cannot use the following form, it will raise an error:
        # return self.trans(in_data_1, in_data_2, in_data_4, in_data_8, in_data_16)
        return in_data_1, in_data_2, in_data_4, in_data_8, in_data_16

    def checkpoint_de_16(self, in_data_16):
        out_data_16 = self.upconv16(self.sim16(in_data_16) + in_data_16)  # 1024
        return out_data_16

    def checkpoint_de_8(self, in_data_8, out_data_16):
        out_data_8 = self.upsample_add(out_data_16, in_data_8)
        out_data_8 = self.upconv8(self.sim8(out_data_8) + out_data_8)  # 512
        return out_data_8

    def checkpoint_de_4(self, in_data_4, out_data_8):
        out_data_4 = self.upsample_add(out_data_8, in_data_4)
        out_data_4 = self.upconv4(self.sim4(out_data_4) + out_data_4)  # 256
        return out_data_4

    def checkpoint_de_2(self, in_data_2, out_data_4):
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        out_data_2 = self.upconv2(self.sim2(out_data_2) + out_data_2)  # 64
        return out_data_2

    def checkpoint_de_1(self, in_data_1, out_data_2):
        out_data_1 = self.upsample_add(out_data_2, in_data_1)
        out_data_1 = self.upconv1(self.sim1(out_data_1) + out_data_1)  # 32
        return out_data_1


class CPLightMINet_Res50(nn.Module):
    def __init__(self):
        super(CPLightMINet_Res50, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self.div_2, self.div_4, self.div_8, self.div_16, self.div_32 = Backbone_ResNet50_in3()

        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.trans = LightAIM(iC_list=(64, 256, 512, 1024, 2048), oC_list=(64, 64, 64, 64, 64))

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
        in_data_2 = checkpoint(self.checkpoint_en_2, in_data, self.dummy_tensor)
        in_data_4 = checkpoint(self.checkpoint_en_4, in_data_2)
        in_data_8 = checkpoint(self.checkpoint_en_8, in_data_4)
        in_data_16 = checkpoint(self.checkpoint_en_16, in_data_8)
        in_data_32 = checkpoint(self.checkpoint_en_32, in_data_16)

        in_data_2, in_data_4, in_data_8, in_data_16, in_data_32 = checkpoint(
            self.checkpoint_trans, in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        )

        out_data_32 = checkpoint(self.checkpoint_de_32, in_data_32)
        out_data_16 = checkpoint(self.checkpoint_de_16, in_data_16, out_data_32)
        out_data_8 = checkpoint(self.checkpoint_de_8, in_data_8, out_data_16)
        out_data_4 = checkpoint(self.checkpoint_de_4, in_data_4, out_data_8)
        out_data_2 = checkpoint(self.checkpoint_de_2, in_data_2, out_data_4)
        out_data = checkpoint(self.checkpoint_de_1, out_data_2)
        return out_data.sigmoid()

    def checkpoint_en_2(self, in_data, dummy_arg=None):
        # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
        assert dummy_arg is not None
        in_data_2 = self.div_2(in_data)
        return in_data_2

    def checkpoint_en_4(self, in_data_2):
        in_data_4 = self.div_4(in_data_2)
        return in_data_4

    def checkpoint_en_8(self, in_data_4):
        in_data_8 = self.div_8(in_data_4)
        return in_data_8

    def checkpoint_en_16(self, in_data_8):
        in_data_16 = self.div_16(in_data_8)
        return in_data_16

    def checkpoint_en_32(self, in_data_16):
        in_data_32 = self.div_32(in_data_16)
        return in_data_32

    def checkpoint_trans(self, in_data_2, in_data_4, in_data_8, in_data_16, in_data_32):
        in_data_2, in_data_4, in_data_8, in_data_16, in_data_32 = self.trans(
            in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        )
        # Here we cannot use the following form, it will raise an error:
        # return self.trans(in_data_1, in_data_2, in_data_4, in_data_8, in_data_16)
        return in_data_2, in_data_4, in_data_8, in_data_16, in_data_32

    def checkpoint_de_32(self, in_data_32):
        out_data_32 = self.upconv32(self.sim32(in_data_32) + in_data_32)  # 1024
        return out_data_32

    def checkpoint_de_16(self, in_data_16, out_data_32):
        out_data_16 = self.upsample_add(out_data_32, in_data_16)  # 1024
        out_data_16 = self.upconv16(self.sim16(out_data_16) + out_data_16)
        return out_data_16

    def checkpoint_de_8(self, in_data_8, out_data_16):
        out_data_8 = self.upsample_add(out_data_16, in_data_8)
        out_data_8 = self.upconv8(self.sim8(out_data_8) + out_data_8)  # 512
        return out_data_8

    def checkpoint_de_4(self, in_data_4, out_data_8):
        out_data_4 = self.upsample_add(out_data_8, in_data_4)
        out_data_4 = self.upconv4(self.sim4(out_data_4) + out_data_4)  # 256
        return out_data_4

    def checkpoint_de_2(self, in_data_2, out_data_4):
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        out_data_2 = self.upconv2(self.sim2(out_data_2) + out_data_2)  # 64
        return out_data_2

    def checkpoint_de_1(self, out_data_2):
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        out_data = self.classifier(out_data_1)
        return out_data
