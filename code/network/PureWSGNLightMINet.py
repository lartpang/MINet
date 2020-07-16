import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from backbone.wsgn.from_wsgn import Backbone_ResNet50_in3
from module.WSGNLightModule import (BasicConv2d as WSGNBasicConv2d, LightAIM as WSGNLightAIM,
                                    SIM as WSGNSIM)
from utils.tensor_ops import cus_sample, upsample_add


class WSGNCPLMINet_Res50(nn.Module):
    def __init__(self):
        super(WSGNCPLMINet_Res50, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        
        self.div_2, self.div_4, self.div_8, self.div_16, self.div_32 = Backbone_ResNet50_in3()
        
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        
        self.trans = WSGNLightAIM(iC_list=(64, 256, 512, 1024, 2048), oC_list=(64, 64, 64, 64, 64))
        
        self.sim32 = WSGNSIM(64, 32)
        self.sim16 = WSGNSIM(64, 32)
        self.sim8 = WSGNSIM(64, 32)
        self.sim4 = WSGNSIM(64, 32)
        self.sim2 = WSGNSIM(64, 32)
        
        self.upconv32 = WSGNBasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = WSGNBasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = WSGNBasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = WSGNBasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = WSGNBasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = WSGNBasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
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
