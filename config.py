# -*- coding: utf-8 -*-
import os

__all__ = ["proj_root", "arg_config"]

from collections import OrderedDict

from utils import dataset_configs as rgb_data

proj_root = os.path.dirname(__file__)

data_setting = dict(
    tr=rgb_data.DUTS_TR,
    te=OrderedDict(
        {
            "PASCAL-S": rgb_data.PASCALS,
            "ECSSD": rgb_data.ECSSD,
            "HKU-IS": rgb_data.HKUIS,
            "DUTS-TE": rgb_data.DUTS_TE,
            "DUT-OMRON": rgb_data.DUTOMRON,
            "SOC": rgb_data.SOC_TE,
        }
    ),
)

arg_config = {
    "info": "",  # 关于本次实验的额外信息说明，这个会附加到本次试验的exp_name的结尾，如果为空，则不会附加内容。
    "resume_mode": "",  # the mode for resume parameters: ['train', 'test', '']
    "model": "MINet_VGG16",  # 实际使用的模型，需要在`network/__init__.py`中导入
    "use_amp": False,  # 是否使用amp加速训练
    "use_aux_loss": True,  # 是否使用辅助损失
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 50,  # 训练周期, 0: directly test model
    "lr": 0.001,  # 微调时缩小100倍
    "data": data_setting,
    "print_freq": 50,  # >0, 打印病保存迭代过程中的信息
    "ms_training": True,  # 是否使用多尺度训练
    "extra_scales": [0.8, 1.1],
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    "optim": "sgd_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "sche_usebatch": False,
    "lr_type": "poly",
    "warmup_epoch": 1,
    # depond on the special lr_type, only lr_type has 'warmup', when set it to 1, it means no warmup.
    "lr_decay": 0.9,  # poly
    "use_bigt": True,  # 训练时是否对真值二值化（阈值为0.5）
    "batch_size": 4,  # 要是继续训练, 最好使用相同的batchsize
    "num_workers": 4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    "in_size": 320,
}
