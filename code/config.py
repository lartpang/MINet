import os

__all__ = ["proj_root", "arg_config"]

from collections import OrderedDict

proj_root = os.path.dirname(__file__)
datasets_root = "/home/lart/Datasets/"

ecssd_path = os.path.join(datasets_root, "Saliency/RGBSOD", "ECSSD")
dutomron_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUT-OMRON")
hkuis_path = os.path.join(datasets_root, "Saliency/RGBSOD", "HKU-IS")
pascals_path = os.path.join(datasets_root, "Saliency/RGBSOD", "PASCAL-S")
soc_path = os.path.join(datasets_root, "Saliency/RGBSOD", "SOC/Test")
dutstr_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUTS/Train")
dutste_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUTS/Test")

arg_config = {
    "model": "MINet_VGG16",  # 实际使用的模型，需要在`network/__init__.py`中导入
    "info": "",  # 关于本次实验的额外信息说明，这个会附加到本次试验的exp_name的结尾，如果为空，则不会附加内容。
    "use_amp": False,  # 是否使用amp加速训练
    "resume_mode": "",  # the mode for resume parameters: ['train', 'test', '']
    "use_aux_loss": True,  # 是否使用辅助损失
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 50,  # 训练周期, 0: directly test model
    "lr": 0.001,  # 微调时缩小100倍
    "xlsx_name": "result.xlsx",  # the name of the record file
    # 数据集设置
    "rgb_data": {
        "tr_data_path": dutstr_path,
        "te_data_list": OrderedDict(
            {
                "pascal-s": pascals_path,
                "ecssd": ecssd_path,
                "hku-is": hkuis_path,
                "duts": dutste_path,
                "dut-omron": dutomron_path,
                "soc": soc_path,
            },
        ),
    },
    # 训练过程中的监控信息
    "tb_update": 10,  # >0 则使用tensorboard
    "print_freq": 10,  # >0, 保存迭代过程中的信息
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名
    "prefix": (".jpg", ".png"),
    # if you dont use the multi-scale training, you can set 'size_list': None
    # "size_list": [224, 256, 288, 320, 352],
    "size_list": None,  # 不使用多尺度训练
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    # 优化器与学习率衰减
    "optim": "sgd_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "sche_usebatch": False,
    "lr_type": "poly",
    "warmup_epoch": 1,  # depond on the special lr_type, only lr_type has 'warmup', when set it to 1, it means no warmup.
    "lr_decay": 0.9,  # poly
    "use_bigt": True,  # 训练时是否对真值二值化（阈值为0.5）
    "batch_size": 4,  # 要是继续训练, 最好使用相同的batchsize
    "num_workers": 4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    "input_size": 320,
}
