import os
from datetime import datetime

__all__ = ["proj_root", "arg_config", "path_config"]

proj_root = os.path.dirname(os.path.dirname(__file__))
datasets_root = "/home/lart/Datasets/"

ecssd_path = os.path.join(datasets_root, "Saliency/RGBSOD", "ECSSD")
dutomron_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUT-OMRON")
hkuis_path = os.path.join(datasets_root, "Saliency/RGBSOD", "HKU-IS")
pascals_path = os.path.join(datasets_root, "Saliency/RGBSOD", "PASCAL-S")
soc_path = os.path.join(datasets_root, "Saliency/RGBSOD", "SOC/Test")
dutstr_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUTS/Train")
dutste_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUTS/Test")

# 配置区域 #####################################################################
arg_config = {
    # 常用配置
    "exp_name": "MINet_VGG16@e_40_lr_0.025_opti_f3trick_sche_Poly",  # <model_real_name>@<suffix>
    "resume_mode": "",  # the mode for resume parameters: ['train', 'test', '']
    "use_aux_loss": True,  # 是否使用辅助损失
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 50,  # 训练周期, 0: directly test model
    "lr": 0.001,  # 微调时缩小100倍
    "xlsx_name": "result.xlsx",  # the name of the record file
    "rgb_data": {
        "tr_data_path": dutstr_path,
        "te_data_list": {
            "dut-omron": dutomron_path,
            "hku-is": hkuis_path,
            "ecssd": ecssd_path,
            "pascal-s": pascals_path,
            "duts": dutste_path,
            "soc": soc_path,
        },
    },
    "tb_update": 10,  # >0 则使用tensorboard
    "print_freq": 10,  # >0, 保存迭代过程中的信息
    "prefix": (".jpg", ".png"),
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名，
    # "size_list": [224, 256, 288, 320, 352],
    "size_list": None,
    # if you dont use the multi-scale training, you can set 'size_list': None
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    "optim": "f3_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "sche_usebatch": False,
    "lr_type": "poly",
    "warmup_epoch": 1,
    # depond on the special lr_type, only lr_type has 'warmup', when set it to 1, it means no
    # warmup.
    "lr_decay": 0.9,  # poly
    "use_bigt": True,  # 有时似乎打开好，有时似乎关闭好？
    "batch_size": 4,  # 要是继续训练, 最好使用相同的batchsize
    "num_workers": 4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    "input_size": 320,
}
##################################################t##############################

ckpt_path = os.path.join(proj_root, "output")

pth_log_path = os.path.join(ckpt_path, arg_config["exp_name"].replace("@", "_"))
tb_path = os.path.join(pth_log_path, "tb")
save_path = os.path.join(pth_log_path, "pre")
pth_path = os.path.join(pth_log_path, "pth")

final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth.tar")
final_state_path = os.path.join(pth_path, "state_final.pth")

tr_log_path = os.path.join(pth_log_path, f"tr_{str(datetime.now())[:10]}.txt")
te_log_path = os.path.join(pth_log_path, f"te_{str(datetime.now())[:10]}.txt")
cfg_log_path = os.path.join(pth_log_path, f"cfg_{str(datetime.now())[:10]}.txt")
trainer_log_path = os.path.join(pth_log_path, f"trainer_{str(datetime.now())[:10]}.txt")
xlsx_path = os.path.join(ckpt_path, arg_config["xlsx_name"])

path_config = {
    "ckpt_path": ckpt_path,
    "pth_log": pth_log_path,
    "tb": tb_path,
    "save": save_path,
    "pth": pth_path,
    "final_full_net": final_full_model_path,
    "final_state_net": final_state_path,
    "tr_log": tr_log_path,
    "te_log": te_log_path,
    "cfg_log": cfg_log_path,
    "trainer_log": trainer_log_path,
    "xlsx": xlsx_path,
}
