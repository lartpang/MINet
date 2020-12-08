# -*- coding: utf-8 -*-
# @Time    : 2020/12/8
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed, use_cudnn_benchmark):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.enabled = True
    if use_cudnn_benchmark:
        construct_print("We will use `torch.backends.cudnn.benchmark`")
    else:
        construct_print("We will not use `torch.backends.cudnn.benchmark`")
    torch.backends.cudnn.benchmark = use_cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)


def pre_mkdir(path_config: dict):
    # 提前创建好记录文件，避免自动创建的时候触发文件创建事件
    check_mkdir(path_config["pth_log"])
    write_data_to_file(f"=== te_log {datetime.now()} ===", path_config["te_log"])
    write_data_to_file(f"=== tr_log {datetime.now()} ===", path_config["tr_log"])

    # 提前创建好存储预测结果和存放模型以及tensorboard的文件夹
    check_mkdir(path_config["save"])
    check_mkdir(path_config["pth"])
    check_mkdir(path_config["tb"])


def check_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_data_to_file(data_str, file_path):
    with open(file_path, encoding="utf-8", mode="a") as f:
        f.write(data_str + "\n")


def construct_print(out_str: str, total_length: int = 80):
    if len(out_str) >= total_length:
        extended_str = "=="
    else:
        extended_str = "=" * ((total_length - len(out_str)) // 2 - 4)
    out_str = f" {extended_str}>> {out_str} <<{extended_str} "
    print(out_str)


def construct_path(proj_root: str, exp_name: str) -> dict:
    ckpt_path = os.path.join(proj_root, "output")

    pth_log_path = os.path.join(ckpt_path, exp_name)
    tb_path = os.path.join(pth_log_path, "tb")
    save_path = os.path.join(pth_log_path, "pre")
    pth_path = os.path.join(pth_log_path, "pth")

    final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth")
    final_state_path = os.path.join(pth_path, "state_final.pth")

    tr_log_path = os.path.join(pth_log_path, f"tr_{str(datetime.now())[:10]}.txt")
    te_log_path = os.path.join(pth_log_path, f"te_{str(datetime.now())[:10]}.txt")
    cfg_log_path = os.path.join(pth_log_path, f"cfg_{str(datetime.now())[:10]}.txt")
    trainer_log_path = os.path.join(pth_log_path, f"trainer_{str(datetime.now())[:10]}.txt")

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
    }
    return path_config


def construct_exp_name(arg_dict: dict):
    # If you know the function of these two lines, you can uncomment out them.
    # if arg_dict.get("special_name", None):
    #     return arg_dict["special_name"].replace("@", "_")

    # You can modify and supplement it according to your needs.
    focus_item = OrderedDict(
        {
            "input_size": "s",
            "batch_size": "bs",
            "epoch_num": "e",
            "warmup_epoch": "we",
            "use_amp": "amp",
            "lr": "lr",
            "lr_type": "lt",
            "optim": "ot",
            "use_aux_loss": "al",
            "use_bigt": "bi",
            "size_list": "ms",
            "info": "info",
        }
    )
    exp_name = f"{arg_dict['model']}"
    for k, v in focus_item.items():
        item = arg_dict[k]
        if isinstance(item, bool):
            item = "Y" if item else "N"
        elif isinstance(item, (list, tuple)):
            item = "Y" if item else "N"  # 只是判断是否飞空
        elif isinstance(item, str):
            if not item:
                continue
            if "_" in item:
                item = item.replace("_", "")
        elif item == None:
            item = "N"

        if isinstance(item, str):
            item = item.lower()
        exp_name += f"_{v.upper()}{item}"
    return exp_name
