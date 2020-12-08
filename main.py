# -*- coding: utf-8 -*-
import os
import shutil
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import utils.models as network_lib
from config import arg_config, proj_root
from utils.dataloader import create_loader
from utils.metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from utils.pipeline_ops import (
    get_total_loss,
    make_optimizer,
    make_scheduler,
    resume_checkpoint,
    save_checkpoint,
)
from utils.tool_funcs import (
    AvgMeter,
    construct_exp_name,
    construct_path,
    construct_print,
    pre_mkdir,
    set_seed,
    write_data_to_file,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CalTotalMetric(object):
    def __init__(self):
        self.cal_mae = MAE()
        self.cal_fm = Fmeasure()
        self.cal_sm = Smeasure()
        self.cal_em = Emeasure()
        self.cal_wfm = WeightedFmeasure()

    def step(self, pred: np.ndarray, gt: np.ndarray, gt_path: str):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape, gt_path)
        assert pred.dtype == np.uint8, pred.dtype
        assert gt.dtype == np.uint8, gt.dtype

        self.cal_mae.step(pred, gt)
        self.cal_fm.step(pred, gt)
        self.cal_sm.step(pred, gt)
        self.cal_em.step(pred, gt)
        self.cal_wfm.step(pred, gt)

    def get_results(self, bit_width: int = 3) -> dict:
        fm = self.cal_fm.get_results()["fm"]
        wfm = self.cal_wfm.get_results()["wfm"]
        sm = self.cal_sm.get_results()["sm"]
        em = self.cal_em.get_results()["em"]
        mae = self.cal_mae.get_results()["mae"]
        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }
        results = {name: metric.round(bit_width) for name, metric in results.items()}
        return results


class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)


class Solver:
    def __init__(self, exp_name: str, arg_dict: dict, path_dict: dict):
        super(Solver, self).__init__()
        self.exp_name = exp_name
        self.arg_dict = arg_dict
        self.path_dict = path_dict

        # 依赖与前面属性的属性
        self.tr_loader = create_loader(
            training=True,
            data_info=self.arg_dict["data"]["tr"],
            in_size=self.arg_dict["in_size"],
            use_bigt=self.arg_dict["use_bigt"],
            batch_size=self.arg_dict["batch_size"],
            num_workers=self.arg_dict["num_workers"],
            ms_training=self.arg_dict["ms_training"],
            extra_scales=self.arg_dict["extra_scales"],
        )
        self.end_epoch = self.arg_dict["epoch_num"]
        self.iter_num = self.end_epoch * len(self.tr_loader)

        if hasattr(network_lib, self.arg_dict["model"]):
            self.net = getattr(network_lib, self.arg_dict["model"])().to(DEVICE)
        else:
            raise AttributeError
        pprint(self.arg_dict)

        if self.arg_dict["resume_mode"] == "test":
            # resume model only to test model.
            # self.start_epoch is useless
            resume_checkpoint(
                model=self.net,
                load_path=self.path_dict["final_state_net"],
                mode="onlynet",
            )
            return

        self.loss_funcs = [torch.nn.BCEWithLogitsLoss(reduction=self.arg_dict["reduction"]).to(DEVICE)]
        if self.arg_dict["use_aux_loss"]:
            self.loss_funcs.append(CEL().to(DEVICE))

        self.opti = make_optimizer(
            model=self.net,
            optimizer_type=self.arg_dict["optim"],
            optimizer_info=dict(
                lr=self.arg_dict["lr"],
                momentum=self.arg_dict["momentum"],
                weight_decay=self.arg_dict["weight_decay"],
                nesterov=self.arg_dict["nesterov"],
            ),
        )
        self.sche = make_scheduler(
            optimizer=self.opti,
            total_num=self.iter_num if self.arg_dict["sche_usebatch"] else self.end_epoch,
            scheduler_type=self.arg_dict["lr_type"],
            scheduler_info=dict(lr_decay=self.arg_dict["lr_decay"], warmup_epoch=self.arg_dict["warmup_epoch"]),
        )

        # AMP
        if self.arg_dict["use_amp"]:
            construct_print("Now, we will use the amp to accelerate training!")
            from apex import amp

            self.amp = amp
            self.net, self.opti = self.amp.initialize(self.net, self.opti, opt_level="O1")
        else:
            self.amp = None

        if self.arg_dict["resume_mode"] == "train":
            # resume model to train the model
            self.start_epoch = resume_checkpoint(
                model=self.net,
                optimizer=self.opti,
                scheduler=self.sche,
                amp=self.amp,
                exp_name=self.exp_name,
                load_path=self.path_dict["final_full_net"],
                mode="all",
            )
        else:
            # only train a new model.
            self.start_epoch = 0

    def train(self):
        self.net.train()

        for curr_epoch in range(self.start_epoch, self.end_epoch):
            loss_recorder = AvgMeter()
            self._train_per_epoch(curr_epoch, loss_recorder)

            # 根据周期修改学习率
            if not self.arg_dict["sche_usebatch"]:
                self.sche.step()

            # 每个周期都进行保存测试，保存的是针对第curr_epoch+1周期的参数
            save_checkpoint(
                model=self.net,
                optimizer=self.opti,
                scheduler=self.sche,
                amp=self.amp,
                exp_name=self.exp_name,
                current_epoch=curr_epoch + 1,
                full_net_path=self.path_dict["final_full_net"],
                state_net_path=self.path_dict["final_state_net"],
            )  # 保存参数

        if self.arg_dict["use_amp"]:
            # https://github.com/NVIDIA/apex/issues/567
            with self.amp.disable_casts():
                construct_print("When evaluating, we wish to evaluate in pure fp32.")
                self.test()
        else:
            self.test()

    def _train_per_epoch(self, curr_epoch, loss_recorder):
        for batch_id, batch in enumerate(self.tr_loader):
            num_iter_per_epoch = len(self.tr_loader)
            curr_iter = curr_epoch * num_iter_per_epoch + batch_id

            self.opti.zero_grad()

            images = batch[0].to(DEVICE, non_blocking=True)
            masks = batch[1].to(DEVICE, non_blocking=True)
            seg_logits = self.net(images)

            train_loss, loss_item_list = get_total_loss(seg_logits, masks, self.loss_funcs)
            if self.amp:
                with self.amp.scale_loss(train_loss, self.opti) as scaled_loss:
                    scaled_loss.backward()
            else:
                train_loss.backward()
            self.opti.step()

            if self.arg_dict["sche_usebatch"]:
                self.sche.step()

            # 仅在累计的时候使用item()获取数据
            loss_iter = train_loss.item()
            loss_recorder.update(loss_iter, images.size(0))

            # 记录每一次迭代的数据
            if self.arg_dict["print_freq"] > 0 and (curr_iter + 1) % self.arg_dict["print_freq"] == 0:
                lr_str = ",".join([f"{param_groups['lr']:.7f}" for param_groups in self.opti.param_groups])
                log = (
                    f"{batch_id}:{num_iter_per_epoch}/{curr_iter}:{self.iter_num}/"
                    f"{curr_epoch}:{self.end_epoch} {self.exp_name} {list(images.size())}\nLr:{lr_str} "
                    f"M:{loss_recorder.avg:.5f} C:{loss_iter:.5f} {loss_item_list}"
                )
                print(log)
                write_data_to_file(log, self.path_dict["tr_log"])

    def test(self):
        self.net.eval()

        for data_name, data_info in self.arg_dict["data"]["te"].items():
            construct_print(f"Testing with testset: {data_name}")
            te_loader = create_loader(
                training=False,
                data_info=data_info,
                in_size=self.arg_dict["in_size"],
                use_bigt=self.arg_dict["use_bigt"],
                batch_size=self.arg_dict["batch_size"],
                num_workers=self.arg_dict["num_workers"],
            )
            save_path = os.path.join(self.path_dict["save"], data_name)
            if not os.path.exists(save_path):
                construct_print(f"{save_path} do not exist. Let's create it.")
                os.makedirs(save_path)
            results = self._test_process(loader=te_loader, save_pre=self.arg_dict["save_pre"], save_path=save_path)
            msg = f"Results on the testset({data_name}:'{data_info['root']}'): {results}"
            construct_print(msg)
            write_data_to_file(msg, self.path_dict["te_log"])

    def _test_process(self, loader, save_pre, save_path):
        cal_total_seg_metrics = CalTotalMetric()

        tqdm_iter = tqdm(enumerate(loader), total=len(loader), leave=False, ncols=79)
        for batch_id, batch in tqdm_iter:
            tqdm_iter.set_description(f"{self.exp_name}: te=>{batch_id + 1}")

            in_data = batch[0].to(DEVICE, non_blocking=True)
            with torch.no_grad():
                outputs = self.net(in_data)

            outputs_np = outputs.sigmoid().cpu().detach()

            in_mask_paths = batch[1]
            for item_id, out_item in enumerate(outputs_np):
                gt_path = os.path.join(in_mask_paths[item_id])
                gt_image = Image.open(gt_path).convert("L")
                out_img = to_pil_image(out_item).resize(gt_image.size, resample=Image.NEAREST)

                if save_pre:
                    oimg_path = os.path.join(save_path, os.path.basename(gt_path))
                    out_img.save(oimg_path)

                gt_array = np.array(gt_image)
                out_array = np.array(out_img)
                cal_total_seg_metrics.step(pred=out_array, gt=gt_array, gt_path=gt_path)
        fixed_seg_results = cal_total_seg_metrics.get_results()
        return fixed_seg_results


if __name__ == "__main__":
    construct_print(f"{datetime.now()}: Initializing...")
    construct_print(f"Project Root: {proj_root}")
    init_start = datetime.now()

    exp_name = construct_exp_name(arg_config)
    path_config = construct_path(proj_root=proj_root, exp_name=exp_name)
    pre_mkdir(path_config)
    set_seed(seed=0, use_cudnn_benchmark=not arg_config["ms_training"])

    solver = Solver(exp_name, arg_config, path_config)
    construct_print(f"Total initialization time：{datetime.now() - init_start}")

    shutil.copy(f"{proj_root}/config.py", path_config["cfg_log"])
    shutil.copy(f"{proj_root}/main.py", path_config["trainer_log"])

    construct_print(f"{datetime.now()}: Start...")
    if arg_config["resume_mode"] == "test":
        solver.test()
    else:
        solver.train()
    construct_print(f"{datetime.now()}: End...")
