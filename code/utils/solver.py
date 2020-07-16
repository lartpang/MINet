import os
from pprint import pprint

import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.nn import BCELoss
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

import network as network_lib
from loss.CEL import CEL
from utils.config import arg_config, path_config
from utils.imgs.create_loader_imgs import create_loader
from utils.metric import cal_maxf, cal_pr_mae_meanf
from utils.misc import AvgMeter, construct_print, make_log, write_xlsx
from utils.pipeline_ops import (
    get_total_loss,
    make_optimizer,
    make_scheduler,
    resume_checkpoint,
    save_checkpoint,
)


class Solver:
    def __init__(self, args, path):
        super(Solver, self).__init__()
        self.args = args
        self.path = path
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to_pil = transforms.ToPILImage()
        self.exp_name = args["exp_name"]
        if "@" in self.exp_name:
            network_realname = self.exp_name.split("@")[0]
        else:
            network_realname = self.exp_name
        self.exp_name = self.exp_name.replace("@", "_")

        self.tr_data_path = self.args["rgb_data"]["tr_data_path"]
        self.te_data_list = self.args["rgb_data"]["te_data_list"]

        self.save_path = self.path["save"]
        self.save_pre = self.args["save_pre"]
        if self.args["tb_update"] > 0:
            self.tb = SummaryWriter(self.path["tb"])

        # 依赖与前面属性的属性
        self.pth_path = self.path["final_state_net"]
        self.tr_loader = create_loader(
            data_path=self.tr_data_path,
            mode="train",
            get_length=False,
            prefix=self.args["prefix"],
            size_list=self.args["size_list"],
        )
        self.end_epoch = self.args["epoch_num"]
        self.iter_num = self.end_epoch * len(self.tr_loader)

        if hasattr(network_lib, network_realname):
            self.net = getattr(network_lib, network_realname)().to(self.dev)
        else:
            raise AttributeError
        pprint(self.args)

        self.opti = make_optimizer(
            model=self.net,
            optimizer_type=self.args["optim"],
            optimizer_info=dict(
                lr=self.args["lr"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
                nesterov=self.args["nesterov"],
            ),
        )
        self.sche = make_scheduler(
            optimizer=self.opti,
            total_num=self.iter_num if self.args["sche_usebatch"] else self.end_epoch,
            scheduler_type=self.args["lr_type"],
            scheduler_info=dict(
                lr_decay=self.args["lr_decay"], warmup_epoch=self.args["warmup_epoch"]
            ),
        )

        if self.args["resume_mode"] == "train":
            # resume model to train the model
            self.start_epoch = resume_checkpoint(
                model=self.net,
                optimizer=self.opti,
                scheduler=self.sche,
                exp_name=self.exp_name,
                load_path=self.path["final_full_net"],
                mode="all",
            )
            self.only_test = False
        elif self.args["resume_mode"] == "test":
            # resume model only to test model.
            # self.start_epoch is useless
            resume_checkpoint(
                model=self.net, load_path=self.pth_path, mode="onlynet",
            )
            self.only_test = True
        elif not self.args["resume_mode"]:
            # only train a new model.
            self.start_epoch = 0
            self.only_test = False
        else:
            raise NotImplementedError

        if not self.only_test:
            # 损失函数
            self.loss_funcs = [BCELoss(reduction=self.args["reduction"]).to(self.dev)]
            if self.args["use_aux_loss"]:
                self.loss_funcs.append(CEL().to(self.dev))

    def train(self):
        for curr_epoch in range(self.start_epoch, self.end_epoch):
            train_loss_record = AvgMeter()
            for train_batch_id, train_data in enumerate(self.tr_loader):
                curr_iter = curr_epoch * len(self.tr_loader) + train_batch_id

                self.opti.zero_grad()
                train_inputs, train_masks, *train_other_data = train_data
                train_inputs = train_inputs.to(self.dev, non_blocking=True)
                train_masks = train_masks.to(self.dev, non_blocking=True)
                train_preds = self.net(train_inputs)

                train_loss, loss_item_list = get_total_loss(
                    train_preds, train_masks, self.loss_funcs
                )
                train_loss.backward()
                self.opti.step()

                if self.args["sche_usebatch"]:
                    self.sche.step()

                # 仅在累计的时候使用item()获取数据
                train_iter_loss = train_loss.item()
                train_batch_size = train_inputs.size(0)
                train_loss_record.update(train_iter_loss, train_batch_size)

                # 显示tensorboard
                if self.args["tb_update"] > 0 and (curr_iter + 1) % self.args["tb_update"] == 0:
                    self.tb.add_scalar("data/trloss_avg", train_loss_record.avg, curr_iter)
                    self.tb.add_scalar("data/trloss_iter", train_iter_loss, curr_iter)
                    for idx, param_groups in enumerate(self.opti.param_groups):
                        self.tb.add_scalar(f"data/lr_{idx}", param_groups["lr"], curr_iter)
                    tr_tb_mask = make_grid(train_masks, nrow=train_batch_size, padding=5)
                    self.tb.add_image("trmasks", tr_tb_mask, curr_iter)
                    tr_tb_out_1 = make_grid(train_preds, nrow=train_batch_size, padding=5)
                    self.tb.add_image("trsodout", tr_tb_out_1, curr_iter)

                # 记录每一次迭代的数据
                if self.args["print_freq"] > 0 and (curr_iter + 1) % self.args["print_freq"] == 0:
                    lr_str = ",".join(
                        [f"{param_groups['lr']:.7f}" for param_groups in self.opti.param_groups]
                    )
                    log = (
                        f"[I:{curr_iter}/{self.iter_num}][E:{curr_epoch}:{self.end_epoch}]>"
                        f"[{self.exp_name}]"
                        f"[Lr:{lr_str}]"
                        f"[Avg:{train_loss_record.avg:.5f}|Cur:{train_iter_loss:.5f}|"
                        f"{loss_item_list}]"
                    )
                    print(log)
                    make_log(self.path["tr_log"], log)

            # 根据周期修改学习率
            if not self.args["sche_usebatch"]:
                self.sche.step()

            # 每个周期都进行保存测试，保存的是针对第curr_epoch+1周期的参数
            save_checkpoint(
                model=self.net,
                optimizer=self.opti,
                scheduler=self.sche,
                exp_name=self.exp_name,
                current_epoch=curr_epoch + 1,
                full_net_path=self.path["final_full_net"],
                state_net_path=self.path["final_state_net"],
            )  # 保存参数

        total_results = self.test()
        # save result into xlsx file.
        write_xlsx(self.exp_name, total_results)

    def test(self):
        self.net.eval()

        total_results = {}
        for data_name, data_path in self.te_data_list.items():
            construct_print(f"Testing with testset: {data_name}")
            self.te_loader = create_loader(
                data_path=data_path, mode="test", get_length=False, prefix=self.args["prefix"],
            )
            self.save_path = os.path.join(self.path["save"], data_name)
            if not os.path.exists(self.save_path):
                construct_print(f"{self.save_path} do not exist. Let's create it.")
                os.makedirs(self.save_path)
            results = self.__test_process(save_pre=self.save_pre)
            msg = f"Results on the testset({data_name}:'{data_path}'): {results}"
            construct_print(msg)
            make_log(self.path["te_log"], msg)

            total_results[data_name.upper()] = results

        self.net.train()
        return total_results

    def __test_process(self, save_pre):
        loader = self.te_loader

        pres = [AvgMeter() for _ in range(256)]
        recs = [AvgMeter() for _ in range(256)]
        meanfs = AvgMeter()
        maes = AvgMeter()

        tqdm_iter = tqdm(enumerate(loader), total=len(loader), leave=False)
        for test_batch_id, test_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.exp_name}: te=>{test_batch_id + 1}")
            with torch.no_grad():
                in_imgs, in_names, in_mask_paths = test_data
                in_imgs = in_imgs.to(self.dev, non_blocking=True)
                outputs = self.net(in_imgs)

            outputs_np = outputs.cpu().detach()

            for item_id, out_item in enumerate(outputs_np):
                gimg_path = os.path.join(in_mask_paths[item_id])
                gt_img = Image.open(gimg_path).convert("L")
                out_img = self.to_pil(out_item).resize(gt_img.size, resample=Image.NEAREST)

                if save_pre:
                    oimg_path = os.path.join(self.save_path, in_names[item_id] + ".png")
                    out_img.save(oimg_path)

                gt_img = np.asarray(gt_img)
                out_img = np.array(out_img)
                ps, rs, mae, meanf = cal_pr_mae_meanf(out_img, gt_img)
                for pidx, pdata in enumerate(zip(ps, rs)):
                    p, r = pdata
                    pres[pidx].update(p)
                    recs[pidx].update(r)
                maes.update(mae)
                meanfs.update(meanf)
        maxf = cal_maxf([pre.avg for pre in pres], [rec.avg for rec in recs])
        results = {"MAXF": maxf, "MEANF": meanfs.avg, "MAE": maes.avg}
        return results


if __name__ == "__main__":
    solver = Solver(args=arg_config, path=path_config)
    print(solver.exp_name)
