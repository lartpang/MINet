# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 上午9:54
# @Author  : Lart Pang
# @FileName: cal_fps.py
# @Project : MINet
# @GitHub  : https://github.com/lartpang

import os
import time

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import ecssd_path
from network.MINet import MINet_VGG16


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class GPUFPSer:
    def __init__(self, proj_name, args, pth_path):
        super(GPUFPSer, self).__init__()
        self.args = args
        self.to_pil = transforms.ToPILImage()
        self.proj_name = proj_name
        self.dev = torch.device("cuda:0")
        self.net = self.args[proj_name]["net"]().to(self.dev)
        self.net.eval()

        self.test_image_transform = transforms.Compose(
            [
                # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
                transforms.Resize(
                    (self.args["new_size"], self.args["new_size"]), interpolation=Image.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if pth_path != None:
            print(f"导入模型...{pth_path}")
            checkpoint = torch.load(pth_path)
            model_dict = self.net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.net.load_state_dict(model_dict)
            print("初始化完毕...")
        else:
            print("不加载权重")

    def test(self, data_path, save_path):

        if save_path:
            print(f"保存路径为{save_path}")
            check_mkdir(save_path)

        print(f"开始统计...{data_path}")
        img_path = os.path.join(data_path, "Image")
        img_list = os.listdir(img_path)
        total_time = 0

        tqdm_iter = tqdm(enumerate(img_list), total=len(img_list), leave=False)
        for idx, img_name in tqdm_iter:
            tqdm_iter.set_description(f"{self.proj_name}:te=>{idx + 1}")

            img_fullpath = os.path.join(img_path, img_name)
            test_image = Image.open(img_fullpath).convert("RGB")
            img_size = test_image.size

            test_image = self.test_image_transform(test_image)
            test_image = test_image.unsqueeze(0)
            test_image = test_image.to(self.dev)
            with torch.no_grad():
                # https://discuss.pytorch.org/t/how-to-reduce-time-spent-by-torch-cuda-synchronize/29484
                # https://blog.csdn.net/u013548568/article/details/81368019
                torch.cuda.synchronize()
                start_time = time.time()
                outputs = self.net(test_image)  # 按照实际情况改写
                torch.cuda.synchronize()
                total_time += time.time() - start_time

            if save_path:
                outputs_np = outputs.squeeze(0).cpu().detach()
                out_img = self.to_pil(outputs_np).resize(img_size, Image.NEAREST)
                oimg_path = os.path.join(save_path, img_name[:-4] + ".png")
                out_img.save(oimg_path)

        fps = len(img_list) / total_time
        return fps


class CPUFPSer:
    def __init__(self, proj_name, args, pth_path):
        super(CPUFPSer, self).__init__()
        self.args = args
        self.to_pil = transforms.ToPILImage()
        self.proj_name = proj_name
        self.dev = torch.device("cpu")
        self.net = self.args[proj_name]["net"]().to(self.dev)
        self.net.eval()

        self.test_image_transform = transforms.Compose(
            [
                # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
                transforms.Resize(
                    (self.args["new_size"], self.args["new_size"]), interpolation=Image.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if pth_path != None:
            print(f"导入模型...{pth_path}")
            checkpoint = torch.load(pth_path)
            model_dict = self.net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.net.load_state_dict(model_dict)
            print("初始化完毕...")
        else:
            print("不加载权重")

    def test(self, data_path, save_path):
        if save_path:
            print(f"保存路径为{save_path}")
            check_mkdir(save_path)

        print(f"开始统计...{data_path}")
        img_path = os.path.join(data_path, "Image")
        img_list = os.listdir(img_path)
        total_time = 0

        tqdm_iter = tqdm(enumerate(img_list), total=len(img_list), leave=False)
        for idx, img_name in tqdm_iter:
            tqdm_iter.set_description(f"{self.proj_name}:te=>{idx + 1}")

            img_fullpath = os.path.join(img_path, img_name)
            test_image = Image.open(img_fullpath).convert("RGB")

            img_size = test_image.size
            test_image = self.test_image_transform(test_image)
            test_image = test_image.unsqueeze(0)
            test_image = test_image.to(self.dev)
            with torch.no_grad():
                start_time = time.time()
                outputs = self.net(test_image)  # 按照实际情况改写
                total_time += time.time() - start_time

            if save_path:
                outputs_np = outputs.squeeze(0).detach()
                out_img = self.to_pil(outputs_np).resize(img_size, Image.NEAREST)
                oimg_path = os.path.join(save_path, img_name[:-4] + ".png")
                out_img.save(oimg_path)

        fps = len(img_list) / total_time
        return fps


if __name__ == "__main__":
    proj_list = ["MINet"]

    arg_dicts = {
        "MINet": {"net": MINet_VGG16, "pth_path": None, "save_root": ""},  # 必须有
        "new_size": 320,
        "test_on_gpu": True,
    }

    data_dicts = {
        "ecssd1": ecssd_path,
        "ecssd2": ecssd_path,
        "ecssd3": ecssd_path,
        "ecssd4": ecssd_path,
    }

    for proj_name in proj_list:
        pth_path = arg_dicts[proj_name]["pth_path"]
        for data_name, data_path in data_dicts.items():
            if arg_dicts[proj_name]["save_root"]:
                save_path = os.path.join(arg_dicts[proj_name]["save_root"], data_name)
            else:
                save_path = None
            if arg_dicts["test_on_gpu"]:
                print(f" ==>> 在GPU上测试模型 <<== ")
                fpser = GPUFPSer(proj_name, arg_dicts, pth_path)
            else:
                print(f" ==>> 在CPU上测试模型 <<== ")
                fpser = CPUFPSer(proj_name, arg_dicts, pth_path)
            fps = fpser.test(data_path, save_path)
            print(f"\n ==>> Model: {proj_name}: Dataset: {data_name} \n ==>> FPS: {fps}")
            del fpser

    print(" ==>> 所有模型速度测试完毕 <<== ")
