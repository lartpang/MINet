# -*- coding: utf-8 -*-
# @Time    : 2021/6/6
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

# -*- coding: utf-8 -*-
import os
import sys
from collections import OrderedDict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

sys.path.append("../")
import network as network_lib
from config import dutomron_path, dutste_path, ecssd_path, hkuis_path, pascals_path
from utils.dataloader import _make_dataset, _make_dataset_from_list, _mask_loader
from utils.misc import construct_print
from utils.pipeline_ops import resume_checkpoint


class InferImageSet(Dataset):
    def __init__(self, root, in_size, prefix):
        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will use these images directly.")
            self.imgs = _make_dataset(root)
        elif os.path.isfile(root):
            construct_print(
                f"{root} is a list of images, we will use these paths to read the "
                f"corresponding image"
            )
            self.imgs = _make_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        ori_shape = img.size
        img = self.img_transform(img)
        return dict(image=img, name=img_name, ori_shape=dict(h=ori_shape[1], w=ori_shape[0]))

    def __len__(self):
        return len(self.imgs)


def create_loader(data_path, input_size, prefix=(".jpg", ".png")):
    construct_print(f"Dataset: {data_path}")
    imageset = InferImageSet(data_path, in_size=input_size, prefix=prefix)
    loader = _mask_loader(imageset, shuffle=False, drop_last=False, size_list=None)
    return loader


def inference(model_name, state_path, save_root, input_size, dataset_info):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = getattr(network_lib, model_name)().to(dev)
    resume_checkpoint(model=net, load_path=state_path, mode="onlynet")
    net.eval()

    for data_name, data_path in dataset_info.items():
        construct_print(f"Testing with testset: {data_name}")
        loader = create_loader(data_path=data_path, input_size=input_size, prefix=(".jpg", ".png"))
        save_path = os.path.join(save_root, data_name)
        if not os.path.exists(save_path):
            construct_print(f"{save_path} do not exist. Let's create it.")
            os.makedirs(save_path)

        tqdm_iter = tqdm(enumerate(loader), total=len(loader), leave=False, desc="INFER:")
        for batch_id, batch_data in tqdm_iter:
            with torch.no_grad():
                in_imgs = batch_data["image"].to(dev, non_blocking=True)
                outputs = net(in_imgs)

            outputs_np = outputs.sigmoid().cpu().detach()

            batch_names = batch_data["name"]
            ori_hs = batch_data["ori_shape"]["h"]
            ori_ws = batch_data["ori_shape"]["w"]
            for item_id, out_item in enumerate(outputs_np):
                out_img = to_pil_image(out_item).resize(
                    (ori_ws[item_id], ori_hs[item_id]), resample=Image.NEAREST
                )
                oimg_path = os.path.join(save_path, batch_names[item_id] + ".png")
                # maybe you need a min-max normalization.
                # out_img = np.array(out_img)
                # out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())
                # out_img = Image.fromarray((out_img * 255).astype(np.uint8))
                out_img.save(oimg_path)


if __name__ == "__main__":
    inference(
        model_name="MINet_Res50",
        state_path="/home/lart/Coding/MINet/output/MINet/MINet_Res50.pth",
        save_root="../output/",
        input_size=320,
        dataset_info=OrderedDict(
            {
                "pascal-s": pascals_path,
                "ecssd": ecssd_path,
                "hku-is": hkuis_path,
                "duts": dutste_path,
                "dut-omron": dutomron_path,
            },
        ),
    )
