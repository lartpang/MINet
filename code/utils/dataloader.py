# -*- coding: utf-8 -*-
# @Time    : 2020/7/22
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : dataloader.py
# @Project : code
# @GitHub  : https://github.com/lartpang
import os
import random
from functools import partial

import torch
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from config import arg_config
from utils.joint_transforms import Compose, JointResize, RandomHorizontallyFlip, RandomRotate
from utils.misc import construct_print


def _get_suffix(path_list):
    ext_list = list(set([os.path.splitext(p)[1] for p in path_list]))
    if len(ext_list) != 1:
        if ".png" in ext_list:
            ext = ".png"
        elif ".jpg" in ext_list:
            ext = ".jpg"
        elif ".bmp" in ext_list:
            ext = ".bmp"
        else:
            raise NotImplementedError
        construct_print(f"数据文件夹中包含多种扩展名，这里仅使用{ext}")
    else:
        ext = ext_list[0]
    return ext


def _make_dataset(root):
    img_path = os.path.join(root, "Image")
    mask_path = os.path.join(root, "Mask")

    img_list = os.listdir(img_path)
    mask_list = os.listdir(mask_path)

    img_suffix = _get_suffix(img_list)
    mask_suffix = _get_suffix(mask_list)

    img_list = [os.path.splitext(f)[0] for f in mask_list if f.endswith(mask_suffix)]
    return [
        (
            os.path.join(img_path, img_name + img_suffix),
            os.path.join(mask_path, img_name + mask_suffix),
        )
        for img_name in img_list
    ]


def _read_list_from_file(list_filepath):
    img_list = []
    with open(list_filepath, mode="r", encoding="utf-8") as openedfile:
        line = openedfile.readline()
        while line:
            img_list.append(line.split()[0])
            line = openedfile.readline()
    return img_list


def _make_dataset_from_list(list_filepath, prefix=(".jpg", ".png")):
    img_list = _read_list_from_file(list_filepath)
    return [
        (
            os.path.join(
                os.path.join(os.path.dirname(img_path), "Image"),
                os.path.basename(img_path) + prefix[0],
            ),
            os.path.join(
                os.path.join(os.path.dirname(img_path), "Mask"),
                os.path.basename(img_path) + prefix[1],
            ),
        )
        for img_path in img_list
    ]


class ImageFolder(Dataset):
    def __init__(self, root, in_size, training, prefix, use_bigt=False):
        self.training = training
        self.use_bigt = use_bigt

        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will test on it.")
            self.imgs = _make_dataset(root)
        elif os.path.isfile(root):
            construct_print(
                f"{root} is a list of images, we will use these paths to read the "
                f"corresponding image"
            )
            self.imgs = _make_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError

        if self.training:
            self.joint_transform = Compose(
                [JointResize(in_size), RandomHorizontallyFlip(), RandomRotate(10)]
            )
            img_transform = [transforms.ColorJitter(0.1, 0.1, 0.1)]
            self.mask_transform = transforms.ToTensor()
        else:
            # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
            img_transform = [
                transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),
            ]
        self.img_transform = transforms.Compose(
            [
                *img_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        if self.training:
            mask = Image.open(mask_path).convert("L")
            img, mask = self.joint_transform(img, mask)
            img = self.img_transform(img)
            mask = self.mask_transform(mask)
            if self.use_bigt:
                mask = mask.ge(0.5).float()  # 二值化
            return img, mask, img_name
        else:
            # todo: When evaluating, the mask path may not exist. But our code defaults to its existence, which makes
            #  it impossible to use dataloader to generate a prediction without a mask path.
            img = self.img_transform(img)
            return img, mask_path, img_name

    def __len__(self):
        return len(self.imgs)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


def _collate_fn(batch, size_list):
    size = random.choice(size_list)
    img, mask, image_name = [list(item) for item in zip(*batch)]
    img = torch.stack(img, dim=0)
    img = interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    mask = torch.stack(mask, dim=0)
    mask = interpolate(mask, size=(size, size), mode="nearest")
    return img, mask, image_name


def _mask_loader(dataset, shuffle, drop_last, size_list):
    return DataLoaderX(
        dataset=dataset,
        collate_fn=partial(_collate_fn, size_list=size_list) if size_list else None,
        batch_size=arg_config["batch_size"],
        num_workers=arg_config["num_workers"],
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )


def create_loader(data_path, training, size_list=None, prefix=(".jpg", ".png"), get_length=False):
    if training:
        construct_print(f"Training on: {data_path}")
        imageset = ImageFolder(
            data_path,
            in_size=arg_config["input_size"],
            prefix=prefix,
            use_bigt=arg_config["use_bigt"],
            training=True,
        )
        loader = _mask_loader(imageset, shuffle=True, drop_last=True, size_list=size_list)
    else:
        construct_print(f"Testing on: {data_path}")
        imageset = ImageFolder(
            data_path, in_size=arg_config["input_size"], prefix=prefix, training=False,
        )
        loader = _mask_loader(imageset, shuffle=False, drop_last=False, size_list=size_list)

    if get_length:
        length_of_dataset = len(imageset)
        return loader, length_of_dataset
    else:
        return loader


if __name__ == "__main__":
    loader = create_loader(
        data_path=arg_config["rgb_data"]["tr_data_path"],
        training=True,
        get_length=False,
        size_list=arg_config["size_list"],
    )

    for idx, train_data in enumerate(loader):
        train_inputs, train_masks, *train_other_data = train_data
        print(f"" f"batch: {idx} ", train_inputs.size(), train_masks.size())
