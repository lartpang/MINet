# -*- coding: utf-8 -*-
# @Time    : 2020/7/22
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : dataloader.py
# @Project : code
# @GitHub  : https://github.com/lartpang
import os
import random
from collections import defaultdict
from functools import partial

import torch
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.nn.functional import interpolate
from torch.utils import data
from torchvision import transforms

from utils import construct_print


class Compose(object):
    def __init__(self, trans):
        self.transforms = trans

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    def __call__(self, img, mask):
        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)
        return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class ImageFolder(data.Dataset):
    def __init__(self, root, in_size, training, use_bigt=False):
        self.training = training
        self.use_bigt = use_bigt

        total_data = self.make_dataset(root)
        self.images = total_data["image"]
        self.masks = total_data["mask"]

        if self.training:
            self.joint_transform = Compose([JointResize(in_size), RandomHorizontallyFlip(), RandomRotate(10)])
            img_transform = [transforms.ColorJitter(0.1, 0.1, 0.1)]
            self.mask_transform = transforms.ToTensor()
        else:
            # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
            img_transform = [transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR)]
        self.img_transform = transforms.Compose(
            [
                *img_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path).convert("RGB")
        if self.training:
            mask = Image.open(mask_path).convert("L")
            image, mask = self.joint_transform(image, mask)
            image = self.img_transform(image)
            mask = self.mask_transform(mask)
            if self.use_bigt:
                mask = mask.ge(0.5).float()  # 二值化
            return image, mask
        else:
            # todo: When evaluating, the mask path may not exist. But our code defaults to its existence, which makes
            #  it impossible to use dataloader to generate a prediction without a mask path.
            image = self.img_transform(image)
            return image, mask_path

    def __len__(self):
        return len(self.images)

    @staticmethod
    def make_dataset(root: dict) -> dict:
        image_root = root["image"]["path"]
        image_ext = root["image"]["suffix"]
        mask_root = root["mask"]["path"]
        mask_ext = root["mask"]["suffix"]

        dataset_info = defaultdict(list)
        for name in sorted(os.listdir(mask_root)):
            name_wo_ext = os.path.splitext(name)[0]
            dataset_info["image"].append(os.path.join(image_root, name_wo_ext + image_ext))
            dataset_info["mask"].append(os.path.join(mask_root, name_wo_ext + mask_ext))
        return dataset_info

    @staticmethod
    def collate_fn(batch, size_list):
        size = random.choice(size_list)
        image, mask, image_name = [list(item) for item in zip(*batch)]
        image = torch.stack(image, dim=0)
        image = interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
        mask = torch.stack(mask, dim=0)
        mask = interpolate(mask, size=(size, size), mode="nearest")
        return image, mask, image_name


class DataLoaderX(data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


def create_loader(training, data_info, in_size, use_bigt, batch_size, num_workers=2, size_list=None, get_length=False):
    construct_print(f"{['Test', 'Training'][training]} on: {data_info['root']}")
    imageset = ImageFolder(root=data_info, in_size=in_size, training=training, use_bigt=use_bigt)

    loader_kwargs = dict(
        dataset=imageset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=training,
        drop_last=training,
        pin_memory=True,
    )
    if training and float(torch.__version__[:3]) >= 1.2:
        loader_kwargs["collate_fn"] = partial(imageset.collate_fn, size_list=size_list)
    loader = DataLoaderX(**loader_kwargs)

    if get_length:
        length_of_dataset = len(imageset)
        return loader, length_of_dataset
    else:
        return loader
