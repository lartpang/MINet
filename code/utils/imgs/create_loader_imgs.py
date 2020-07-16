import random
from functools import partial

import torch
from prefetch_generator import BackgroundGenerator
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

from utils.config import arg_config
from utils.imgs.create_rgb_datasets_imgs import TestImageFolder, TrainImageFolder
from utils.misc import construct_print


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


def _make_trloader(dataset, shuffle=True, drop_last=False, size_list=None):
    if size_list == None:
        return DataLoaderX(
            dataset=dataset,
            batch_size=arg_config["batch_size"],
            num_workers=arg_config["num_workers"],
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )
    else:
        collate_fn = partial(_collate_fn, size_list=size_list)
        return DataLoaderX(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=arg_config["batch_size"],
            num_workers=arg_config["num_workers"],
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )


def _make_teloader(dataset, shuffle=True, drop_last=False):
    return DataLoaderX(
        dataset=dataset,
        batch_size=arg_config["batch_size"],
        num_workers=arg_config["num_workers"],
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )


def create_loader(data_path, mode, get_length=False, prefix=(".jpg", ".png"), size_list=None):
    if mode == "train":
        construct_print(f"Training on: {data_path}")
        train_set = TrainImageFolder(
            data_path,
            in_size=arg_config["input_size"],
            prefix=prefix,
            use_bigt=arg_config["use_bigt"],
        )
        loader = _make_trloader(train_set, shuffle=True, drop_last=True, size_list=size_list)
        length_of_dataset = len(train_set)
    elif mode == "test":
        construct_print(f"Testing on: {data_path}")
        test_set = TestImageFolder(data_path, in_size=arg_config["input_size"], prefix=prefix)
        loader = _make_teloader(test_set, shuffle=False, drop_last=False)
        length_of_dataset = len(test_set)
    else:
        raise NotImplementedError

    if get_length:
        return loader, length_of_dataset
    else:
        return loader


if __name__ == "__main__":
    loader = create_loader(
        data_path=arg_config["rgb_data"]["tr_data_path"],
        mode="train",
        get_length=False,
        size_list=arg_config["size_list"],
    )

    for idx, train_data in enumerate(loader):
        train_inputs, train_masks, *train_other_data = train_data
        print(f"" f"batch: {idx} ", train_inputs.size(), train_masks.size())
