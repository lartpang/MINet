# -*- coding: utf-8 -*-
import argparse
import os
import warnings

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

import utils.models as network_lib

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def only_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="MINet_Res50", help="Name of model",
                        choices=[name for name in network_lib.__dict__.keys() if name.startswith("MINet_")])
    parser.add_argument("--weight", type=str, required=True, help="Path of trained weight.")
    parser.add_argument("--in-size", type=int, default=320, help="Size of the input.")
    parser.add_argument("-i", "--input-path-or-dir", type=str, required=True, help="Path/Dir of input images.")
    parser.add_argument("--save-dir", type=str, required=True, help="Path/Dir of output predictions.")
    args = parser.parse_args()

    net: nn.Module = network_lib.__dict__[args.model](pretrained=False)
    if not os.path.isfile(args.weight):
        warnings.warn(f"args.weight ({args.weight}) must be a file.")
    else:
        net.load_state_dict(torch.load(args.weight, map_location='cpu'))
    net.to(DEVICE)
    net.eval()

    # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.in_size, args.in_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if os.path.isfile(args.input_path_or_dir):
        image_paths = [args.input_path_or_dir]
    else:
        image_paths = [os.path.join(args.input_path_or_dir, name) for name in os.listdir(args.input_path_or_dir)
                       if os.path.isfile(name)]
    os.makedirs(args.save_dir, exist_ok=True)

    for image_path in image_paths:
        image_name_woext = os.path.basename(image_path).split('.')[:-1]
        image_name_woext = ".".join(image_name_woext)

        image = Image.open(image_path).convert("RGB")
        ori_w, ori_h = image.size
        image = test_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = net(image).sigmoid().squeeze().cpu().detach()  # H,W

        out_img = to_pil_image(output).resize((ori_w, ori_h), resample=Image.NEAREST)
        out_img.save(os.path.join(args.save_dir, image_name_woext + ".png"))


if __name__ == "__main__":
    only_predict()
