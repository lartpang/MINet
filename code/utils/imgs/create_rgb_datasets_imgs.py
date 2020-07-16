import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.joint_transforms import Compose, JointResize, RandomHorizontallyFlip, RandomRotate
from utils.misc import construct_print


def _get_ext(path_list):
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

    img_ext = _get_ext(img_list)
    mask_ext = _get_ext(mask_list)

    img_list = [os.path.splitext(f)[0] for f in mask_list if f.endswith(mask_ext)]
    return [
        (os.path.join(img_path, img_name + img_ext), os.path.join(mask_path, img_name + mask_ext))
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


def _make_test_dataset_from_list(list_filepath, prefix=(".jpg", ".png")):
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


class TestImageFolder(Dataset):
    def __init__(self, root, in_size, prefix):
        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will test on it.")
            self.imgs = _make_dataset(root)
        elif os.path.isfile(root):
            construct_print(
                f"{root} is a list of images, we will use these paths to read the "
                f"corresponding image"
            )
            self.imgs = _make_test_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError
        self.test_img_trainsform = transforms.Compose(
            [
                # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
                transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        img = Image.open(img_path).convert("RGB")
        img_name = (img_path.split(os.sep)[-1]).split(".")[0]

        img = self.test_img_trainsform(img)
        return img, img_name, mask_path

    def __len__(self):
        return len(self.imgs)


def _make_train_dataset_from_list(list_filepath, prefix=(".jpg", ".png")):
    # list_filepath = '/home/lart/Datasets/RGBDSaliency/FinalSet/rgbd_train_jw.lst'
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


class TrainImageFolder(Dataset):
    def __init__(self, root, in_size, prefix, use_bigt=False):
        self.use_bigt = use_bigt

        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will test on it.")
            self.imgs = _make_dataset(root)
        elif os.path.isfile(root):
            construct_print(
                f"{root} is a list of images, we will use these paths to read the "
                f"corresponding image"
            )
            self.imgs = _make_train_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError
        self.train_joint_transform = Compose(
            [JointResize(in_size), RandomHorizontallyFlip(), RandomRotate(10)]
        )
        self.train_img_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
            ]
        )
        self.train_mask_transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if len(img.split()) != 3:
            img = img.convert("RGB")
        if len(mask.split()) == 3:
            mask = mask.convert("L")

        img, mask = self.train_joint_transform(img, mask)
        mask = self.train_mask_transform(mask)
        img = self.train_img_transform(img)

        if self.use_bigt:
            mask = mask.ge(0.5).float()  # 二值化

        img_name = (img_path.split(os.sep)[-1]).split(".")[0]

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    img_list = _make_train_dataset_from_list()
    construct_print(len(img_list))
