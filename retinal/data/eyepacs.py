import torch
from torch.utils.data import DataLoader
import os.path as osp
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Optional, Callable, List, Any
from torch.utils.data.dataset import Dataset


class EyePacsDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transformer: Optional[Callable] = None,
        return_id: bool = False,
    ):
        assert split in {"train", "val", "trainval", "test"}
        self.data_root = data_root
        self.split = split
        self.transformer = transformer
        self.return_id = return_id

        self.num_classes = 5

        self.img_dir = osp.join(self.data_root, "eyepacs_all_ims")

        self.load_list()

    def load_list(self):
        split_file = osp.join(self.data_root, "{}_eyepacs.csv".format(self.split))
        self.img_names = []
        self.labels = []
        with open(split_file, "r") as f:
            f.readline()  # skip the head line
            for line in f:
                name, label = line.strip().split(",")
                self.img_names.append(name)
                self.labels.append(int(label))

    def load_img(self, ind: int) -> np.ndarray:
        img_path = osp.join(self.img_dir, self.img_names[ind])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, ind) -> List[Any]:
        img = self.load_img(ind)
        label = self.labels[ind]

        if self.transformer is not None:
            result = self.transformer(image=img)
            img = result["image"]

        ret = [img, label]

        if self.return_id:
            ret.append(self.img_names[ind])

        return ret

    def __len__(self) -> int:
        return len(self.img_names)

    def __repr__(self) -> str:
        return (
            "EyePacs(data_root={}, split={})\tSamples : {}".format(
                self.data_root, self.split, self.__len__()
            )
        )


def data_transformation(is_train: bool = True):
    if is_train:
        transformer = A.Compose([
            A.Resize(width=512, height=512),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        transformer = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(),
            ToTensorV2()
        ])
    return transformer


def get_data_loader(
    data_root: str,
    batch_size: int = 8,
    split: str = "train",
    num_workers: int = 8,
    return_id: bool = False,

):
    assert split in [
        "train", "val", "test",
    ], "Split '{}' not supported".format(split)

    data_transformer = data_transformation(is_train=(split == "train"))
    dataset = EyePacsDataset(
        data_root=data_root,
        split=split,
        transformer=data_transformer,
        return_id=return_id,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=(split == "train")
    )

    return data_loader
