import torch
from torch.utils.data import DataLoader
import os.path as osp
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Optional, Callable, List, Any
from torch.utils.data.dataset import Dataset

from .eyepacs import data_transformation


class Diagnos(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "test",
        transformer: Optional[Callable] = None,
        return_id: bool = False,
    ):
        self.data_root = data_root
        self.split = split
        self.transformer = transformer
        self.return_id = return_id
        self.num_classes = 5
        self.img_dir = osp.join(self.data_root, "Images")
        self.load_list()

    def load_list(self):
        split_file = osp.join(self.data_root, self.split + ".csv")
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
            "Diagnos(data_root={}, split={})\tSamples : {}".format(
                self.data_root, self.split, self.__len__()
            )
        )


def get_dataset(
    data_root: str,
    split: str = "test",
    return_id: bool = False,

):
    assert split in [
        "test",
    ], "Split '{}' not supported".format(split)

    transformer = A.Compose([
        A.Normalize(),
    ])
    dataset = Diagnos(
        data_root=data_root,
        split=split,
        transformer=transformer,
        return_id=return_id,
    )

    return dataset
