import torch
import os.path as osp
import cv2
import numpy as np
import albumentations as A
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
        if split == "test":
            split = "val"
        self.data_root = data_root
        self.split = split
        self.transformer = transformer
        self.return_id = return_id

        self.num_classes = 5

        if self.split != "test":
            self.img_dir = osp.join(self.data_root, "train")
        else:
            self.img_dir = osp.join(self.data_root, "test")
        self.load_list()

    def load_list(self):
        split_file = osp.join(self.data_root, "{}.txt".format(self.split))
        self.img_names = []
        self.labels = []
        with open(split_file, "r") as f:
            for line in f:
                name, label = line.strip().split(",")
                self.img_names.append(name)
                self.labels.append(int(label))

    def load_img(self, ind: int) -> np.ndarray:
        img_path = osp.join(self.img_dir, self.img_names[ind] + ".jpeg")
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
