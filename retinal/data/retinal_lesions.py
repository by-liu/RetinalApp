import torch
import os.path as osp
import cv2
import numpy as np
import albumentations as A
from typing import List, Optional
from torch.utils.data.dataset import Dataset

from retinal.utils import load_list


class RetinalLesions(Dataset):
    def __init__(
        self,
        data_root: str,
        samples_path: str,
        data_transformer: Optional[A.Compose] = None,
        return_classes: bool = False,
        return_id: bool = False
    ) -> None:
        self.data_root = data_root
        self.samples_path = samples_path
        self.data_transformer = data_transformer
        self.return_classes = return_classes
        self.return_id = return_id

        self.abbrev_classes = [
            "IRMA", "CWS", "HaEx", "MA",
            "NV", "pHE", "rHE", "vHE", "FP"
        ]
        self.classes = [
            "IRMA", "cotton_wool_spots", "hard_exudate",
            "microaneurysm", "neovascularization", "preretinal_hemorrhage",
            "retinal_hemorrhage", "vitreous_hemorrhage", "fibrous_proliferation",
        ]
        self.samples = load_list(self.samples_path)

    def get_target(self, img_shape: np.ndarray, sample_name: str) -> np.ndarray:
        sample_name = osp.splitext(sample_name)[0]
        target = np.zeros((img_shape[0], img_shape[1], len(self.classes)), dtype=np.uint8)
        label = torch.zeros(len(self.classes))
        for i, class_name in enumerate(self.classes):
            mask_dir = osp.join(self.data_root, "masks",  class_name)
            mask_path = osp.join(mask_dir, sample_name + ".png")
            if osp.exists(mask_path):
                img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = np.zeros_like(img)
                mask[np.where(img > 125)] = 1
                if np.any(mask):
                    target[:, :, i] = mask
                    label[i] = 1
        return target, label

    def __getitem__(self, index: int) -> List:
        sample_name = self.samples[index]
        img = cv2.imread(
            osp.join(self.data_root, "images", sample_name)
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask, label = self.get_target(img.shape, sample_name)

        if self.data_transformer is not None:
            result = self.data_transformer(image=img, mask=mask)
            img = result["image"]
            mask = result["mask"].long()
            mask = torch.einsum("ijk->kij", mask)

        ret = [img, mask]

        if self.return_classes:
            ret.append(label)

        if self.return_id:
            ret.append(sample_name)

        return ret

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            "RetinalLesions(data_root={}, samples_path={})\tSamples : {}".format(
                self.data_root, self.samples_path, self.__len__()
            )
        )