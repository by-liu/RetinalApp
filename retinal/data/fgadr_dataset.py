import torch
import os.path as osp
import cv2
import numpy as np
import albumentations as A
from typing import List, Optional
from torch.utils.data.dataset import Dataset

from retinal.utils import load_list


class FGADRDataset(Dataset):
    """
    Wrapper for FGADR datset
    """
    def __init__(self, data_root: str,
                 samples_path: str,
                 classes_path: str,
                 data_transformer: Optional[A.Compose] = None,
                 return_dr_grade: bool = False,
                 return_id: bool = False) -> None:
        self.data_root = data_root
        self.samples_path = samples_path
        self.classes_path = classes_path
        self.data_transformer = data_transformer
        self.return_id = return_id
        self.return_dr_grade = return_dr_grade
        self.samples: List[str] = load_list(self.samples_path)
        self.classes: List[str] = load_list(self.classes_path)
        if self.return_dr_grade:
            self.load_dr_grade()

    def load_dr_grade(self) -> None:
        self.dr_grade = {}
        path = osp.join(self.data_root, "DR_Seg_Grading_Label.csv")
        with open(path, "r") as f:
            for line in f:
                sample_name, label = line.strip().split(",")
                label = int(label)
                self.dr_grade[sample_name] = label

    def get_target(self, img_shape: np.ndarray, sample_name: str) -> np.ndarray:
        target = np.zeros((img_shape[0], img_shape[1], len(self.classes)), dtype=np.uint8)
        for i, class_name in enumerate(self.classes):
            mask_dir = osp.join(self.data_root, "{}_Masks".format(class_name))
            mask_path = osp.join(mask_dir, sample_name)
            if osp.exists(mask_path):
                img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = np.zeros_like(img)
                mask[np.where(img > 125)] = 1
                target[:, :, i] = mask

        return target

    def __getitem__(self, index: int) -> List:
        sample_name = self.samples[index]
        img = cv2.imread(osp.join(self.data_root, "Original_Images", sample_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.get_target(img.shape, sample_name)

        if self.data_transformer is not None:
            result = self.data_transformer(image=img, mask=mask)
            img = result["image"]
            mask = result["mask"].long()
            mask = torch.einsum("ijk->kij", mask)

        ret = [img, mask]

        if self.return_dr_grade:
            grade = self.dr_grade[sample_name]
            ret.append(grade)

        if self.return_id:
            ret.append(sample_name)

        return ret

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            "FGADRDataset(data_root={}, samples_path={})\tSamples : {}".format(
                self.data_root, self.samples_path, self.__len__()
            )
        )
