import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os.path as osp
from yacs.config import CfgNode as CN
import albumentations as A
from typing import Callable

from ..config.registry import Registry
from .retinal_lesions import RetinalLesions
from .fgadr_dataset import FGADRDataset
from .cityscapes import CityscapesDataset
from .image_folder import ImageFolder
from .data_transform import build_image_transform

logger = logging.getLogger(__name__)

DATASET_REGISTRY = Registry("dataset")


@DATASET_REGISTRY.register("cityscapes")
def cityscapes(cfg : CN, data_transform : A.Compose, split : str = "train") -> Dataset:
    dataset = CityscapesDataset(
        data_root=cfg.DATA.DATA_ROOT,
        split=split,
        transforms=data_transform,
        return_id=True if split in ("val", "test") else False
    )
    return dataset


@DATASET_REGISTRY.register("retinal-lesions")
def retinal_lesions(cfg : CN, data_transform : A.Compose, split : str = "train") -> Dataset:
    data_path = cfg[split.upper()]["DATA_PATH"]
    if not data_path:
        data_path = "{}.txt".format(split)

    data_root = cfg.DATA.DATA_ROOT
    dataset = RetinalLesions(
        data_root,
        osp.join(data_root, data_path),
        data_transform,
        return_id=True if split != "train" else False
    )

    logger.info("Successfully build dataset : {}".format(dataset))

    return dataset


@DATASET_REGISTRY.register("FGADR")
def FGADR(cfg: CN, data_transform: Callable, split: str = "train") -> Dataset:
    data_root = cfg.DATA.DATA_ROOT
    samples_path = cfg[split.upper()]["DATA_PATH"]
    samples_path = osp.join(data_root, samples_path)
    classes_path = osp.join(data_root, "classes.txt")

    dataset = FGADRDataset(
        data_root, samples_path, classes_path,
        data_transformer=data_transform,
        return_id=True if split == "test" else False
    )

    logger.info("Successfully build dataset : {}".format(dataset))

    return dataset


@DATASET_REGISTRY.register("image-folder")
def image_folder(cfg: CN, data_transform: Callable, **kwargs):
    data_root = cfg.DATA.DATA_ROOT
    dataset = ImageFolder(
        data_root,
        transforms=data_transform,
        return_id=True
    )

    return dataset


def build_data_pipeline(cfg : CN, split : str = "train") -> DataLoader:
    assert split in [
        "train", "val", "test",
    ], "Split '{}' not supported".format(split)

    data_transform = build_image_transform(cfg, is_train=(split == "train"))
    batch_size = cfg[split.upper()]["BATCH_SIZE"]
    dataset = DATASET_REGISTRY.get(cfg.DATA.NAME)(cfg, data_transform=data_transform, split=split)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        shuffle=(split == "train")
    )

    return data_loader


def build_dataset(cfg: CN, split: str = "train") -> Dataset:
    assert split in [
        "train", "val", "test",
    ], "Split '{}' not supported".format(split)

    data_transform = build_image_transform(cfg, is_train=(split == "train"))
    dataset = DATASET_REGISTRY.get(cfg.DATA.NAME)(cfg, data_transform, split)

    return dataset
