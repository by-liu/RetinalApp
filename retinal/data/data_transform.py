"""
File: data_transform.py
Author: Bingyuan Liu (bingyuan.liu@etsmtl.ca)
Date: Dec 15, 2020
Brief: data transform wrapper
"""

from albumentations.augmentations.transforms import HorizontalFlip, RandomBrightness, RandomContrast, RandomGamma
from albumentations.core.composition import OneOf
from yacs.config import CfgNode
import logging
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import retinal.data.paired_transforms_tv04 as p_tr
from ..config.registry import Registry

DATA_TRANSFORM = Registry("data_transform")

logger = logging.getLogger(__name__)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def retinal_lesion2(cfg: CfgNode, is_train: bool = True) -> A.Compose:
    height, width = cfg.DATA.RESIZE
    if is_train:
        transformer = A.Compose([
            A.OneOf([
                A.Sequential([
                    A.SmallestMaxSize(max_size=640),
                    A.RandomCrop(height=512, width=512),
                ]),
                A.Resize(512, 512),
            ], p=1),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
            # A.HueSaturationValue(),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.LongestMaxSize(max_size=896),
            # A.PadIfNeeded(
            #     min_height=640, min_width=640,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     value=(0, 0, 0), mask_value=[255] * 9, 
            # ),
            A.Normalize(),
            ToTensorV2()
        ])

    return transformer


@DATA_TRANSFORM.register("retinal-lesions")
@DATA_TRANSFORM.register("FGADR")
def retinal_lesion(cfg: CfgNode, is_train: bool = True) -> A.Compose:
    height, width = cfg.DATA.RESIZE
    if is_train:
        transformer = A.Compose([
            A.HorizontalFlip(),
            A.OneOf([
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness(),
                ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
            A.ShiftScaleRotate(),
            A.Resize(height, width, always_apply=True),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.Resize(height, width, always_apply=True),
            A.Normalize(),
            ToTensorV2()
        ])

    return transformer


@DATA_TRANSFORM.register("eyepacs")
def img_augment(cfg: CfgNode, is_train: bool = True):
    if is_train:
        transformer = A.Compose([
            # A.SmallestMaxSize(max_size=448),
            # A.RandomScale(scale_limit=[0.5, 1], p=1),
            # A.RandomCrop(height=224, width=224),
            # A.OneOf([
            #     A.RandomContrast(),
            #     A.RandomGamma(),
            #     A.RandomBrightness(),
            #     ], p=0.5),
            # A.HorizontalFlip(),
            A.Resize(width=760, height=760),
            A.RandomCrop(height=728, width=728),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.Blur(p=0.3),
            # A.CLAHE(p=0.3),
            # A.ColorJitter(p=0.3),
            # A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
            # A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            # A.SmallestMaxSize(max_size=256),
            # A.CenterCrop(height=224, width=224),
            A.Resize(height=728, width=728),
            A.Normalize(),
            ToTensorV2()
        ])
    return transformer

# @DATA_TRANSFORM.register("retinal-lesions")
# def retinal_lesion(cfg : CfgNode, is_train : bool = True) -> p_tr.Compose:
#     normalize = p_tr.Normalize(
#         mean=cfg.DATA.MEAN, std=cfg.DATA.STD
#     )
#     if is_train:
#         transformer = p_tr.Compose([
#             p_tr.Resize(cfg.DATA.RESIZE),
#             p_tr.RandomChoice([
#                 p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20)),
#                 p_tr.RandomAffine(degrees=0, translate=(0.05, 0)),
#                 p_tr.RandomRotation(degrees=45, fill=(0, 0, 0), fill_tg=(0,))
#             ]),
#             p_tr.RandomHorizontalFlip(),
#             p_tr.RandomVerticalFlip(),
#             p_tr.ToTensor(),
#             normalize
#         ])
#     else:
#         transformer = p_tr.Compose([
#             p_tr.Resize(cfg.DATA.RESIZE),
#             p_tr.ToTensor(),
#             normalize,
#         ])

#     return transformer


@DATA_TRANSFORM.register("cityscapes")
def cityscapes(cfg : CfgNode, is_train : bool = True) -> A.Compose:
    height, width = cfg.DATA.RESIZE
    if is_train:
        transformer = A.Compose([
            A.OneOrOther(
                A.Resize(height, width, interpolation=cv2.INTER_CUBIC),
                A.Sequential([
                    A.RandomScale([0.5, 1.5], interpolation=cv2.INTER_CUBIC, p=1.),
                    A.RandomCrop(height, width),
                ], p=1),
                p=0.3
            ),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(),
            # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transformer = A.Compose([
            A.Resize(1024, 2048),
            A.Normalize(),
            ToTensorV2()
        ])

    return transformer


@DATA_TRANSFORM.register("image-folder")
def image_folder(cfg : CfgNode, **kwargs) -> p_tr.Compose:
    normalize = p_tr.Normalize(
        mean=cfg.DATA.MEAN, std=cfg.DATA.STD
    )
    transformer = p_tr.Compose([
        p_tr.Resize(cfg.DATA.RESIZE),
        p_tr.ToTensor(),
        normalize,
    ])

    return transformer


def build_image_transform(cfg : CfgNode, is_train : bool = True) -> A.Compose:
    transformer = DATA_TRANSFORM.get(cfg.DATA.NAME)(cfg, is_train=is_train)
    logger.info("Successfully build image tranform : {}".format(transformer))

    return transformer
