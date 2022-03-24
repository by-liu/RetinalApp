import os.path as osp
import torch
import torch.nn as nn
import logging
from typing import Optional, Any
import segmentation_models_pytorch as smp
import timm

from retinal.config.registry import Registry
from retinal.utils.checkpoint import load_checkpoint
from .unet import UNet

logger = logging.getLogger(__name__)

ACT_REGISTRY = Registry("activation")
ACT_REGISTRY.register("softmax", nn.Softmax(dim=1))
ACT_REGISTRY.register("sigmoid", nn.Sigmoid())
ACT_REGISTRY.register("relu", nn.ReLU())
ACT_REGISTRY.register("identity", nn.Identity())

NETWORK_REGISTRY = Registry("model")


@NETWORK_REGISTRY.register("Fpn")
def Fpn(cfg) -> smp.FPN:
    model = smp.FPN(
        encoder_name=cfg.MODEL.ENCODER,
        encoder_weights=cfg.MODEL.ENCODER_WEIGHTS if cfg.MODEL.ENCODER_WEIGHTS else None,
        in_channels=cfg.MODEL.INPUT_CHANNELS,
        classes=cfg.MODEL.NUM_CLASSES
    )
    setattr(model, "act", ACT_REGISTRY.get(cfg.MODEL.ACT_FUNC))
    return model


@NETWORK_REGISTRY.register("FpnCls")
def FpnWithClsHead(cfg) -> smp.FPN:
    model = smp.FPN(
        encoder_name=cfg.MODEL.ENCODER,
        encoder_weights=cfg.MODEL.ENCODER_WEIGHTS if cfg.MODEL.ENCODER_WEIGHTS else None,
        in_channels=cfg.MODEL.INPUT_CHANNELS,
        classes=cfg.MODEL.NUM_CLASSES,
        aux_params={
            "classes": cfg.MODEL.NUM_CLASSES,
            "pooling": "avg",
            "dropout": cfg.MODEL.DROPOUT
        }
    )
    setattr(model, "act", ACT_REGISTRY.get(cfg.MODEL.ACT_FUNC))
    return model


@NETWORK_REGISTRY.register("NaiveUnet")
def NaviUnet(cfg) -> UNet:
    model = UNet(
        n_channels=cfg.MODEL.INPUT_CHANNELS,
        n_classes=cfg.MODEL.NUM_CLASSES,
        bilinear=True
    )
    setattr(model, "act", ACT_REGISTRY.get(cfg.MODEL.ACT_FUNC))
    return model


@NETWORK_REGISTRY.register("Unet")
def Unet(cfg) -> smp.Unet:
    model = smp.Unet(
        encoder_name=cfg.MODEL.ENCODER,
        encoder_weights=cfg.MODEL.ENCODER_WEIGHTS if cfg.MODEL.ENCODER_WEIGHTS else None,
        in_channels=cfg.MODEL.INPUT_CHANNELS,
        classes=cfg.MODEL.NUM_CLASSES
    )
    setattr(model, "act", ACT_REGISTRY.get(cfg.MODEL.ACT_FUNC))
    return model


@NETWORK_REGISTRY.register("DeepLabV3Plus")
def DeepLabV3Plus(cfg) -> smp.DeepLabV3Plus:
    model = smp.DeepLabV3Plus(
        encoder_name=cfg.MODEL.ENCODER,
        encoder_weights=cfg.MODEL.ENCODER_WEIGHTS if cfg.MODEL.ENCODER_WEIGHTS else None,
        in_channels=cfg.MODEL.INPUT_CHANNELS,
        classes=cfg.MODEL.NUM_CLASSES
    )
    setattr(model, "act", ACT_REGISTRY.get(cfg.MODEL.ACT_FUNC))
    return model


@NETWORK_REGISTRY.register("timm")
def create_timm_model(cfg) -> Any:
    model = timm.create_model(
        model_name=cfg.MODEL.ENCODER,
        pretrained=True,
        num_classes=cfg.MODEL.NUM_CLASSES,
        drop_rate=0.5,
    )

    return model


def build_model(cfg, model_path : Optional[str] = None):
    """
    Builds the segmentation model.
    Args:
        cfg : configs to build the backbone. Detains can ben seen in configs/defaults.py
    """

    arch = cfg.MODEL.ARCH
    logger.info("Construct model : {}".format(arch))
    model = NETWORK_REGISTRY.get(arch)(cfg)

    if model_path:
        load_checkpoint(model_path, model, cfg.DEVICE)
        logger.info("Successfully load weights from {}".format(model_path))
    elif cfg.MODEL.PRETRAINED:
        load_checkpoint(cfg.MODEL.PRETRAINED, model, cfg.DEVICE)
        logger.info("Successfully load weights from {}".format(cfg.MODEL.PRETRAINED))

    return model
