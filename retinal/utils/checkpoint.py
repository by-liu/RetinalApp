import os
import os.path as osp
import logging
import torch
from typing import Optional, Tuple

from .file_io import mkdir, load_list

logger = logging.getLogger(__name__)


def save_train_checkpoint(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scheduler: Optional[object] = None,
    best_checkpoint: bool = False,
    val_score: Optional[float] = None,
    keep_checkpoint_num: int = 1,
    keep_checkpoint_interval: int = 0
) -> None:
    mkdir(save_dir)
    # model_name = "checkpoint_epoch_{}.pth".format(epoch + 1)
    # model_path = osp.join(save_dir, model_name)
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler:
        state["scheduler"] = scheduler.state_dict()
    if val_score:
        state["val_score"] = val_score
    torch.save(state, osp.join(save_dir, "last.pth"))
    if best_checkpoint:
        torch.save(state, osp.join(save_dir, "best.pth"))

    if keep_checkpoint_num > 1:
        "Keep more checkpoints than the last.pth"
        torch.save(state, osp.join(save_dir, "epoch_{}.pth".format(epoch + 1)))
        remove_file = osp.join(save_dir, "epoch_{}.pth".format(epoch + 1 - keep_checkpoint_num))
        if osp.exists(remove_file):
            os.remove(remove_file)

    if keep_checkpoint_interval > 0:
        "Keep the checkpoints for every interval epochs"
        if (epoch + 1) % keep_checkpoint_interval == 0:
            torch.save(
                state, osp.join(save_dir, "epoch_{}.pth".format(epoch + 1))
            )


def save_checkpoint(
    model: torch.nn.Module,
    save_dir: str,
    model_name: str,
) -> None:
    mkdir(save_dir)
    state = {"state_dict": model.state_dict()}
    torch.save(state, osp.join(save_dir, model_name))


def load_checkpoint(model_path: str, model: torch.nn.Module, device) -> None:
    if not osp.exists(model_path):
        raise FileNotFoundError(
            "Model not found : {}".format(model_path)
        )
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    logger.info("Succeed to load weights from {}".format(model_path))
    if missing_keys:
        logger.warn("Missing keys : {}".format(missing_keys))
    if unexpected_keys:
        logger.warn("Unexpected keys : {}".format(unexpected_keys))


def load_train_checkpoint(
    work_dir: str,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None
) -> Tuple:
    """
    Returns:
        Tuple: [epoch, best_epoch, best_score]
    """

    try:
        last_checkpoint_path = osp.join(work_dir, "last.pth")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("Succeed to load train info from {}".format(last_checkpoint_path))

        best_checkpoint_path = osp.join(work_dir, "best.pth")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        best_epoch = checkpoint["epoch"]
        best_score = checkpoint["val_score"] if "val_score" in checkpoint else None
        return epoch + 1, best_epoch, best_score
    except Exception:
        return 0, -1, None
