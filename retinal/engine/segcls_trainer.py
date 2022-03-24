# Training a segmentation model with classification head
import os.path as osp
import time
import pprint
import logging
import torch
from yacs.config import CfgNode as CN
import wandb

from retinal.engine.trainer import DefaultTrainer
from retinal.config import convert_cfg_to_dict
from retinal.modeling import CompoundLoss
from retinal.solver import get_lr
from retinal.evaluation import SegmentationEvaluator, MultilabelEvaluator, AverageMeter, LossMeter

logger = logging.getLogger(__name__)


class SegClsTrainer(DefaultTrainer):
    def __init__(self, cfg: CN):
        self.cfg = cfg
        logger.info("SegmentTrainer with config : ")
        logger.info(pprint.pformat(cfg))
        self.device = torch.device(cfg.DEVICE)
        self.build_model()
        self.build_dataloader()
        self.build_solver()
        self.build_meter()
        self.init_wandb_or_not()

    def build_meter(self):
        self.classes = self.train_loader.dataset.classes
        self.seg_evaluator = SegmentationEvaluator(
            classes=self.classes,
            include_background=True,
            ignore_index=225
        )
        self.cls_evaluator = MultilabelEvaluator(
            classes=self.classes,
            thres=self.cfg.THRES,
        )
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_meter = LossMeter(self.loss_func.num_terms)

    def reset_meter(self):
        self.seg_evaluator.reset()
        self.cls_evaluator.reset()
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()

    def init_wandb_or_not(self):
        if self.cfg.WANDB.ENABLE:
            wandb.init(
                project=self.cfg.WANDB.PROJECT,
                entity=self.cfg.WANDB.ENTITY,
                config=convert_cfg_to_dict(self.cfg),
                tags=["train"]
            )
            wandb.watch(self.model, log=None)

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        end = time.time()
        logger.info("====== Start training epoch {} ======".format(epoch + 1))
        for i, samples in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            inputs, masks, labels = (
                samples[0].to(self.device),
                samples[1].to(self.device),
                samples[2].to(self.device),
            )

        return super().train_epoch(epoch)

    def train(self):
        self.start_or_resume()
        # Perform the training loop
        logger.info("Start training ... ")
        for epoch in range(self.start_epoch, self.cfg.SOLVER.MAX_EPOCH):
            # train phase
            self.train_epoch(epoch)
            val_loss, val_score = self.val_epoch_or_not(epoch)
            self.save_checkpoint_or_not(epoch, val_score)
            if self.scheduler.name not in {"reduce_on_plateau", "poly"}:
                self.scheduler.step()
            if isinstance(self.loss_func, CompoundLoss):
                self.loss_func.adjust_alpha(epoch)

        logger.info("Complete training !")
        logger.info(
            ("Best performance on validation subset - "
             "model epoch {}, score {:.4f}").format(self.best_epoch + 1, self.best_score)
        )
        # Peform test phase if requried
        self.test_epoch_or_not()
        self.wandb_best_model_or_not()