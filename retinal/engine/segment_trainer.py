import os.path as osp
from retinal.engine.trainer import DefaultTrainer
import time
import pprint
import logging
import torch
from yacs.config import CfgNode as CN
import wandb

from retinal.config import convert_cfg_to_dict
from retinal.modeling import build_model, get_loss_func, CompoundLoss
from retinal.solver import build_optimizer, build_scheduler, get_lr
from retinal.evaluation import SegmentationEvaluator, AverageMeter, LossMeter
from retinal.data import build_data_pipeline
from retinal.utils import (
    TensorboardWriter, load_train_checkpoint, save_checkpoint, load_checkpoint,
    load_list
)

logger = logging.getLogger(__name__)


class SegmentTrainer(DefaultTrainer):
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
        # if self.cfg.DATA.CLASSES_PATH:
        #     self.classes = load_list(self.cfg.DATA.CLASSES_PATH)
        # else:
        #     self.classes = [str(i) for i in range(self.cfg.MODEL.NUM_CLASSES)]
        self.classes = self.train_loader.dataset.classes_abbrev
        self.evaluator = SegmentationEvaluator(
            classes=self.classes,
            include_background=True
        )
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_meter = LossMeter(self.loss_func.num_terms)

    def init_wandb_or_not(self):
        if self.cfg.WANDB.ENABLE:
            wandb.init(
                project=self.cfg.WANDB.PROJECT,
                entity=self.cfg.WANDB.ENTITY,
                config=convert_cfg_to_dict(self.cfg)
            )
            wandb.watch(self.model, log_freq=1000)

    def wandb_iter_info_or_not(
        self, iter, max_iter, epoch, phase="train",
        loss_meter=None, score=None, lr=None
    ):
        if not self.cfg.WANDB.ENABLE:
            return
        if loss_meter is not None:
            loss_dict = loss_meter.get_vals()
            for key, val in loss_dict.items():
                wandb.log("{}/Iter/{}".format(phase, key), val)
        if score is not None:
            wandb.log("{}/Iter/{}".format(phase, self.evaluator.main_metric()), score)
        if lr is not None:
            wandb.log("{}/Iter/lr".format(phase), lr)

    def wandb_epoch_info_or_not(
        self, epoch, phase="train", evaluator=None,
        loss_meter=None
    ):
        if not self.cfg.NEPTUNE.ENABLE:
            return
        if loss_meter is not None:
            loss_dict = loss_meter.get_vals()
            for key, val in loss_dict.items():
                wandb.log("{}/Epoch/{}".format(phase, key), val)
        if isinstance(self.loss_func, CompoundLoss):
            wandb.log("{}/Epoch/alpha".format(phase), self.loss_func.alpha)
        if evaluator is not None:
            wandb.log(
                "{}/Epoch/{}".format(phase, evaluator.main_metric()), evaluator.mean_score()
            )

    def wandb_best_model_or_not(self):
        if self.cfg.WANDB.ENABLE:
            epoch = self.best_epoch if self.cfg.TEST.BEST_CHECKPOINT else self.cfg.SOLVER.MAX_EPOCH
            model_path = osp.join(
                self.cfg.OUTPUT_DIR, "model", "checkpoint_epoch_{}.pth".format(epoch)
            )
            wandb.save(model_path)

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)
        lr = get_lr(self.optimizer)

        end = time.time()
        logger.info("====== Start training epoch {} ======".format(epoch + 1))
        for i, samples in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            inputs, labels = samples[0].to(self.device), samples[1].to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels.type_as(outputs))
            if isinstance(loss, tuple):
                # For compounding loss, make sure the first term represents the overall loss
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # Adjusting LR by iteration if poly is used
            if self.scheduler.name == "poly":
                self.scheduler.step(epoch=epoch)
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            predicts = self.model.act(outputs)
            pred_label = (predicts > self.cfg.THRES).float()
            score = self.evaluator.update(pred_label.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter, epoch,
                                   phase="train",
                                   data_time_meter=self.data_time_meter,
                                   batch_time_meter=self.batch_time_meter,
                                   loss_meter=self.loss_meter,
                                   score=score, lr=lr)
                self.wandb_iter_info_or_not(
                    i, max_iter, epoch,
                    phase="train",
                    loss_meter=self.loss_meter,
                    score=score,
                    lr=lr
                )
            end = time.time()
        self.log_epoch_info(epoch,
                            phase="train",
                            evaluator=self.evaluator,
                            loss_meter=self.loss_meter)
        self.wandb_epoch_info_or_not(
            epoch, phase="train",
            evaluator=self.evaluator,
            loss_meter=self.loss_meter
        )
        logger.info("====== Complete training epoch {} ======".format(epoch + 1))

    @torch.no_grad()
    def eval_epoch(self, data_loader, epoch, phase="val"):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)
        end = time.time()
        for i, samples in enumerate(data_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            # decouple samples
            inputs, labels = samples[0].to(self.device), samples[1].to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels.type_as(outputs))
            # metric
            self.loss_meter.update(loss)
            predicts = self.model.act(outputs)
            pred_label = (predicts > self.cfg.THRES).float()
            score = self.evaluator.update(pred_label.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter, epoch,
                                   phase=phase,
                                   data_time_meter=self.data_time_meter,
                                   batch_time_meter=self.batch_time_meter,
                                   loss_meter=self.loss_meter,
                                   score=score)
                self.wandb_iter_info_or_not(
                    i, max_iter, epoch, phase=phase, loss_meter=self.loss_meter, score=score
                )
            end = time.time()
        self.log_epoch_info(epoch,
                            phase=phase,
                            evaluator=self.evaluator,
                            loss_meter=self.loss_meter)
        self.wandb_epoch_info_or_not(
            epoch, phase=phase, evaluator=self.evaluator,
            loss_meter=self.loss_meter
        )

        return self.loss_meter.avg(0), self.evaluator.mean_score()

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
