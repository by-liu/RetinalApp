import os.path as osp
from shutil import copyfile
import time
import json
import pprint
import logging
import torch
from yacs.config import CfgNode as CN
import wandb
import torch.nn.functional as F
from terminaltables.ascii_table import AsciiTable

from retinal.engine.trainer import DefaultTrainer
from retinal.config import convert_cfg_to_dict
from retinal.modeling import CompoundLoss
from retinal.solver import get_lr
from retinal.evaluation import MultiClassEvaluator, AverageMeter, LossMeter
from retinal.utils import round_dict

logger = logging.getLogger(__name__)


class DRTrainer(DefaultTrainer):
    def __init__(self, cfg: CN):
        self.cfg = cfg
        logger.info("DRTrainer with config : ")
        logger.info(pprint.pformat(cfg))
        self.device = torch.device(cfg.DEVICE)
        self.build_model()
        self.build_dataloader()
        self.build_solver()
        self.build_meter()
        self.init_wandb_or_not()

    def build_meter(self):
        self.num_classes = self.train_loader.dataset.num_classes
        self.evaluator = MultiClassEvaluator(
            num_classes=self.num_classes
        )
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_meter = LossMeter(self.loss_func.num_terms)

    def init_wandb_or_not(self):
        if self.cfg.WANDB.ENABLE:
            wandb.init(
                project=self.cfg.WANDB.PROJECT,
                entity=self.cfg.WANDB.ENTITY,
                config=convert_cfg_to_dict(self.cfg),
                tags=["train"]
            )
            wandb.run.name = "{}-{}-{}".format(
                wandb.run.id, self.cfg.MODEL.ARCH, self.cfg.LOSS.NAME
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

    def log_iter_info(self, iter, max_iter, epoch, phase="Train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_vals())
        log_dict.update(self.evaluator.curr_score())
        logger.info("{} Iter[{}/{}][{}]\t{}".format(
            phase, iter + 1, max_iter, epoch + 1,
            json.dumps(round_dict(log_dict))
        ))
        if self.cfg.WANDB.ENABLE and phase.lower() == "train":
            wandb_log_dict = {"iter": epoch * max_iter + iter}
            wandb_log_dict.update(dict(
                ("{}/Iter/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_epoch_info(self, epoch, phase="Train"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, CompoundLoss):
            log_dict["alpha"] = self.loss_func.alpha
        metric, table_data = self.evaluator.mean_score(print=False)
        log_dict.update(metric)
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if phase.lower() != "train":
            logger.info("\n" + AsciiTable(table_data).table)
        if self.cfg.WANDB.ENABLE:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            if phase.lower() != "train":
                wandb_log_dict["{}/score_table".format(phase)] = wandb.Table(
                    columns=table_data[0], data=table_data[1:]
                )
            wandb.log(wandb_log_dict)

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
            inputs, labels = samples[0].to(self.device), samples[1].to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
                # For compounding loss, make sure the first term represents the overall loss
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            predicts = F.softmax(outputs, dim=1)
            # pred_labels = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                predicts.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_epoch_info(epoch)

    @torch.no_grad()
    def eval_epoch(
        self, data_loader, epoch,
        phase="Val",
    ):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)
        end = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                predicts.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()
        self.log_epoch_info(epoch, phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score(all_metric=False)[0]

    def train(self):
        self.start_or_resume()
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

    def wandb_best_model_or_not(self):
        if self.cfg.WANDB.ENABLE:
            copyfile(
                osp.join(self.cfg.OUTPUT_DIR, "model", "best.pth"),
                osp.join(self.cfg.OUTPUT_DIR, "model", "{}-best.pth".format(wandb.run.name)),
            )