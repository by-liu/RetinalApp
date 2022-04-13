import os.path as osp
from shutil import copyfile
import time
import json
import pprint
import logging
import torch
from yacs.config import CfgNode as CN
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
import torch.nn.functional as F
from terminaltables.ascii_table import AsciiTable

from retinal.config import convert_cfg_to_dict
from retinal.modeling import CompoundLoss
from retinal.solver import get_lr
from retinal.evaluation import MultiClassEvaluator, AverageMeter, LossMeter
from retinal.utils import round_dict
from retinal.utils.checkpoint import load_train_checkpoint, save_train_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class DRTrainer:
    def __init__(self, cfg: CN):
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.device = torch.device(self.cfg.device)
        self.build_data_loader()
        self.build_model()
        self.build_loss()
        self.build_solver()
        self.build_meter()
        self.init_wandb_or_not()

    def build_data_loader(self) -> None:
        self.train_loader = instantiate(self.cfg.data.object.train)
        self.val_loader = instantiate(self.cfg.data.object.val)
        logger.info("Data pipeline initialized for train and val")

    def build_model(self) -> None:
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        logger.info("Model initialized : {}".format(self.cfg.model.name))

    def build_loss(self) -> None:
        self.loss_func = instantiate(self.cfg.loss.object)
        self.loss_func.to(self.device)
        logger.info("Loss initialized : {}".format(self.loss_func))

    def build_solver(self) -> None:
        # build solver
        self.optimizer = instantiate(
            self.cfg.optim.object, self.model.parameters()
        )
        if self.cfg.scheduler.name == "one_cycle":
            self.scheduler = instantiate(
                self.cfg.scheduler.object, self.optimizer,
                steps_per_epoch=len(self.train_loader)
            )
            logger.info("LR schedulear : {}".format(self.scheduler))
        else:
            self.scheduler = instantiate(
                self.cfg.scheduler.object, self.optimizer
            )
        logger.info("Solver initialized")

    def build_meter(self):
        self.num_classes = self.train_loader.dataset.num_classes
        self.evaluator = MultiClassEvaluator(
            num_classes=self.num_classes
        )
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        if hasattr(self.loss_func, "names"):
            self.loss_meter = LossMeter(
                num_terms=len(self.loss_func.names),
                names=self.loss_func.names
            )
        else:
            self.loss_meter = LossMeter()

    def reset_meter(self):
        self.evaluator.reset()
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()

    def init_wandb_or_not(self):
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=self.cfg.wandb.tags,
            )
            wandb.run.name = "{}-{}-{}".format(
                wandb.run.id, self.cfg.model.name, self.cfg.loss.name
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

    def start_or_resume(self):
        if self.cfg.train.resume:
            self.start_epoch, self.best_epoch, self.best_score = (
                load_train_checkpoint(
                    self.work_dir, self.device, self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )
            )
        else:
            self.start_epoch, self.best_epoch, self.best_score = 0, -1, None
        self.max_epoch = self.cfg.train.max_epoch

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
        if self.cfg.wandb.enable and phase.lower() == "train":
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
        metric, table_data = self.evaluator.mean_score()
        log_dict.update(metric)
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if phase.lower() != "train":
            logger.info("\n" + AsciiTable(table_data).table)
        if self.cfg.wandb.enable:
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
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            # iteration-level lr scheduler
            if self.cfg.scheduler.name == "one_cycle":
                self.scheduler.step()
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
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()
        self.log_epoch_info(epoch, phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score(all_metric=False)[0]

    def train(self):
        self.start_or_resume()
        logger.info(
            "Everything is perfect so far. Let's start training. Good luck!"
        )
        for epoch in range(self.start_epoch, self.max_epoch):
            logger.info("=" * 20)
            logger.info(" Start epoch {}".format(epoch + 1))
            logger.info("=" * 20)
            # train phase
            self.train_epoch(epoch)
            val_loss, val_score = self.eval_epoch(self.val_loader, epoch, phase="Val")
            # epoch-level lr scheduler
            if self.cfg.scheduler.name not in ("one_cycle"):
                self.scheduler.step()
            # saveing checkpoint
            if self.best_score is None or val_score > self.best_score:
                self.best_score, self.best_epoch = val_score, epoch
                best_checkpoint = True
            else:
                best_checkpoint = False
            save_train_checkpoint(
                self.work_dir, self.model, self.optimizer, epoch,
                scheduler=self.scheduler,
                best_checkpoint=best_checkpoint,
                val_score=val_score,
                keep_checkpoint_num=self.cfg.train.keep_checkpoint_num,
                keep_checkpoint_interval=self.cfg.train.keep_checkpoint_interval
            )
            # logging best performance on val so far
            logger.info(
                "Epoch[{}]\tBest {} on Val : {:.4f} at epoch {}".format(
                    epoch + 1, self.evaluator.main_metric(),
                    self.best_score, self.best_epoch + 1
                )
            )
            if self.cfg.wandb.enable and best_checkpoint:
                wandb.log({
                    "epoch": epoch,
                    "Val/best_epoch": self.best_epoch,
                    "Val/best_{}".format(self.evaluator.main_metric()): self.best_score,
                    "Val/best_score_table": self.evaluator.wandb_score_table()
                })
        self.wandb_best_model_or_not()

    def wandb_best_model_or_not(self):
        if self.cfg.wandb.enable:
            copyfile(
                osp.join(self.work_dir, "best.pth"),
                osp.join(self.work_dir, "{}-best.pth".format(wandb.run.name))
            )
            artifact = wandb.Artifact(
                name="{}-{}".format(self.cfg.data.name, self.cfg.model.name),
                type="model",
            )
            artifact.add_file(osp.join(self.work_dir, "best.pth"))
            wandb.run.log_artifact(artifact)

    def test(self):
        logger.info("We are almost done : final testing ...")
        self.test_loader = instantiate(self.cfg.data.object.test)
        # test best pth
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(" Test at best epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Best epoch[{}] :".format(epoch + 1))
        load_checkpoint(
            osp.join(self.work_dir, "best.pth"), self.model, self.device
        )
        self.eval_epoch(self.test_loader, epoch, phase="Test")

    def run(self):
        self.train()
        self.test()
