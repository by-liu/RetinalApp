import os.path as osp
import numpy as np
from shutil import copyfile
from typing import Optional
import time
import json
import logging
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch.nn.functional as F
from terminaltables.ascii_table import AsciiTable
import wandb

import retinal.data.test_augment as ta
from retinal.modeling import CompoundLoss
from retinal.evaluation import MultiClassEvaluator, AverageMeter, LossMeter
from retinal.utils import round_dict, get_lr
from retinal.utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


class DRTester:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.device = torch.device(self.cfg.device)
        self.build_data_loader()
        self.build_model(self.cfg.test.checkpoint)
        self.build_meter()
        self.init_wandb_or_not()

    def init_wandb_or_not(self):
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=self.cfg.wandb.tags.split(","),
            )
            wandb.run.name = "{}-test-{}-{}".format(
                wandb.run.id, self.cfg.model.name, self.cfg.data.name
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

    def build_data_loader(self) -> None:
        # data pipeline
        self.test_loader = instantiate(self.cfg.data.object.test_ta)

    def build_model(self, checkpoint: Optional[str] = "") -> None:
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        logger.info("Model initialized")
        self.checkpoint_path = osp.join(
            self.work_dir, "best.pth" if checkpoint == "" else checkpoint
        )
        load_checkpoint(self.checkpoint_path, self.model, self.device)

    def build_meter(self):
        if self.cfg.data.name != "folder":
            self.num_classes = self.test_loader.num_classes
            self.evaluator = MultiClassEvaluator(
                num_classes=self.num_classes
            )
        self.batch_time_meter = AverageMeter()
        logger.info("Meters initialized")

    def reset_meter(self):
        if self.cfg.data.name != "folder":
            self.evaluator.reset()
        self.batch_time_meter.reset()

    def log_iter_info(self, iter, max_iter):
        log_dict = {}
        log_dict["batch_time"] = self.batch_time_meter.val
        if self.cfg.data.name != "folder":
            log_dict.update(self.evaluator.curr_score())
        logger.info(
            "Test iter[{}/{}]\t{}".format(
                iter + 1, max_iter, json.dumps(round_dict(log_dict))
            )
        )

    def log_epoch_info(self):
        log_dict = {}
        if self.cfg.data.name != "folder":
            log_dict["samples"] = self.evaluator.num_samples()
            metric, table_data = self.evaluator.mean_score(print=False)
            log_dict.update(metric)
            logger.info("\n" + AsciiTable(table_data).table)
        logger.info("Test Epoch\t{}".format(
            json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable:
            wandb_log_dict = dict(
                ("Test/{}".format(key), value) for (key, value) in log_dict.items()
            )
            wandb_log_dict["Test/score_table"] = wandb.Table(
                columns=table_data[0], data=table_data[1:]
            )
            wandb_log_dict["Test/conf_mat"] = wandb.sklearn.plot_confusion_matrix(
                self.evaluator.labels,
                self.evaluator.pred_labels,
                self.evaluator.classes,
            )
            wandb.log(wandb_log_dict)

    @torch.no_grad()
    def test_epoch(self):
        self.reset_meter()
        self.model.eval()

        max_iter = len(self.test_loader)
        end = time.time()

        if self.cfg.test.save_prediction:
            fsave = open(osp.join(self.work_dir, "predicts.csv"), "w")

        for i, samples in enumerate(self.test_loader):
            # import ipdb; ipdb.set_trace()
            # img = ta.preprocess(samples[0])
            # image from a folder
            img = samples[0]  # 3 x 512 x 512 # numpy ? 
            inputs = ta.augment(
                img,
                self.cfg.test.augment
            )   # [3 x 512 x512, 3 x512x 512, ...] 4 augmented images
            if isinstance(inputs, list):
                outputs = [
                    self.model(torch.from_numpy(x).to(self.device)) for x in inputs
                ] # [6 x 1, 6x1, ....]
                outputs = torch.cat(outputs, dim=0)  # 6 x 4 
            else:
                inputs = torch.from_numpy(inputs).to(self.device)
                outputs = self.model(inputs)
            label = samples[1]

            predicts = F.softmax(outputs, dim=1)
            predicts = ta.fuse_predicts(predicts, reduce=self.cfg.test.augment.fuse)
            pred_label = torch.argmax(predicts)

            if self.cfg.data.name != "folder":
                self.evaluator.update(
                    np.expand_dims(predicts.detach().cpu().numpy(), axis=0),
                    np.expand_dims(label, axis=0),
                )

            if self.cfg.test.save_prediction:
                fsave.write("{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(
                    osp.splitext(samples[2])[0],
                    pred_label,
                    predicts.max(),
                    predicts[0], predicts[1], predicts[2], predicts[3], predicts[4], 
                ))

            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter)
            end = time.time()
        self.log_epoch_info()
        if self.cfg.test.save_prediction:
            fsave.close()

    def test(self):
        self.test_epoch()
