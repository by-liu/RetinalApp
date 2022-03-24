import time
import pprint
import logging
import torch
import os.path as osp
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import json
from terminaltables.ascii_table import AsciiTable

from retinal.engine.tester import DefaultTester, ImageFolderTester
from retinal.evaluation import MultiClassEvaluator, AverageMeter
from retinal.data import build_data_pipeline
from retinal.utils import round_dict

logger = logging.getLogger(__name__)


class DRTester(DefaultTester):
    def __init__(self, cfg: CN):
        self.cfg = cfg
        logger.info("DRTeser with config : ")
        logger.info(pprint.pformat(self.cfg))
        self.device = torch.device(self.cfg.DEVICE)
        self.data_loader = build_data_pipeline(self.cfg, split="test")
        self.build_model()
        self.build_meter()

    def build_meter(self):
        self.num_classes = self.data_loader.dataset.num_classes
        self.evaluator = MultiClassEvaluator(
            num_classes=self.num_classes
        )
        self.batch_time_meter = AverageMeter()

    def log_iter_info(self, iter, max_iter, phase="Test"):
        log_dict = {}
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict.update(self.evaluator.curr_score())
        logger.info("{} Iter[{}/{}]\t{}".format(
            phase, iter + 1, max_iter,
            json.dumps(round_dict(log_dict))
        ))

    def log_epoch_info(self, phase="Test"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        metric, table_data = self.evaluator.mean_score(print=False)
        log_dict.update(metric)
        logger.info("{} Epoch\t{}".format(
            phase, json.dumps(round_dict(log_dict))
        ))
        if phase.lower() != "train":
            logger.info("\n" + AsciiTable(table_data).table)

    @torch.no_grad()
    def test(self):
        self.reset_meter()
        self.model.eval()

        max_iter = len(self.data_loader)
        end = time.time()
        for i, samples in enumerate(self.data_loader):
            inputs, labels = samples[0].to(self.device), samples[1].to(self.device)
            # forward
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                predicts.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)

            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter)
            end = time.time()
        self.log_epoch_info()
        # wandb.log({"examples": image_mask_list})


class DRFolderTester(ImageFolderTester):
    def __init__(self, cfg: CN, save_path: str):
        super().__init__(cfg, save_path)

    @torch.no_grad()
    def test(self):
        timer = AverageMeter()

        self.model.eval()
        max_iter = len(self.data_loader)
        end = time.time()

        if self.cfg.TEST.SAVE_LABELS:
            fsave = open(osp.join(self.save_path, "predicts.txt"), "w")

        for i, samples in enumerate(self.data_loader):
            inputs, sample_ids = samples[0].to(self.device), samples[1]
            # forward
            outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            # save predicts to predicted mask image
            if self.cfg.TEST.SAVE_LABELS:
                for j, sample_id in enumerate(sample_ids):
                    fsave.write(
                        "{} {}\n".format(osp.splitext(sample_id)[0], pred_labels[j])
                    )
            timer.update(time.time() - end)
            logger.info(
                "Test Epoch[{}/{}] Time {timer.val:.3f} ({timer.avg:.3f})".format(
                    i + 1, max_iter, timer=timer)
            )
            end = time.time()
        logger.info("Done with test Samples[{}]".format(len(self.data_loader.dataset)))

        