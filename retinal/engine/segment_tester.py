import time
import pprint
import logging
import torch
from yacs.config import CfgNode as CN
import wandb

from retinal.config import convert_cfg_to_dict
from retinal.engine.tester import DefaultTester
from retinal.evaluation import SegmentationEvaluator, AverageMeter
from retinal.data import build_data_pipeline

logger = logging.getLogger(__name__)


class SegmentTester(DefaultTester):
    def __init__(self, cfg: CN):
        self.cfg = cfg
        logger.info("SegmentTeser with config : ")
        logger.info(pprint.pformat(self.cfg))
        self.device = torch.device(self.cfg.DEVICE)
        self.data_loader = build_data_pipeline(self.cfg, split="test")
        self.build_model()
        # if self.cfg.TEST.SAVE_PREDICTS:
        #     self.save_path = osp.join(self.cfg.OUTPUT_DIR,
        #                               "{}_results".format(self.cfg.TEST.SPLIT))
        #     mkdir(self.save_path)
        self.build_meter()
        self.init_wandb_or_not()

    def init_wandb_or_not(self):
        if self.cfg.WANDB.ENABLE:
            wandb.init(
                project=self.cfg.WANDB.PROJECT,
                entity=self.cfg.WANDB.ENTITY,
                config=convert_cfg_to_dict(self.cfg),
                tags=["test"]
            )
            wandb.watch(self.model, log=None)

    def build_meter(self):
        self.classes = self.data_loader.dataset.classes
        self.evaluator = SegmentationEvaluator(
            classes=self.classes,
            include_background=True
        )
        self.batch_time_meter = AverageMeter()

    def wandb_iter_info(self, score=None):
        if not self.cfg.WANDB.ENABLE:
            return
        if score is not None:
            wandb.log(
                {"test/Iter/{}".format(self.evaluator.main_metric()): score}
            )

    def wandb_epoch_info_or_not(self, evaluator=None):
        if not self.cfg.WANDB.ENABLE:
            return
        if evaluator is not None:
            wandb.log(
                {"test/Epoch/{}".format(evaluator.main_metric()): evaluator.mean_score()}
            )
            if len(evaluator.classes) > 1:
                df = evaluator.class_score(return_dataframe=True)
                table = wandb.Table(dataframe=df)
                wandb.log({"test/Epoch/class_score": table})

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
            predicts = self.model.act(outputs)
            if self.cfg.MODEL.MODE == "multilabel" or self.cfg.MODEL.NUM_CLASSES == 1:
                pred_labels = (predicts > self.cfg.THRES).int()
            else:
                pred_labels = torch.argmax(predicts, dim=1)

            score = self.evaluator.update(pred_labels.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.LOG_PERIOD == 0:
                self.log_iter_info(i, max_iter,
                                   batch_time_meter=self.batch_time_meter,
                                   score=score)
                self.wandb_iter_info(score)
            end = time.time()
        self.log_epoch_info(self.evaluator)
        self.wandb_epoch_info_or_not(self.evaluator)
