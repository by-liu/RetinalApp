import logging
import time
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch.nn.functional as F
import os.path as osp

from retinal.evaluation import MultiClassEvaluator, AverageMeter, LossMeter
import retinal.modeling.tent as tent
from .dr_tester import DRTester
from ..utils.checkpoint import save_checkpoint


logger = logging.getLogger(__name__)


class DRTent(DRTester):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.init_tent()

    def build_data_loader(self) -> None:
        self.val_loader = instantiate(self.cfg.data.object.val)
        self.test_loader = instantiate(self.cfg.data.object.test_ta)

    def build_meter(self):
        self.num_classes = self.val_loader.dataset.num_classes
        self.evaluator = MultiClassEvaluator(
            num_classes=self.num_classes
        )
        self.batch_time_meter = AverageMeter()
        logger.info("Meters initialized")

    def init_tent(self):
        self.model = tent.configure_model(self.model)
        params, param_names = tent.collect_params(self.model)
        self.optimizer = instantiate(
            self.cfg.optim.object, params
        )
        self.tented_model = tent.Tent(self.model, self.optimizer, steps=1)

    def tent_epoch(self):
        self.reset_meter()

        max_iter = len(self.val_loader)
        end = time.time()

        for i, samples in enumerate(self.val_loader):
            # decouple samples
            inputs, labels = samples[0].to(self.device), samples[1].to(self.device)
            # forward
            outputs = self.tented_model(inputs)
            # outputs = self.model(inputs)
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                predicts.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter)
            end = time.time()
        self.log_epoch_info()

    def test(self):
        for epoch in range(self.cfg.train.max_epoch):
            logger.info("=" * 20)
            logger.info(" Start tent epoch {}".format(epoch + 1))
            logger.info("=" * 20)
            self.tent_epoch()
        save_checkpoint(
            self.model,
            self.work_dir,
            f"tent-{osp.split(self.cfg.test.checkpoint)[1]}"
        )

        self.test_epoch()
