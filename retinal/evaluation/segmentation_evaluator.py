import numpy as np
import logging
from typing import List, Optional
import pandas as pd
from terminaltables import AsciiTable

from retinal.evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class SegmentationEvaluator(DatasetEvaluator):
    def __init__(self,
                 classes: Optional[List[str]] = None,
                 include_background: bool = True,
                 ignore_index: int = -1) -> None:
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.include_background = include_background
        self.ignore_index = ignore_index

    def num_samples(self):
        return self.nsamples

    def reset(self):
        self.total_area_inter = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_union = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_pred = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_target = np.zeros((self.num_classes, ), dtype=np.float)
        self.nsamples = 0

    def main_metric(self):
        return "DSC"

    def ignore_background(self, pred: np.ndarray, target: np.ndarray):
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        target = target[:, 1:] if target.shape[1] > 1 else target
        return pred, target

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Update all the metric from batch size prediction and target.

        Args:
            pred: predictions to be evaluated in one-hot formation
            y: ground truth. It should be one-hot format.
        """
        if not self.include_background:
            pred, target = self.ignore_background(pred, target)

        assert pred.shape == target.shape, "pred and target should have same shapes"

        self.nsamples += pred.shape[0]
        area_inter = np.einsum("ncij,ncij->nc", pred, target)
        area_pred = np.einsum("ncij->nc", pred)
        area_target = np.einsum("ncij->nc", target)
        area_union = area_pred + area_target - area_inter

        # accumulate for the batch
        area_inter = area_inter.sum(axis=0)
        area_union = area_union.sum(axis=0)
        area_pred = area_pred.sum(axis=0)
        area_target = area_target.sum(axis=0)

        # update the total
        self.total_area_inter += area_inter
        self.total_area_union += area_union
        self.total_area_pred += area_pred
        self.total_area_target += area_target

        dice = 2 * area_inter.sum() / (np.spacing(1) + area_pred.sum() + area_target.sum())

        return dice

    def mean_score(self):
        mdice = (
            2 * self.total_area_inter
            / (np.spacing(1) + self.total_area_pred + self.total_area_target)
        ).mean()
        return mdice

    def class_score(self, return_dataframe=False):
        class_acc = self.total_area_inter / (np.spacing(1) + self.total_area_target)
        class_dice = (
            2 * self.total_area_inter
            / (np.spacing(1) + self.total_area_pred + self.total_area_target)
        )
        class_iou = self.total_area_inter / (np.spacing(1) + self.total_area_union)
        columns = ["id", "Class", "DSC", "IoU", "ACC"]
        class_table_data = [columns]
        for i in range(class_acc.shape[0]):
            class_table_data.append(
                [i] + [self.classes[i]]
                + ["{:.4f}".format(class_dice[i])]
                + ["{:.4f}".format(class_iou[i])]
                + ["{:.4f}".format(class_acc[i])]
            )
        class_table_data.append(
            [""] + ["mean"]
            + ["{:.4f}".format(np.mean(class_dice))]
            + ["{:.4f}".format(np.mean(class_iou))]
            + ["{:.4f}".format(np.mean(class_acc))]
        )
        table = AsciiTable(class_table_data)

        if return_dataframe:
            data = {key: [] for key in columns}
            for i in range(class_acc.shape[0]):
                data[columns[0]].append(i)
                data[columns[1]].append(self.classes[i])
                data[columns[2]].append(class_dice[i])
                data[columns[3]].append(class_iou[i])
                data[columns[4]].append(class_acc[i])
            return pd.DataFrame(data, columns=columns)
        else:
            logger.info("\n" + table.table)
            return class_dice
