import numpy as np
import logging
from typing import List, Optional
from terminaltables import AsciiTable
from sklearn.metrics import precision_score

from retinal.evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class MultilabelEvaluator(DatasetEvaluator):
    def __init__(
        self,
        classes: Optional[List[str]] = None,
        thres: float = 0.5
    ):
        super().__init__()
        self.classes = classes
        self.thres = thres

        self.num_classes = len(self.classes)

    def reset(self) -> None:
        self.preds = None
        self.labels = None

    def main_metric(self) -> None:
        return "macc"

    def num_samples(self):
        return (
            self.labels.shape[0]
            if self.labels is not None
            else 0
        )

    def update(self, pred: np.ndarray, label: np.ndarray):
        """update

        Args:
            pred (np.ndarray): n x num_classes
            label (np.ndarray): n x num_classes
        """
        assert pred.shape == label.shape
        pred = (pred > self.thres).astype(np.float32)

        if self.preds is None:
            self.preds = pred
            self.labels = label
        else:
            self.preds = np.concatenate((self.preds, pred), axis=0)
            self.labels = np.concatenate((self.labels, label), axis=0)

        macc = precision_score(label, pred, average="macro")

        self.curr = {"macc": macc}
        return macc

    def curr_score(self):
        return self.curr

    def mean_score(self, print=False):
        acc = precision_score(self.labels, self.preds, average=None)

        macc = np.mean(acc)

        metric = {"macc": macc}

        columns = ["class", "samples", "acc"]
        table_data = [columns]

        for i, cls in enumerate(self.classes):
            num_sample = np.sum(self.labels[:, i])
            table_data.append([
                self.classes[i],
                str(num_sample),
                "{:.5f}".format(acc[i]),
            ])

        if print:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)

        return metric, table_data
