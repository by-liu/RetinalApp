import numpy as np
import wandb
import logging
from typing import List, Optional
from terminaltables import AsciiTable
import sklearn.metrics
from sklearn.metrics import precision_score

from retinal.evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class MultiClassEvaluator(DatasetEvaluator):
    def __init__(
        self,
        num_classes: int,
        classes : Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if classes is None:
            self.classes = [str(i) for i in range(self.num_classes)]
        else:
            self.classes = classes
        assert (
            num_classes == len(self.classes)
        ), "Number of classes doesn't match"
        self.reset()

    def reset(self) -> None:
        self.preds = None
        self.labels = None

    def main_metric(self):
        return "kappa"
        # return "macc"

    def num_samples(self):
        return (
            self.labels.shape[0]
            if self.labels is not None
            else 0
        )

    def update(self, pred: np.ndarray, label: np.ndarray) -> float:
        """update

        Args:
            pred (np.ndarray): n x num_classes
            label (np.ndarray): n x 1

        Returns:
            float: acc
        """
        assert pred.shape[0] == label.shape[0]
        if self.preds is None:
            self.preds = pred
            self.labels = label
        else:
            self.preds = np.concatenate((self.preds, pred), axis=0)
            self.labels = np.concatenate((self.labels, label), axis=0)

        pred_label = np.argmax(pred, axis=1)
        acc = (pred_label == label).astype("int").sum() / label.shape[0]

        self.curr = {"acc": acc}

        return acc

    def curr_score(self):
        return self.curr

    def mean_score(self, print=False, all_metric=True):
        pred_labels = np.argmax(self.preds, axis=1)
        acc = (
            (pred_labels == self.labels).astype("int").sum() 
            / self.labels.shape[0]
        )
        confusion = sklearn.metrics.confusion_matrix(self.labels, pred_labels)
        class_acc = []
        for i in range(self.num_classes):
            class_acc.append(
                confusion[i, i] / np.sum(confusion[i])
            )
        macc = np.array(class_acc).mean()
        kappa = sklearn.metrics.cohen_kappa_score(self.labels, pred_labels, weights="quadratic")

        metric = {"acc": acc, "macc": macc, "kappa": kappa}

        columns = ["id", "Class", "acc"]
        table_data = [columns]
        for i in range(self.num_classes):
            table_data.append(
                [i, self.classes[i], "{:.4f}".format(class_acc[i])]
            )
        table_data.append(
            [None, "macc", "{:.4f}".format(macc)]
        )
        table_data.append(
            [None, "kappa", "{:.4f}".format(kappa)]
        )

        if print:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)

        if all_metric:
            return metric, table_data
        else:
            return metric[self.main_metric()], table_data

    def wandb_score_table(self):
        _, table_data = self.mean_score(print=False)
        return wandb.Table(
            columns=table_data[0],
            data=table_data[1:]
        )


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
