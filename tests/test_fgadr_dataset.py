import unittest
import os.path as osp
import numpy as np
from terminaltables import AsciiTable
from tqdm import tqdm

from retinal.config import get_cfg
from retinal.data.fgadr_dataset import FGADRDataset
from retinal.data.data_transform import retinal_lesion


class TestFGADRDataset(unittest.TestCase):
    def test(self):
        data_root = "./data/FGADR-Seg/Seg-set"
        samples_path = osp.join(data_root, "test.txt")
        classes_path = osp.join(data_root, "classes.txt")
        cfg = get_cfg()
        transformer = retinal_lesion(cfg)

        dataset = FGADRDataset(
            data_root, samples_path, classes_path,
            data_transformer=None
        )

        total_area = 0
        class_area = [0 for _ in range(6)]
        for i in tqdm(range(len(dataset))):
            img, target = dataset[i]
            total_area += target.shape[0] * target.shape[1]
            for j in range(6):
                class_area[j] += np.sum(target[:, :, j])

        table_data = [["id"] + ["Class"] + ["Area"] + ["Proportion"]]
        for i in range(6):
            table_data.append(
                [i] + [dataset.classes[i]]
                + [class_area[i]]
                + ["{:.4f}".format(class_area[i] / total_area)]
            )
        table_data.append(
            [""] + ["all"] + [total_area] + [""]
        )
        table = AsciiTable(table_data)
        print(table.table)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
