import unittest
import os.path as osp

from retinal.config import get_cfg
from retinal.data.fgadr_dataset import FGADRDataset
from retinal.data.data_transform import retinal_lesion


class TestFGADRDataset(unittest.TestCase):
    def test(self):
        data_root = "./data/FGADR-Seg/Seg-set"
        samples_path = osp.join(data_root, "all.txt")
        classes_path = osp.join(data_root, "classes.txt")
        cfg = get_cfg()
        transformer = retinal_lesion(cfg)

        dataset = FGADRDataset(
            data_root, samples_path, classes_path,
            data_transformer=transformer
        )

        for i in range(len(dataset)):
            img, target = dataset[i]

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
