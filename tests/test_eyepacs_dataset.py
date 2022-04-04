import unittest
import os.path as osp
import numpy as np

from retinal.config import get_cfg
from retinal.data.eyepacs import EyePacsDataset
from retinal.data.data_transform import img_augment


class TestEyePacsDataset(unittest.TestCase):
    def test(self):
        data_root = "data/eyepacs"
        dataset = EyePacsDataset(
            data_root=data_root,
            split="train"
        )

        for i in range(5):
            img, label = dataset[i]
            print(img.shape)
            print(label)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
