import unittest
import torch
import random
import os.path as osp
from torch.utils.data import DataLoader

from retinal.config import get_cfg
from retinal.data.retinal_lesions import RetinalLesions
from retinal.data.data_transform import retinal_lesion


class TestRetinalDataset(unittest.TestCase):
    def test(self):
        cfg = get_cfg()
        cfg.merge_from_file("./configs/dr/eyepacs_resnext_ce.yaml")

        data_root = "./data/retinal-lesions"
        samples_path = osp.join(data_root, "train.txt")

        data_transform = retinal_lesion(cfg, is_train=True)

        dataset = RetinalLesions(
            data_root,
            samples_path,
            data_transformer=data_transform,
            return_classes=True
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=16,
            num_workers=1,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
        )

        for i, samples in enumerate(data_loader):
            inputs, masks, labels = samples
            print(i)
            print("=" * 10)
            print(inputs.shape)
            print(masks.shape)
            print(labels.shape)
            # i = random.randint(0, len(dataset))
            # sample = dataset[i]
            # print(sample[0].shape, sample[1].shape, sample[2].shape)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()