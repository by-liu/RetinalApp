import argparse
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm


def dice_coef(pred, target, eps=1e-10):
    assert pred.shape == target.shape, "The shapes of input and target do not match"

    inter = np.einsum("ij,ij->", pred, target)
    union = np.einsum("ij->", pred) + np.einsum("ij->", target)

    dice = (2 * inter + eps) / (union + eps)

    return dice


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask


def load_file_list(path):
    file_list = []
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            file_list.append(name)
    return file_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="")
    parser.add_argument("--test-path", default="test.txt")
    parser.add_argument("--test-result")
    parser.add_argument("--out-path", default="")

    return parser.parse_args()


def main():
    args = parse_args()

    # test_list = load_file_list(osp.join(args.data_root, args.test_path))
    test_list = load_file_list(args.test_path)

    dices = []
    # import ipdb; ipdb.set_trace()
    for i, name in tqdm(enumerate(test_list)):
        gt_mask = load_mask(osp.join(args.data_root, "binary_mask", name + ".png"))
        gt_mask = (gt_mask == 255).astype(int)
        pred_mask = load_mask(osp.join(args.test_result, name + ".png"))
        gt_mask = cv2.resize(gt_mask, pred_mask.shape, interpolation=cv2.INTER_NEAREST_EXACT)
        dice = dice_coef(pred_mask.astype(float), gt_mask.astype(float))

        dices.append(dice)
    print("mean : {:.4f}".format(np.mean(np.array(dices))))

    if args.out_path != "":
        with open(args.out_path, "w") as f:
            for name, dice in zip(test_list, dices):
                f.write("{} {:.4f}\n".format(name, dice))


if __name__ == "__main__":
    main()
