import os.path as osp
import math
import random
import numpy as np
from tqdm import tqdm

from retinal.data.retinal_lesions import RetinalLesions
from retinal.utils import load_list, save_list


def stats():
    data_root = "./data/retinal-lesions/"
    dataset = RetinalLesions(
        data_root,
        osp.join(data_root, "all.txt"),
        return_id=True
    )
    
    lesions = {}

    for i in tqdm(range(len(dataset))):
        img, target, sample_name = dataset[i]
        target = np.sum(target, axis=(0, 1))
        lesions[sample_name] = []
        for j in range(target.shape[0]):
            if target[j] > 0:
                lesions[sample_name].append(str(j))

    out_path = osp.join(data_root, "lesions.txt")
    with open(out_path, "w") as f:
        for sample_name in lesions:
            f.write("{} {}\n".format(sample_name, ",".join(lesions[sample_name])))


def save_samples(samples, path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write("{}\n".format(s))


def split_by_class(path):
    data_root = "./data/retinal-lesions/"
    classes_path = osp.join(data_root, "classes.txt")
    classes = load_list(classes_path)
    classes = [x.split(",")[0] for x in classes]
    classes.append("void")

    #classes_samples = [[] for _ in range(len(classes))]
    classes_samples = dict(zip(classes, [[] for _ in range(len(classes))]))
    with open(path, "r") as f:
        for line in f:
            fields = line.strip().split(" ")
            sample_name = fields[0]
            ids = "" if len(fields) == 1 else fields[1]
            # sample_name, ids = line.strip().split(" ")
            if ids == "":
                classes_samples["void"].append(sample_name)
                continue
            ids = [int(x) for x in ids.split(",")]
            for i in ids:
                classes_samples[classes[i]].append(sample_name)

    sorted_classes_samples = sorted(classes_samples.items(), key=lambda item: len(item[1]))
    train, val = set(), set()
    for class_name, samples in sorted_classes_samples:
        random.shuffle(samples)
        all_number = len(samples)
        val_number = min(math.ceil(all_number * 0.20), 200)
        train_number = all_number - val_number
        print("========")
        print("{} - train {}  val {}".format(class_name, train_number, val_number))

        index = 0
        cnt = 0
        while cnt < val_number and index < min(len(samples), val_number * 2):
            if samples[index] not in train and samples[index] not in val:
                val.add(samples[index])
                cnt += 1
                index += 1
            else:
                index += 1
        print("val {}".format(cnt))

        index = 0
        cnt = 0
        # while cnt < train_number and index < len(samples):
        while index < len(samples):
            if samples[index] not in train and samples[index] not in val:
                train.add(samples[index])
                cnt += 1
                index += 1
            else:
                index += 1
        print("train {}".format(cnt))

        print("After {} - train {}  val {}".format(class_name, len(train), len(val)))
    print("train : {} {}".format(len(train), len(set(train))))
    print("val : {} {}".format(len(val), len(set(val))))

    save_samples(list(train), osp.join(data_root, "train.txt"))
    save_samples(list(val), osp.join(data_root, "val.txt"))


def lesion_dist(split):
    data_root = "./data/retinal-lesions/"
    classes_path = osp.join(data_root, "classes.txt")
    classes = load_list(classes_path)
    classes = [x.split(",")[0] for x in classes]
    classes.append("void")

    samples = {}
    lesion_path = osp.join(data_root, "lesions.txt")
    with open(lesion_path, "r") as f:
        for line in f:
            fields = line.strip().split(" ")
            sample_name = fields[0]
            ids = "" if len(fields) == 1 else fields[1]
            if ids == "":
                samples[sample_name] = [-1]
                continue
            ids = [int(x) for x in ids.split(",")]
            samples[sample_name] = ids

    classes_count = {}
    for i in range(len(classes)):
        classes_count[classes[i]] = 0

    split_path = osp.join(data_root, "{}.txt".format(split))
    with open(split_path, "r") as f:
        for line in f:
            sample_name = line.strip()
            ids = samples[sample_name]
            for i in ids:
                classes_count[classes[i]] += 1

    print(split)
    print("========")
    for i in range(len(classes)):
        print("{} {} {}".format(i, classes[i], classes_count[classes[i]]))
    print("========")

# stats()
data_root = "./data/retinal-lesions/"
lesion_dist("train")
lesion_dist("val")
# split_by_class(osp.join(data_root, "lesions.txt"))

# data_dir = "./data/retinal-lesions/"

# all_list = load_list(osp.join(data_dir, "all.txt"))

# random.shuffle(all_list)

# train_num = int(len(all_list) * 0.8)
# val_num = len(all_list) - train_num

# train_list = all_list[:train_num]
# val_list = all_list[train_num:]

# print("{} + {} = {}".format(len(train_list), len(val_list), len(all_list)))

# save_list(osp.join(data_dir, "train.txt"), train_list)
# save_list(osp.join(data_dir, "val.txt"), val_list)
# print("done")