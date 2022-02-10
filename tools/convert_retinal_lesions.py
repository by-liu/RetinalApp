import os
import os.path as osp
import shutil

src_dir = "/home/bliu/work/Code/RetinalApp/data/retinal-lesions-v20191227/lesion_segs_896x896"
save_dir = "/home/bliu/work/Data/retinal-lesions/masks"

samples = sorted(os.listdir(src_dir))
print("{} samples found".format(len(samples)))

for sample in samples:
    sample_dir = osp.join(src_dir, sample)
    for mask in sorted(os.listdir(sample_dir)):
        mask_name, _ = osp.splitext(mask)
        shutil.copy(
            osp.join(sample_dir, mask),
            osp.join(save_dir, mask_name, "{}.png".format(sample))
        )

print("done")