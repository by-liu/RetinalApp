import wandb
import numpy as np


def wandb_mask(image, pred_mask, gt_mask, classes):
    # import ipdb; ipdb.set_trace()
    # image = (image * 255).astype(np.uint8)
    image = np.einsum("kij->ijk", image)
    pred_mask = pred_mask.astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)

    masks = {}
    for i, class_name in enumerate(classes):
        masks["{}_pred".format(class_name)] = {
            "mask_data": pred_mask[i]
        }
        masks["{}_gt".format(class_name)] = {
            "mask_data": gt_mask[i]
        }

    return wandb.Image(image, masks=masks)


def wandb_batch_mask(images, pred_masks, gt_masks, classes):
    columns = ["id", "image"]
    for class_name in classes:
        columns.append("pred_" + class_name)
        columns.append("gt_" + class_name)
    my_data = []
    for i in range(pred_masks.shape[0]):
        data = [i]
        image = np.einsum("kij->ijk", images[i])
        data.append(wandb.Image(image))
        pred_mask = pred_masks[i].astype(np.uint8) * 255
        gt_mask = gt_masks[i].astype(np.uint8) * 255
        for j, class_name in enumerate(classes):
            data.append(wandb.Image(pred_mask[j]))
            data.append(wandb.Image(gt_mask[j]))
            # wandb_image = wandb.Image(
            #     image,
            #     masks={
            #         "pred": {
            #             "mask_data": pred_mask[j],
            #             # "class_labels": {1: class_name}
            #         },
            #         "gt": {
            #             "mask_data": gt_mask[j],
            #             # "class_labels": {1: class_name}
            #         }
            #     }
            # )
            # wandb.log({"mask": wandb_image})
            # data.append(wandb_image)
        my_data.append(data)
    table = wandb.Table(data=my_data, columns=columns, allow_mixed_types=True)
    wandb.log({"predictions": table})
