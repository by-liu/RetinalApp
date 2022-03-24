# Retinal Applications

## Prerequisites

Note : I only test the code with python 3.8 and 3.9. Environment manager like conda or virtualenv is strongly recommended.

1. Install [pytorch](https://pytorch.org/) and opencv tailored for your environment:
    ```
    torch==1.7.1
    torchvision==0.8.2
    opencv-python==4.5.1.48
    ```

2. Other depencencies
    ```
    pip install -r requirements.txt
    ```

3. Install the library
    ```
    pip install -e .
    ```

## Test with Retinal Lesions segmentation model

Download : [model](https://drive.google.com/file/d/10qFflD10qwI-dHjLGOQpPub7RHv31SAi/view?usp=sharing)

Run :
```
python tools/test_net_with_folder.py --config-file ./configs/retinal-lesions_fpn_ce.yaml \
    --image-path [PATH_OF_IMAGE_FOLDER] \
    --save-path results/ \
    TEST.SAVE_LABELS True  TEST.CHECKPOINT_PATH [PATH_OF_CHECKPOINT]
```

Note : Please modify the PATH_OF_IMAGE_FOLDER and PATH_OF_CHECKPOINT in the command.

After the job is done, the resulting folder would be like:
```
results
├── 0178_1
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── 6.png
├── 0464_1
│   ├── 2.png
│   ├── 3.png
│   └── 6.png
├── 0498_3
│   ├── 1.png
│   ├── 2.png
│   └── 6.png
├── 12481_right
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── 6.png
├── 35515_right
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── 6.png
└── predicts.txt
```

As the model is under multi-label setting, there could be multiple lesions for the pixels in each image. Therefore, we output multiple masks for each image, with each corresponding to a particular class.

The indices of the classes are as follows:

| Index        | Class |
|--------------|-----------|
| 0 | IRMA       |
| 1     | cotton_wool_spots     |
| 2     | hard_exudate      |
| 3     | microaneurysm      |
| 4     | neovascularization      |
| 5     | preretinal_hemorrhage      |
| 6     | retinal_hemorrhage      |
| 7     | vitreous_hemorrhage      |
| 8     | fibrous_proliferation |

If "TEST.SAVE_LABELS" is set to true, a "predicts.txt" file is also generated in the result folder, presenting the lesion labels for each image:

```
0178_1 1,2,3,6
0464_1 2,3,6
0498_3 1,2,6
12481_right 1,2,3,6
35515_right 1,2,3,6
```

## Test with DR grading model

Download : [model](https://drive.google.com/file/d/16NT-i5lMG-Ht0Ty0gNkHYjk1uiqQdUKC/view?usp=sharing)
Some info about the model can be found at the [report](https://wandb.ai/newton/retinal/reports/DR-grading-baseline-20220311--VmlldzoxNjc1MDgy?accessToken=ngzk5hfdd228lahukgeg2sm5byrt0d8qiglgts591qyv18wh5t3j6nwfxempofwx).

Run :
```
python tools/test_net_with_folder.py \
    --task dr \
    --config-file ./configs/dr/eyepacs_resnet_ce_adam.yaml \
    --image-path [PATH_OF_IMAGE_FOLDER] \
    --save-path results/ \
    TEST.SAVE_LABELS True  TEST.CHECKPOINT_PATH [PATH_OF_CHECKPOINT]
```

Note : Please modify the PATH_OF_IMAGE_FOLDER and PATH_OF_CHECKPOINT in the command.

After the job is done, the prediction could be found in the resulting folder.


## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.
