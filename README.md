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

## Test with DR grading model

Download : [resnext101 model](https://drive.google.com/file/d/1GOZyktzFki_lJNyAO_oVL3hwAbnb_ofT/view?usp=sharing)
Some info about the model can be found at the [report](https://wandb.ai/newton/echo/reports/DR-classification-2022-04-21--VmlldzoxODc2OTUz?accessToken=8u9f2vljxh9w2zq3kgd9nc0rfc3nk9w5axqi4z6jm4y0jidihd2l8q0ukehr0ksk).

Run :
```
python tools/test_net.py \  
    model=resnext101_32x16d \
    data=folder \
    data.data_root=ABS_PATH_OF_IMAGE_FOLDER \
    hydra.run.dir=FOLDER_OF_CHECKPOINT \
    test.checkpoint=NAME_OF_CHECKPOINT 
```

Note : pass PATH_OF_IMAGE_FOLDER, DIR_OF_CHECKPOINT and NAME_OF_CHECKPOINT in the command.

After the job is done, the prediction could be found in the DIR_OF_CHECKPOINT folder with format "sample_id,label,probability"

Example :
```python
python tools/test_net.py model=resnext101_32x16d data=folder data.data_root=/home/bliu/work/Data/diagnos_dr hydra.run.dir=./trained test.checkpoint=3k3ud6f6-resnext101_32x16d-ce-best.pth
```


## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.
