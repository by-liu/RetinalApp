# Domain adaptation with TENT

Steps for performing domain daptation with TENT on target dataset.


## Requirments before training

* Trained model (like the model trained on eyepacs [Resnext-101](https://wandb.ai/newton/echo/reports/DR-classification-2022-04-21--VmlldzoxODc2OTUz?accessToken=8u9f2vljxh9w2zq3kgd9nc0rfc3nk9w5axqi4z6jm4y0jidihd2l8q0ukehr0ksk))

* Target domain dataset:

  * training set
  
  * testing set

## Prepare dataset

Putting all the samples from the target domain under one directory and organizing it as the example:

```
diagnos_all
├── Images
├── test-3067.csv
└── test-465.csv
```

Here, all the image files (training and testing) are included in the `Images` directory.
The training set is indicated by `test-3067.csv`, while the testing set is indicated by `test-465.csv`.
The format of the csv files is:

```
image_id,label
000528db-3bb1-c7b9-6459-b031971c4cfd.jpg,0
0078967b-6655-b013-ce70-c9db4f2d044d.jpg,0
009ee472-1810-6a8f-2d46-326934d4cedd.jpg,0
```

Note: the head is expected.


## Training by TENT

Run:

```
python tools/test_net.py \
    task=dr_tent \
    model=resnext101_32x16d \
    optim=sgd optim.lr=0.00005 \
    train.max_epoch=1 \
    data.batch_size=8 \
    data=diagnos \
    data.data_root=/home/bliu/work/Data/diagnos_all \
    data.val_split=test-3067 \
    data.test_split=test-465 \
    test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth 
```

The arguments related to the `paths` and `split` should be changed according to your environment.
The parameters of `lr`, `max_epoch` and `batch_size` could be tuned for optmizing the final performance.

After the job is done, the new checkpoint (a file named as tent-***.pth) could be found under a generated directory under the directory of `outputs`, like:

```
(bing) ➜  RetinalApp git:(main) ✗ ls outputs/diagnos/resnext101_32x16d-ce-sgd/20230208-17:51:33-598139
tent-3k3ud6f6-resnext101_32x16d-ce-best.pth  test_net.log
```

The resuling directory would be a little different for each run, as the name of the directory includes the timestamp.


## Testing

The above training set already performed testing, but without test-time data augmentation (TTDA).
If you want to test with TTDA, please refer to https://github.com/by-liu/ttda
