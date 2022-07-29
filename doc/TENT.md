# Experiments with Tent 

The related code is transfered from the official repo : [tent](https://github.com/DequanWang/tent).

What we have before the experiments: 

* Model trained on EyePacs : [Resnext-101](https://wandb.ai/newton/echo/reports/DR-classification-2022-04-21--VmlldzoxODc2OTUz?accessToken=8u9f2vljxh9w2zq3kgd9nc0rfc3nk9w5axqi4z6jm4y0jidihd2l8q0ukehr0ksk)

| classes | samples |
|---------|---------|
| 0       | 42995     |
| 1       | 4079      |
| 2       | 8693      |
| 3       | 1393      |
| 4       | 1240      |
  
* Small Diagnos dataset : 200 samples

| classes | samples |
|---------|---------|
| 0       | 121     |
| 1       | 24      |
| 2       | 48      |
| 3       | 3       |
| 4       | 4       |

* Large Diagnos dataset : 3,067 samples with labels

| classes | samples |
|---------|---------|
| 0       | 2982    |
| 1       | 40      |
| 2       | 34      |
| 3       | 1       |
| 4       | 10      |


## Results

### 1. Peform Tent on the 200 samples and evaluate on the whole set.

***TTDA : test-time data augmentation***

| model      | Tent  | TTDA | macc | kappa | cmd |
| ---- | :---: | :---: | :---: | :---: | ---|
| Resnext-101   |  - | - |  0.6001 |  0.7038  | <details><summary>python</summary>```python tools/test_net.py model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize test.augment.flip=False```</details> |
| Resnext-101  | - | &check; | 0.6001  |  **0.7598**  | <details><summary>python</summary>```python tools/test_net.py model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize_and_centercrop test.augment.flip=True```</details> |
| Resnext-101  | &check; | - | **0.6167**  |  0.7253  | <details><summary>python</summary>```python tools/test_net.py task=dr_tent model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize test.augment.flip=False optim=sgd optim.lr=0.00005 train.max_epoch=5 data.batch_size=16```</details> |
| Resnext-101  | &check; | &check; | 0.5584  |  0.7489  | <details><summary>python</summary>```python tools/test_net.py task=dr_tent model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize_and_centercrop test.augment.flip=True optim=sgd optim.lr=0.00005 train.max_epoch=5 data.batch_size=16```</details> |

### 2. Divide the diagnos samples : 100 as val, 100 as test.Use Tent to finetune the model on val set and then evaluate on test set.

| model      | Tent  | TTDA | macc | kappa | cmd |
| ---- | :---: | :---: | :---: | :---: | ---|
| Resnext-101   |  - | - |  0.5717 |  0.6833  | <details><summary>python</summary>```python tools/test_net.py model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize test.augment.flip=False data.test_split=test-test```</details> |
| Resnext-101  | - | &check; | 0.5801  |  **0.7167**  | <details><summary>python</summary>```python tools/test_net.py model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize_and_centercrop test.augment.flip=True data.test_split=test-test```</details> |
| Resnext-101  | &check; | - | 0.5884  |  0.6449  | <details><summary>python</summary>```python tools/test_net.py task=dr_tent model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize test.augment.flip=False optim=sgd optim.lr=0.00005 train.max_epoch=5 data.batch_size=16 data.val_split=test-val data.test_split=test-test```</details> |
| Resnext-101  | &check; | &check; | 0.6051	| 0.6646  | <details><summary>python</summary>```python tools/test_net.py task=dr_tent model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize_and_centercrop test.augment.flip=True optim=sgd optim.lr=0.00005 train.max_epoch=5 data.batch_size=16```</details> |


### 3. Results on the larger set (3,067 samples with labels)

***TTDA : test-time data augmentation***

| model      | Tent  | TTDA | macc | kappa | cmd |
| ---- | :---: | :---: | :---: | :---: | ---|
| Resnext-101   |  - | - | 0.2830   | 0.2390  | <details><summary>python</summary>```python tools/test_net.py model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize test.augment.flip=False```</details> |
| Resnext-101  | - | &check; | 0.2594 |  0.2064  | <details><summary>python</summary>```python tools/test_net.py model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize_and_centercrop test.augment.flip=True```</details> |
| Resnext-101  | &check; | - | 0.2721  | 0.1869  | <details><summary>python</summary>```python tools/test_net.py task=dr_tent model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize test.augment.flip=False optim=sgd optim.lr=0.00005 train.max_epoch=5 data.batch_size=16```</details> |
| Resnext-101  | &check; | &check; | 0.2747  | 0.2243  | <details><summary>python</summary>```python tools/test_net.py task=dr_tent model=resnext101_32x16d data=diagnos test.checkpoint=/home/bliu/work/Code/RetinalApp/trained/3k3ud6f6-resnext101_32x16d-ce-best.pth test.save_prediction=False test.augment.method=resize_and_centercrop test.augment.flip=True optim=sgd optim.lr=0.00005 train.max_epoch=5 data.batch_size=16```</details> |


Score for each class:

| Class | samples | acc    |
| --- | --- | :---: |
| 0     | 2982 | 0.9883 |
| 1     | 40 | 0.1500 |
| 2     | 34 | 0.1765 |
| 3     | 1 | 0.0000 |
| 4     | 10 | 0.1000 |
| macc  |  | 0.2829 |
| kappa |  | 0.2390 |