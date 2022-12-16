# Experiments on DIAGNOS-465 test set

* Network : [Resnext-101](https://wandb.ai/newton/echo/reports/DR-classification-2022-04-21--VmlldzoxODc2OTUz?accessToken=8u9f2vljxh9w2zq3kgd9nc0rfc3nk9w5axqi4z6jm4y0jidihd2l8q0ukehr0ksk)


Trained on the public dataset, EyePacs: 
 
| classes | samples |
|---------|---------|
| 0       | 42995     |
| 1       | 4079      |
| 2       | 8693      |
| 3       | 1393      |
| 4       | 1240      |


Target domain: Diagnos

Train-3067

| classes | samples |
|---------|---------|
| 0       | 2982    |
| 1       | 40      |
| 2       | 34      |
| 3       | 1       |
| 4       | 10      |


Test-465

| classes | samples |
|---------|---------|
| 0       | 390 |
| 1       | 29 |
| 2       | 43      |
| 3       | 3      |
| 4       | 0 |


## Results

kappa score (the most widely used (papers and competitions) metric for imbalanced dataset)


| method   |  kappa |
|----------|:------:|
| baseline | 0.6316 |
| TENT     | 0.6188 |
| TTDA     | 0.6381 |
| TENT + TTDA | **0.6446** |

accuracy of each class (correct_prediction / sample_number_per_class)
auc of each class (covert to binary result for each class)

| method    |  acc_0 |  auc_0 |  acc_1 |  auc_1 |  acc_2 |  auc_2 |  acc_3 |  auc_3 |
|-----------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| baseline  | 0.9744 | 0.8840 | 0.4138 | 0.9269 | 0.4651 | 0.8538 | 0.6667 | 0.9986 |
| TENT      | 0.9410 | 0.8945 | 0.4828 | 0.9326 | 0.6279 | 0.8713 | 0.6667 |  1.000 |
| TTDA      | 0.9769 | 0.8987 | 0.3103 | 0.9342 | 0.4884 | 0.8682 | 0.6667 | 0.9993 |
| TENT+TTDA | 0.9308 | 0.9031 | 0.4828 | 0.9249 | 0.6744 | 0.8769 | 0.6667 |  1.000 |
