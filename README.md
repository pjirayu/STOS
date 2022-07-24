# Project setup & training preparation guide

## Introduction
- This repository contains code for our article **High-Intensified Resemblance and Statistic-Restructured Alignment for Few-Shot Domain Adaptation** under-reviewing

## Activity recorded
- 2022/07/11 Undergoing to re-arrange and -directory for all file as to simpler version ...

## All file repository in STOS establishment
- Coming soon

## Setup
* **Dataset** can be downloaded here [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/)

* **requirements** Python==3.8, torch==1.9.0, torchvision==0.10.0, numpy==1.18.1

## Training

**Training under vanilla classifier**
```python
python main.py --model resnet18 --n_epoches_3 100 --n_target_samples 5 --batch_size 31 --mini_batch_size_g_h 31 --data_type office31 --source amazon --target webcam --dim 31 --C 31 --K 1 --la 1 --att_type n --tf_inv_loss spectralcoral --robust_order 6 --metatest n --mutation r --mutation_style mixup --alpha_mix 0.2 --src_train adapting --da_type UDA
```

**Training under SoftTriplet classifier (as the proposed STOS)**
```python
python main.py --model resnet18 --n_epoches_3 100 --n_target_samples 5 --batch_size 31 --mini_batch_size_g_h 31 --data_type office31 --source amazon --target webcam --dim 155 --C 31 --K 5 --la 5 --att_type orcat --tf_inv_loss spectralcoral --robust_order 6 --metatest n --mutation r --mutation_style mixup --alpha_mix 0.2 --src_train adapting --da_type UDA
```
