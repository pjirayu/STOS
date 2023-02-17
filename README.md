# Project setup & training preparation guide

## Introduction
This repository contains code for our article **High-Intensified Resemblance and Statistic-Restructured Alignment in Few-Shot Domain Adaptation for Industrial-Specialized Employment**<br/> If you find this method helpful please cite us. [article](https://ieeexplore.ieee.org/document/10045719)
###### <ins>Note</ins> all of this material is purely educational. In the case of actual production, The authors cannot affirm or verify the outcomes based on data obtained outside of this demonstration.

## Setup
* **Dataset** can be downloaded here [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/) then create a new "data" folder and put into it.

* **requirements** python==3.8, torch==1.9.0, torchvision==0.10.0, numpy==1.18.1
###### <ins>Note</ins> the considered environment runs on GPU with CUDA 11.1 and cuDNN 8.1 package versions. We can not confirm how this would affect working this env. with other package versions.

### To utilize Spectral-CORAL and other related resources in the demonstration
* Reproducible or reimplementable is by using the regarding function and any information in this repo (please read the license and policy, open and/or discuss the issues with us, and cite our paper). 
* Once, the below exemplary algorithm is using inferred outputs as inputs from both source and target domains calculated in our proposed function to obtain re-patterned covariance matrices for aligning.

#### For simply re-structural (b=1) correlation alignment
###### <ins>Note</ins> the testing was conducted with 3x3 toy covariance matrices running on the CPU implementation for demonstration. The available results showed only the 1st iteration for Covsqrt and Spectralcov in [Colab](https://colab.research.google.com/drive/1GV9XwNr2ONMmCTTVkFGj-4P-RouCphCh#scrollTo=CrQgvne8fF0Y).
```python3
def simplestrucCORAL(source, target):
    d = source.data.shape[1]
    # Stardardization
    s_ = source - torch.mean(source, 0, keepdim=True)
    t_ = target - torch.mean(target, 0, keepdim=True)
    # Normal correlation
    simple_cov_s = s_ @ s_.t()
    simple_cov_t = t_ @ t_.t()
    # Re-structuring (b=1 only)
    I = torch.eye(int(simple_cov_s.shape[0]))
    D_s = torch.diag(torch.sub(I, torch.mm(s_, s_.t())))
    D_t = torch.diag(torch.sub(I, torch.mm(t_, t_.t())))
    # Correlation matrix with b=1 factor structure
    cov_s = simple_cov_s + D_s
    cov_t = simple_cov_t + D_t
    # Frobenius Norm
    L2 = torch.mul((cov_s - cov_t), (cov_s - cov_t))
    mean = torch.mean(L2)
    loss = mean/(4*d*d)
    return loss
```
#### For re-structural (With the number of b factors) correlation alignment
```python3
def b_structure(Cov, order=1):
  # Initial
  iter = 1
  A0 = Cov
  # Identity matrix
  I = torch.eye(int(A0.shape[0])).cuda()
  # First factor (b=1)
  # diag
  diag_b1 = torch.diag(I-A0)
  #Structural Symmetric Correlation Matrix (A @ b=1)
  A = A0 + diag_b1
  # b factor>=2
  while iter < order:
    if order==1: print("break b factor iterative re-patterning corr if b==1"); break  # double-check for breaking iterative corr
    iter += 1
    # b factor iterative structural nearest corr; X(t)
    A = A + torch.diag(I-A)
  return A

def spectralCORAL(source, target, order=2):
  d = source.data.shape[1]
  s_ = source - torch.mean(source, 0, keepdim=True)
  t_ = target - torch.mean(target, 0, keepdim=True)
  cov_s = torch.matmul(s_, s_.t())
  cov_t = torch.matmul(t_, t_.t())
  b_cov_s = b_structure(cov_s, order)
  b_cov_t = b_structure(cov_t, order)
  # L2 Frobenius Norm
  L2 = torch.mul((b_cov_s - b_cov_t), (b_cov_s - b_cov_t))
  mean = torch.mean(L2)
  loss = mean/(4*d*d)
  return loss
```

## Training

**Training Spectral-CORAL adaptation task under vanilla classifier**
###### <ins>Remark:</ins> The initial state and whole training was set the initial state at varying-way five-shot training all along.
```bash
python main.py --model resnet50 --n_epoches 100 --n_target_samples 5 --batch_size 31 --mini_batch_size_g_h 31 --data_type office31 --source amazon --target webcam --dim 31 --C 31 --K 1 --la 1 --att_type n --tf_inv_loss spectralcoral --robust_order 6 --metatest n --mutation r --mutation_style mixup --alpha_mix 0.2 --da_type UDA
```

**Training Spectral-CORAL adaptation task 'n Attention Orchestration with SoftTriplet classifier (as the proposed STOS scheme)**<br/>
###### <ins>Remark:</ins> We set five multiple centers as follows in our hyperparameter setting of our published article for batch training.
```bash
python main.py --model resnet50 --n_epoches 100 --n_target_samples 5 --batch_size 31 --mini_batch_size_g_h 31 --data_type office31 --source amazon --target webcam --dim 155 --C 31 --K 5 --la 5 --att_type orcat --tf_inv_loss spectralcoral --robust_order 6 --metatest n --mutation r --mutation_style mixup --alpha_mix 0.2 --da_type UDA
```

## Activity recorded
- 2023/02/16 - Article pre-released (Early access)
- 2022/07/11 - All related files are under preparation.
- 2022/08/12 - All techniques and networks in this study are available.
- 2022/08/14 - Pre-release of the batch training in adaptation task.
- 2022/08/26 - Apache License 2.0 is included in this resource repository
