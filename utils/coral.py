import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import math
from .utils import onehotembedding
from utils import entropy
'''
variance-covariance matrices (second moment)
sample statistic                            :       x.t * x / (n-1) or y.t * x / (n-1)
population parameter/empirical sample       :       x.t * x / n     or y.t * x / n

//Note// matrix operator
a @ b == torch.matmul(a,b) == torch.mm(a,b)
a * b == torch.mul(a,b)
'''

###################### deep coral (full algorithm) #########################
#@author: Baixu Chen
#@contact: cbx_99_hasta@outlook.com

class CorrelationAlignmentLoss(nn.Module):
    r"""The `Correlation Alignment Loss` in
    `Baochen Sun et al., Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016) <https://arxiv.org/pdf/1607.01719.pdf>`_.
    Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by
    .. math::
        C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S))
    .. math::
        C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T))
    where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
    source and target samples, respectively. We use :math:`d` to denote feature dimension, use
    :math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
    given by
    .. math::
        l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
    Shape:
        - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
        - Outputs: scalar.
    """

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff

#--- deep coral (simplified version) ---
# order = 1 ==> normal correlation alignment (coral)
def CORALloss(source, target):
    d = source.data.shape[1]
    # covariance matrix equality
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    #print(loss)
    loss = loss/(4*d*d)
    return loss


#---log-coral loss---///the worst result///
#converted torch version
def LogCORALloss(src,tgt):
    d = src.size(1)
    xm = torch.mean(src, 0, keepdim=True) - src
    src_c = xm.t() @ xm
    # target covariance
    xmt = torch.mean(tgt, 0, keepdim=True) - tgt
    tgt_c = xmt.t() @ xmt
    #src_c = coral(src)
    #tgt_c = coral(tgt)
    # eigen decomposition
    src_vals, src_vecs = torch.symeig(src_c,eigenvectors = True)
    tgt_vals, tgt_vecs = torch.symeig(tgt_c,eigenvectors = True)
    src_cc = torch.mm(src_vecs,torch.mm(torch.diag(torch.log(src_vals)),src_vecs.t()))
    tgt_cc = torch.mm(tgt_vecs,torch.mm(torch.diag(torch.log(tgt_vals)),tgt_vecs.t()))
    # Returns the Frobenius norm
    loss = torch.mean(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)))
    #loss = torch.sum(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)))
    loss = loss / (4 * d * d)
    return loss

#---Denman-Beavers iteration
'''
    #ref: https://en.wikipedia.org/wiki/Square_root_of_a_matrix
    Arg:
        @inproceedings{chen2020HoMM,
          title={HoMM: Higher-order Moment Matching for Unsupervised Domain Adaptation},
          author={Chao Chen, Zhihang Fu, Zhihong Chen, Sheng Jin, Zhaowei Cheng, Xinyu Jin, Xian-Sheng Hua},
          booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
          volume={34},
          year={2020}
        }
'''

# The functional original high-order covariance
def CovSqrt(Cov, order=6):
    Cov = Cov/torch.trace(Cov)
    Y0 = Cov
    Z0 = torch.eye(int(Cov.shape[0])).cuda()        #matrix identity equal to Y0 matrix dimension
    I = torch.eye(int(Cov.shape[0])).cuda()         #Z0 = I
    if order>=1:
        Y1 = 0.5 * torch.mul(Y0, 3 * I - torch.mul(Z0, Y0))
        Z1 = 0.5 * torch.mul(3 * I - torch.mul(Z0, Y0), Z0)
        if order!=1:
            pass
        else:
            Y1 = torch.multiply(torch.sign(Y1), torch.sqrt(torch.abs(Y1) + 1e-12))
            Y1 = Y1 / torch.norm(Y1)
        return Y1
    if order>=2:
        Y2 = 0.5 * torch.mul(Y1, 3 * I - torch.mul(Z1, Y1))
        Z2 = 0.5 * torch.mul(3 * I - torch.mul(Z1, Y1), Z1)
        if order!=2:
            pass
        else:
            Y2 = torch.multiply(torch.sign(Y2), torch.sqrt(torch.abs(Y2) + 1e-12))
            Y2 = Y2 / torch.norm(Y2)
        return Y2
    if order>=3:
        Y3 = 0.5 * torch.mul(Y2, 3 * I - torch.mul(Z2, Y2))
        Z3 = 0.5 * torch.mul(3 * I - torch.mul(Z2, Y2), Z2)
        if order!=3:
            pass
        else:
            Y3 = torch.multiply(torch.sign(Y3), torch.sqrt(torch.abs(Y3) + 1e-12))
            Y3 = Y3 / torch.norm(Y3)
        return Y3
    if order>=4:
        Y4 = 0.5 * torch.mul(Y3, 3 * I - torch.mul(Z3, Y3))
        Z4 = 0.5 * torch.mul(3 * I - torch.mul(Z3, Y3), Z3)
        if order!=4:
            pass
        else:
            Y4 = torch.multiply(torch.sign(Y4), torch.sqrt(torch.abs(Y4) + 1e-12))
            Y4 = Y4 / torch.norm(Y4)
        return Y4
    if order>=5:
        Y5 = 0.5 * torch.mul(Y4, 3 * I - torch.mul(Z4, Y4))
        Z5 = 0.5 * torch.mul(3 * I - torch.mul(Z4, Y4), Z4)
        if order!=5:
            pass
        else:
            Y5 = torch.multiply(torch.sign(Y5), torch.sqrt(torch.abs(Y5) + 1e-12))
            Y5 = Y5 / torch.norm(Y5)
        return Y5
    if order>=6:
        Y6 = 0.5 * torch.mul(Y5, 3 * I - torch.mul(Z5, Y5))
        Z6 = 0.5 * torch.mul(3 * I - torch.mul(Z5, Y5), Z5)
        if order!=6:
            pass
        else:
            Y6 = torch.multiply(torch.sign(Y6), torch.sqrt(torch.abs(Y6) + 1e-12))
            Y6 = Y6 / torch.norm(Y6)
        return Y6

# order = 0.5 (i.e., much higher order converge as 1/2 ==> square-root normalization (robust) correlation alignment"
#torch-based function following their work, derived by P.Jirayu
def RobustCORALloss(source, target, order):
    d = source.data.shape[1]
    n_batch_size = source.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    #xc = (xm.t() @ xm) * (1/(n_batch_size-1))
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    #xct = (xmt.t() @ xmt) * (1/(n_batch_size-1))
    #order
    xc_p = CovSqrt(xc, order)
    xct_p = CovSqrt(xct, order)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc_p - xct_p), (xc_p - xct_p)))
    #print(loss)
    loss = loss * 2.
    return loss

#----------------------undergoing in sqrt-norm coral runs----------------------------




#---The established function following the fundamental of sqrt-norm iteration cov matrix (test)---
#---Babylonian iteration
def Babylonian_iter(Cov, order=1):
    Cov = Cov/torch.trace(Cov)
    #prelim
    A = Cov
    I = torch.eye(int(Cov.shape[0])).cuda()
    X_k = I
    #iterative babylonian method (X[k+1] = 0.5 * (X[k] + Cov * inv(X[k]))
    #1st order
    for idx in range(order):
      X_k = 0.5 * (X_k + A * torch.inverse(X_k))
    return X_k


#Denver Newton's iteration
def Denver_iter(Cov, order=1):
    Cov = Cov/torch.trace(Cov)
    Y0 = Cov
    Z0 = torch.eye(int(Cov.shape[0])).cuda()        #matrix identity equal to Y0 matrix dimension
    #I = torch.eye(int(Cov.shape[0])).cuda()         #Z0 = I
    for idx in range(order):
        if idx==0:
            Y = 0.5 * torch.add(Y0, torch.inverse(Z0))
            Z = 0.5 * torch.mul(Z0, torch.inverse(Y0))
        else:
            Y = 0.5 * torch.add(Y, torch.inverse(Z))
            Z = 0.5 * torch.mul(Z, torch.inverse(Y))
        Z = (Cov * Z) + Z

    Y = torch.multiply(torch.sign(Y), torch.sqrt(torch.abs(Y) + 1e-12))
    Y = Y / torch.norm(Y)
    return Y

#Kung–Traub-Type Iterative (2m matrix–matrix multiplications)--test //worse//
def speedyiter(Cov, order=1):
    #Moore-Penrose Inverse Initial state
    #V0 = 2* Cov.t()/torch.sqrt(torch.trace(torch.mm(Cov.t(), Cov)))
    Cov = Cov/torch.trace(Cov)
    V0 = Cov
    I = torch.eye(int(Cov.shape[0])).cuda()
    #Covariance Matrix inversion
    for idx in range(order):
        if idx==0:
            R = I - Cov * V0
            V = V0 * (I + R + R**2 + R**3 + R**4 + R**5 + R**6 + R**7) #+ R**8 + R**9)
        else:
            R = I - Cov * V
            V = V * (I + R + R**2 + R**3 + R**4 + R**5 + R**6 + R**7) #+ R**8 + R**9)
    return V

def blockiter(Cov, order):  #//worse//
    #A0 = 2* Cov.t()/torch.sqrt(torch.trace(torch.mm(Cov.t(), Cov)))
    Cov = Cov/torch.trace(Cov)
    A0 = Cov
    for idx in range(order):
        if idx==0:
            A = torch.mul(2, A0) - torch.mul(torch.mul(A0, Cov), A0)
        else:
            A = torch.mul(2, A) - torch.mul(torch.mul(A, Cov), A)
    return A


#---Iterative covariance inversion derived by P.Jirayu---
def IterCORALloss(source: torch.tensor,
                  target: torch.tensor,
                  iter_type: str,
                  order: int):
    d = source.data.shape[1]
    n_batch_size = source.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    #xc = (xm.t() @ xm) * (1/(n_batch_size-1))
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    #xct = (xmt.t() @ xmt) * (1/(n_batch_size-1))
    if iter_type=='babylonian':
        xc_p = Babylonian_iter(xc, order)
        xct_p = Babylonian_iter(xct, order)
    elif iter_type=='denver':
        xc_p = Newton_denver_iter(xc, order)
        xct_p = Newton_denver_iter(xct, order)
    elif iter_type=='speedy':
        xc_p = speedyiter(xc, order)
        xct_p = speedyiter(xct, order)
    elif iter_type=='block':
        xc_p = blockiter(xc, order)
        xct_p = blockiter(xct, order)
    else:
        raise ValueError('Iterative method should be selected in processing...')

    # Frobenius normalization between source and target
    loss = torch.mean(torch.mul((xc_p - xct_p), (xc_p - xct_p)))
    #print(loss)
    loss = loss * 2.
    return loss


#*******************************************************************************************************
#---Our proposed method---
#*******************************************************************************************************
'''
    Arg:
        J. Petchhan and S.-F. Su. High-intensified resemblance & Statistic-Restructured Alignment for Few-Shot DA. in IEEE TCE, 2022.
'''
#-------------------------------------------------------------------------------------------------------
#Simple (k=1 only) the nearest correlation matrix with factor structure
def simplestrucCORAL(source, target):
    d = source.data.shape[1]
    # Standardization
    s_ = source - torch.mean(source, 0, keepdim=True)
    t_ = target - torch.mean(target, 0, keepdim=True)
    # Normal correlation
    simple_cov_s = s_ @ s_.t()
    simple_cov_t = t_ @ t_.t()
    # Re-structuring (b=1 only)
    I = torch.eye(int(simple_cov_s.shape[0])).cuda()
    D_s = torch.diag(torch.sub(I, torch.mm(s_, s_.t())))
    D_t = torch.diag(torch.sub(I, torch.mm(t_, t_.t())))
    # Correlation structure with k=1
    cov_s = simple_cov_s + D_s
    cov_t = simple_cov_t + D_t
    # Frobenius Norm
    L2 = torch.mul((cov_s - cov_t), (cov_s - cov_t))
    mean = torch.mean(L2)
    loss = mean/(4*d*d)
    return loss

#---b nearest for factor structure
def b_structure(Cov, order=1):
  '''
  Referred to in Borsdorf et al. Computing a Nearest Correlation Matrix with k Factor Structure. 2010.
  Arg:
    To minimize cost F; argmin||A - F(X(t-1)) - matmul(X, X.t())||F
    F = I - torch.diag(A)
    A = A - A0 - F
    A = torch.trace(torch.mm(Y.t(), Y))
  '''
  # Initialization
  iter = 1
  # A0 = Cov = Cov/torch.trace(Cov)
  A0 = Cov
  # Identity matrix
  I = torch.eye(int(A0.shape[0])).cuda()

  # First factor (b=1)
  # diag
  # diag_b1 = torch.diag(torch.sub(I, torch.mm(mean_pop, mean_pop.t())))
  diag_b1 = torch.diag(A0)
  #Structural Symmetric Correlation Matrix (A @ b=1)
  A = A0 + diag_b1

  # matrix iter b>=2
  while iter < order:
    if order==1: print("break b factor iterative nearest corr"); break
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
#-------------------------------------------------------------------------------------------------------