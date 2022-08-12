import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pylab

from torch.autograd import Variable
from functools import partial

#class MMDloss(nn.Module):
#    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
#        super(MMDloss, self).__init__()
#        self.kernel_num = kernel_num
#        self.kernel_mul = kernel_mul
#        self.fix_sigma = None

#    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#        n_samples = int(source.size()[0])+int(target.size()[0])
#        total = torch.cat([source, target], dim=0)
#        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#        L2_distance = ((total0-total1)**2).sum(2) 

#        if fix_sigma:
#            bandwidth = fix_sigma
#        else:
#            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        
#        bandwidth /= kernel_mul ** (kernel_num // 2)
#        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
#        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#        return sum(kernel_val)

#    def forward(self, source, target):
#    	batch_size = int(source.size()[0])
#    	kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#    	XX = kernels[:batch_size, :batch_size]
#    	YY = kernels[batch_size:, batch_size:]
#    	XY = kernels[:batch_size, batch_size:]
#    	YX = kernels[batch_size:, :batch_size]
#    	loss = torch.mean(XX + YY - XY -YX)
#    	return loss

#Tensor = torch.cuda.FloatTensor()

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmdloss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

#def mmatch(self, x1, x2, n_moments=5):
#        mx1 = torch.mean(x1, 0, keepdim=True)
#        mx2 = torch.mean(x2, 0, keepdim=True) 
#        sx1 = x1 - mx1
#        sx2 = x2 - mx2
#        dm = ((mx1-mx2)**2).sum().sqrt()
#        scms = dm
#        for i in range(n_moments-1):
#            scms+=self.scm(sx1,sx2,i+2)
#        return scms

