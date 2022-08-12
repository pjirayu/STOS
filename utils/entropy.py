#----------------
#Discriminative Feature Alignment: Improving Transferability of Unsupervised Domain Adaptation by Gaussian-guided Latent Alignment
#by Jing Wang et al.
#code available at https://github.com/JingWang18/Discriminative-Feature-Alignment
#----------------

#import some part (loss defined function), not software code at all.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#1
def ent(output):
    return - torch.mean(output * torch.log(output + 1e-6))

#2 reserve
def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))

#kullback-leibler div
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

#for reconstructed image & real image comparison
def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

#euclidean
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

def domain_discrepancy(out1, out2, loss_type, delta_const=1., power=3):
    def huber_loss(e, d=delta_const):
        t =torch.abs(e)
        ret = torch.where(t < d, 0.5 * t ** 2, d * (t - 0.5 * d))
        return torch.mean(ret)

    def pseudo_huber_loss(e, d=delta_const):
        t =torch.abs(e)
        ret = torch.where(t < d, t ** 2, d * torch.sqrt(1 + (t**2 / d**2)))
        return torch.mean(ret)

    def log_huber_loss(e, d=delta_const):
        t =torch.abs(e)
        ret = torch.where(t < d, t ** 2, (d ** 2) * (1 - 2 * math.log(d) - torch.log(t ** 2)))
        return torch.mean(ret)

    def geman_huber_loss(e, d=delta_const):
        t =torch.abs(e)
        ret = torch.where(t < d, 0.5 * (t**2)/(1+t**2), d * (t/(1+t) - 0.5 * d)) 
        return torch.mean(ret)

    def Geman_Mcclure_Power(e, power=3):
        ret = ((diff ** power)/power) / (1+diff ** power)
        return torch.mean(ret)

    def Cauchy_power(e,d=delta_const, power=power):
        t = torch.abs(e)
        ret = d**power/power * torch.log(1 + (diff/d)**power)
        return torch.mean(ret)

    def fair_loss(e,d=delta_const):
        t = torch.abs(e)
        ret = d**2 * (t/d - torch.log(1+t/d))
        return torch.mean(ret)

    def Cauchy_loss(e,d=delta_const):
        t = torch.abs(e)
        ret = d**2/2 * torch.log(1 + (t/d)**2)
        return torch.mean(ret)

    diff = out1 - out2
    if loss_type == 'L1':
        loss = torch.mean(torch.abs(diff))
    elif loss_type == 'huber':
        loss = huber_loss(diff)
    elif loss_type == 'pseudo_huber':
        loss = pseudo_huber_loss(diff)
    elif loss_type == 'log_huber':
        loss = log_huber_loss(diff)
    elif loss_type == 'geman_huber':
        loss = geman_huber_loss(diff)
    elif loss_type == 'fair':
        loss = fair_loss(diff)
    elif loss_type == 'cauchy':
        loss = Cauchy_loss(diff)
    elif loss_type == 'geman_power':
        loss = Geman_Mcclure_Power(diff, power=3)
    elif loss_type == 'cauchy_power':
        loss = Cauchy_power(diff, power=3)
    else:
        #print("using L2 for domain discrepancy...")
        loss = torch.mean(diff*diff)
    return loss

#Jeffrey’s Kullback–Leibler Divergence (JKLD)
def JKLD(source, target):
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    #identity matrix of cov
    I = torch.eye(int(xc.shape[0])).cuda()
    #square distance
    sq_dist = 0.5 * torch.trace(torch.mul(torch.inverse(xct), xc)+torch.mul(torch.inverse(xc), xct)-torch.mul(2, I))
    return sq_dist