import torch
import torch.nn as nn
import numpy as np

from models import featurefusion

#criterion = nn.CrossEntropyLoss()

#default: same data ---link: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



#mixup-cross-domain
def mixup_domain(x_src, x_trg, y_src, alpha=0.2, fix_alpha=False, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0 and fix_alpha==False:
        lam = np.random.beta(alpha, alpha)
    elif alpha > 0 and fix_alpha==True:
        lam = alpha
    else:
        lam = 1

    #batch_size = x_src.size()[0]
    #if use_cuda:
    #    index = torch.randperm(batch_size).cuda()
    #else:
    #    index = torch.randperm(batch_size)

    mixed_x = lam * x_src + (1 - lam) * x_trg
    y_a = y_src
    return mixed_x, y_a

#def mixup_criterion(criterion, pred, y, y lam):
#    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#---cutmix---
def rand_bbox(size, lam):
    W = size[1]     # size = C, H, W
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_domain(x_src, x_trg, y_src, alpha=0.5):
    # generate mixed sample
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    elif fix_alpha==True:
        lam = alpha
    else:
        lam = 1

    #rand_index = torch.randperm(x_src.size()[0]).cuda()
    #print("tensor shape: ", x_src.shape[1])
    bbx1, bby1, bbx2, bby2 = rand_bbox(x_src.size(), lam)
    #x_src_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x_trg[rand_index, :, bbx1:bbx2, bby1:bby2]   #B, C, H, W
    x_src[:, bbx1:bbx2, bby1:bby2] = x_trg[:, bbx1:bbx2, bby1:bby2]  #C, H, W
    y_src_ = y_src
    # adjust lambda to exactly match pixel ratio
    #lam_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_src.size()[-1] * x_src.size()[-2]))

    return x_src, y_src_

def supermutant(x_mu, x_cm):
    supermutant = featurefusion.FeatureFusionModule(in_chan=6, out_chan=3).cuda()
    x_fusion = supermutant(x_mu, x_cm)
    return x_fusion, y_src

#---guassian filter---
def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

# Generate random matrix and multiply the kernel by it
A = np.random.rand(256*256).reshape([256,256])
A = torch.from_numpy(A)
guassian_filter = gkern(256, std=32)
