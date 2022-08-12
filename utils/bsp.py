#---Batch Spectral Penalization---
import torch

def BSP(f_s, f_t):
    #feature_s = f_s.narrow(0, 0, int(f_s.size(0) / 2))
    #feature_t = f_t.narrow(0, int(f_t.size(0) / 2), int(f_t.size(0) / 2))
    #_, s_s, _ = torch.svd(feature_s)
    #_, s_t, _ = torch.svd(feature_t)

    _, s_s, _ = torch.svd(f_s)
    _, s_t, _ = torch.svd(f_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    print("BSP sigma shape: ", sigma.shape,type(sigma))
    return sigma