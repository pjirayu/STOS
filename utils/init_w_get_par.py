import torch.nn as nn

#--- default of init_weight and get_parameters from models ---
def init_weight(self):
    for ly in self.children():
        if isinstance(ly, nn.Conv2d):
            nn.init.kaiming_normal_(ly.weight, a=1)
            if not ly.bias is None: nn.init.constant_(ly.bias, 0)

def get_params(self):
    wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
    for name, child in self.named_children():
        child_wd_params, child_nowd_params = child.get_params()
        if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
            lr_mul_wd_params += child_wd_params
            lr_mul_nowd_params += child_nowd_params
        else:
            wd_params += child_wd_params
            nowd_params += child_nowd_params
    return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params