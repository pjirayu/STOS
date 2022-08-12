import torch.nn.functional as F
import torch.nn as nn

#---Knowledge Distillation---

# not availability
#def kdloss(y, teacher_scores):
#    p = F.log_softmax(y, dim=1)
#    q = F.softmax(teacher_scores, dim=1)
#    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
#    return l_kl

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

'''
Exemplary training
loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach())   # KD = kl_div(Teacher - log(Student))
loss += loss_kd

Exemplary loss eval
if i == 1:
    print("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
'''