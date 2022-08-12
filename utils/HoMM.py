'''
    Arg:
        @inproceedings{chen2020HoMM,
          title={HoMM: Higher-order Moment Matching for Unsupervised Domain Adaptation},
          author={Chao Chen, Zhihang Fu, Zhihong Chen, Sheng Jin, Zhaowei Cheng, Xinyu Jin, Xian-Sheng Hua},
          booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
          volume={34},
          year={2020}
        }https://github.com/chenchao666/HoMM-Master
'''

import torch
#import torch.tensor as tensor
import torch.nn as nn
from models import lina
import torch.nn.functional as F

#tf to torch
#tf.reduce_mean -> tensor.mean
#tf.expand_dims -> tensor.expand
#tf.transpose -> tensor.permute

#tf-based HoMM function follows their work
'''
#---HoMM3---
def HoMM3_loss(self, xs, xt):
        xs = xs - tf.reduce_mean(xs, axis=0)
        xt = xt - tf.reduce_mean(xt, axis=0)
        xs=tf.expand_dims(xs,axis=-1)
        xs = tf.expand_dims(xs, axis=-1)
        xt = tf.expand_dims(xt, axis=-1)
        xt = tf.expand_dims(xt, axis=-1)
        xs_1 = tf.transpose(xs, [0, 2, 1, 3])
        xs_2 = tf.transpose(xs, [0, 2, 3, 1])
        xt_1 = tf.transpose(xt, [0, 2, 1, 3])
        xt_2 = tf.transpose(xt, [0, 2, 3, 1])
        HR_Xs=xs*xs_1*xs_2   # dim: b*L*L*L
        HR_Xs=tf.reduce_mean(HR_Xs,axis=0)   #dim: L*L*L
        HR_Xt = xt * xt_1 * xt_2
        HR_Xt = tf.reduce_mean(HR_Xt, axis=0)
        return tf.reduce_mean(tf.square(tf.subtract(HR_Xs, HR_Xt)))

#---HoMM4---
def HoMM4(self,xs,xt):
	ind=tf.range(tf.cast(xs.shape[1],tf.int32))
	ind=tf.random_shuffle(ind)
	xs=tf.transpose(xs,[1,0])
	xs=tf.gather(xs,ind)
	xs = tf.transpose(xs, [1, 0])
	xt = tf.transpose(xt, [1, 0])
	xt = tf.gather(xt, ind)
	xt = tf.transpose(xt, [1, 0])
	return self.HoMM4_loss(xs[:,:30],xt[:,:30])+self.HoMM4_loss(xs[:,30:60],xt[:,30:60])+self.HoMM4_loss(xs[:,60:90],xt[:,60:90])

def HoMM4_loss(self, xs, xt):
	xs = xs - tf.reduce_mean(xs, axis=0)
	xt = xt - tf.reduce_mean(xt, axis=0)
	xs = tf.expand_dims(xs,axis=-1)
	xs = tf.expand_dims(xs, axis=-1)
	xs = tf.expand_dims(xs, axis=-1)
	xt = tf.expand_dims(xt, axis=-1)
	xt = tf.expand_dims(xt, axis=-1)
	xt = tf.expand_dims(xt, axis=-1)
	xs_1 = tf.transpose(xs,[0,2,1,3,4])
	xs_2 = tf.transpose(xs, [0, 2, 3, 1,4])
	xs_3 = tf.transpose(xs, [0, 2, 3, 4, 1])
	xt_1 = tf.transpose(xt, [0, 2, 1, 3,4])
	xt_2 = tf.transpose(xt, [0, 2, 3, 1,4])
	xt_3 = tf.transpose(xt, [0, 2, 3, 4, 1])
	HR_Xs=xs*xs_1*xs_2*xs_3    # dim: b*L*L*L*L
	HR_Xs=tf.reduce_mean(HR_Xs,axis=0)  # dim: L*L*L*L
	HR_Xt = xt * xt_1 * xt_2*xt_3
	HR_Xt = tf.reduce_mean(HR_Xt, axis=0)
	return tf.reduce_mean(tf.square(tf.subtract(HR_Xs, HR_Xt)))


#---Arbitrary-order Moment Matching---
def HoMM(self,xs, xt, order=3, num=300000):
	xs = xs - tf.reduce_mean(xs, axis=0)
	xt = xt - tf.reduce_mean(xt, axis=0)
	dim = tf.cast(xs.shape[1], tf.int32)
	index = tf.random_uniform(shape=(num, dim), minval=0, maxval=dim - 1, dtype=tf.int32)
	index = index[:, :order]
	xs = tf.transpose(xs)
	xs = tf.gather(xs, index)  ##dim=[num,order,batchsize]
	xt = tf.transpose(xt)
	xt = tf.gather(xt, index)
	HO_Xs = tf.reduce_prod(xs, axis=1)
	HO_Xs = tf.reduce_mean(HO_Xs, axis=1)
	HO_Xt = tf.reduce_prod(xt, axis=1)
	HO_Xt = tf.reduce_mean(HO_Xt, axis=1)
	return tf.reduce_mean(tf.square(tf.subtract(HO_Xs, HO_Xt)))
'''

#----------- High-order matching moment  derived by P.Jirayu -----------

#converted torch version
#---HoMM3--- 2D assert to 3D covariance alignment---
def HoMM3_loss(xs, xt):
    xs_ = xs - torch.mean(xs)
    xt_ = xt - torch.mean(xt)
	#adding new dummy dimension (2D-flatten + 2D-dummy = 4D-created)
    xs_ = torch.unsqueeze(xs_, dim=-1)
    xs_ = torch.unsqueeze(xs_, dim=-1)
    xt_ = torch.unsqueeze(xt_, dim=-1)
    xt_ = torch.unsqueeze(xt_, dim=-1)
    #print("torch.size after expanding dummy dimension: ",xs_.size(), xt_.size())
	#adding dimension permutation (swapping numbers of dimension)
    xs_1 = xs_.permute(0 ,2 ,1 ,3)
    xs_2 = xs_.permute(0, 2, 3, 1)
    xt_1 = xt_.permute(0, 2, 1, 3)
    xt_2 = xt_.permute(0, 2, 3, 1)
    HR_Xs = xs_ * xs_1 * xs_2   # dim: b*L*L*L
    HR_Xs_ = torch.mean(HR_Xs)   #dim: L*L*L
    HR_Xt = xt_ * xt_1 * xt_2
    HR_Xt_ = torch.mean(HR_Xt)
    return torch.mean(torch.square(torch.sub(HR_Xs_, HR_Xt_))), HR_Xs, HR_Xt

#---HoMM4---
def HoMM4_loss(xs, xt):
	xs_ = xs - torch.mean(xs)
	xt_ = xt - torch.mean(xt)
	xs_ = torch.unsqueeze(xs_, dim=-1)
	xs_ = torch.unsqueeze(xs_, dim=-1)
	xs_ = torch.unsqueeze(xs_, dim=-1)
	xt_ = torch.unsqueeze(xt_, dim=-1)
	xt_ = torch.unsqueeze(xt_, dim=-1)
	xt_ = torch.unsqueeze(xt_, dim=-1)
	xs_1 = xs_.permute(0, 2, 1, 3, 4)
	xs_2 = xs_.permute(0, 2, 3, 1, 4)
	xs_3 = xs_.permute(0, 2, 3, 4, 1)
	xt_1 = xs_.permute(0, 2, 1, 3, 4)
	xt_2 = xs_.permute(0, 2, 3, 1, 4)
	xt_3 = xs_.permute(0, 2, 3, 4, 1)
	HR_Xs = xs_ * xs_1 * xs_2 * xs_3		# dim: b*L*L*L*L => [31,1,31,31,31]
	HR_Xs_ = torch.mean(HR_Xs)			# dim: L*L*L*L 
	HR_Xt = xt_ * xt_1 * xt_2 * xt_3 
	HR_Xt_ = torch.mean(HR_Xt)
	return torch.mean(torch.square(torch.sub(HR_Xs_, HR_Xt_))), HR_Xs, HR_Xt

#---KHoMM (3rd, 4th order available as you treat in Ho_Xs & Ho_Xt)---
#sigmas = [1e-6, 5e-5, 3e-5, 1e-5, 5e-4, 3e-4, 1e-4, 5e-3, 3e-3]
def KernelHoMM_loss(Ho_Xs,Ho_Xt,sigma=1e-5):
    #https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html
    pairwise_dist = nn.PairwiseDistance(p=2.0, eps=1e-6)
    dist_ss=pairwise_dist(Ho_Xs, Ho_Xs)	#s,s
    dist_tt=pairwise_dist(Ho_Xt, Ho_Xt)	#t,t
    dist_st=pairwise_dist(Ho_Xs, Ho_Xt)	#s,t
    loss=torch.mean(torch.exp(-sigma*dist_ss))+torch.mean(torch.exp(-sigma*dist_tt))-2*torch.mean(torch.exp(-sigma*dist_st))
    #print("KHoMM loss shape: ", loss.shape)
    return loss
    #return torch.where(loss > 0, loss, 0)




#---test (attention+HoMM)--- //worse//
def lina_HoMM3_loss(xs, xt):

    lin_att = lina.EfficientAttention(in_channels=31,key_channels=31,head_count=31,value_channels=31).to('cuda:0')

    xs_ = xs - torch.mean(xs)
    xt_ = xt - torch.mean(xt)
	#adding new dummy dimension (2D-flatten + 2D-dummy = 4D-created)
    xs_ = torch.unsqueeze(xs_, dim=-1)
    xs_ = torch.unsqueeze(xs_, dim=-1)
    xt_ = torch.unsqueeze(xt_, dim=-1)
    xt_ = torch.unsqueeze(xt_, dim=-1)

	#adding dimension permutation (swapping numbers of dimension)
    xs_1 = xs_.permute(0 ,2 ,1 ,3)
    xs_2 = xs_.permute(0, 2, 3, 1)
    xt_1 = xt_.permute(0, 2, 1, 3)
    xt_2 = xt_.permute(0, 2, 3, 1)
    
    HR_Xs = xs_ * xs_1 * xs_2   # dim: b*L*L*L
	#Source Linear Attention
    HR_Xs_lina = lin_att(HR_Xs)
    HR_Xs_ = torch.mean(HR_Xs_lina)   #dim: mean(L*L*L) ==> torch.size([])
    
    HR_Xt = xt_ * xt_1 * xt_2
	#Target Linear Attention
    HR_Xt_lina = lin_att(HR_Xt)
    HR_Xt_ = torch.mean(HR_Xt_lina)

    return torch.mean(torch.square(torch.sub(HR_Xs_, HR_Xt_))), HR_Xs_lina, HR_Xt_lina

#---high-order coral--- //worse//
def coral3_loss(xs, xt):
    d = xs.data.shape[1]

    xs_ = torch.mean(xs, 0, keepdim=True) - xs
    xt_ = torch.mean(xt, 0, keepdim=True) - xt
	#adding new dummy dimension (2D-flatten + 2D-dummy = 4D-created)
    xs_ = torch.unsqueeze(xs_, dim=-1)
    xs_ = torch.unsqueeze(xs_, dim=-1)
    xt_ = torch.unsqueeze(xt_, dim=-1)
    xt_ = torch.unsqueeze(xt_, dim=-1)

	#adding dimension permutation (swapping numbers of dimension)
    xs_1 = xs_.permute(0 ,2 ,1 ,3)
    xs_2 = xs_.permute(0, 2, 3, 1)
    xt_1 = xt_.permute(0, 2, 1, 3)
    xt_2 = xt_.permute(0, 2, 3, 1)

    #gram matrix matching for higher-order covariance matrices
    xs_cov = xs_.permute(0, 3, 1, 2) @ xs_
    xt_cov = xt_.permute(0, 3, 1, 2) @ xt_
    xs1_cov = xs_1.permute(0, 3, 1, 2) @ xs_1
    xt1_cov = xt_1.permute(0, 3, 1, 2) @ xt_1
    xs2_cov = xs_2.permute(0, 3, 1, 2) @ xs_2
    xt2_cov = xt_2.permute(0, 3, 1, 2) @ xt_2

    #3d corr alignment
    x_df = torch.sub(xs_cov, xt_cov)
    x1_df = torch.sub(xs1_cov, xt1_cov)
    x2_df = torch.sub(xs2_cov, xt2_cov)

    Hcov = torch.mul(torch.mul(x_df, x1_df), x2_df)
    loss = torch.mean(Hcov)

    return loss/(8 * d * d * d)