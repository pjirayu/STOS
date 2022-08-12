'''
    args:
    @InProceedings{Yue_2021_Prototypical,
    author = {Yue, Xiangyu and Zheng, Zangwei and Zhang, Shanghang and Gao, Yang and Darrell, Trevor and Keutzer, Kurt and Sangiovanni-Vincentelli, Alberto},
    title = {Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
    }
    ref: https://github.com/zhengzangw/PCS-FUDA/blob/1b82dd088e5475b45688bec44830f3e96ae65d32/pcs/models/ssda.py#L12
'''

# Transfer Cost Function
# Lcls + lamda1 * L_in_self + lamda2 * L_cross_self + lamda3 * Lmim:{L_mim_s + L_mim_t}

import random

import numpy as np
import torch
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#from pcs.utils import reverse_domain, torchutils

#---torchutil in pcs (start)---
# Setup
def set_seed(seed=1234, determine=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if determine:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Tensor & nn
def expand_1d(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def isin(ar1, ar2):
    # for every element of ar2, is ar2 in ar1
    # return shape same to ar1
    return (ar1[..., None] == ar2).any(-1)


def dot(x, y):
    return torch.sum(x * y, dim=-1)

def contrastive_sim(instances, proto=None, tao=0.05):
    # prob_matrix [bs, dim]
    # proto_dim [nums, dim]
    if proto is None:
        proto = instances
    ins_ext = instances.unsqueeze(1).repeat(1, proto.size(0), 1)
    sim_matrix = torch.exp(dot(ins_ext, proto) / tao)
    return sim_matrix


def contrastive_sim_z(instances, proto=None, tao=0.05):
    sim_matrix = contrastive_sim(instances, proto, tao)
    return torch.sum(sim_matrix, dim=-1)


def contrastive_prob(instances, proto=None, tao=0.05):
    sim_matrix = contrastive_sim(instances, proto, tao)
    return sim_matrix / torch.sum(sim_matrix, dim=-1).unsqueeze(-1)


def pairwise_distance_2(input_1, input_2):
    assert input_1.size(1) == input_2.size(1)
    dis_vec = input_1.unsqueeze(1) - input_2
    dis = torch.norm(dis_vec, dim=2)
    return dis

#---torchutil in pcs (end)---

#---K-mean cluster (from kmeans_cluster instead of previous faiss (Not available for cuda 11.1)----
import torch
import numpy as np
from kmeans_pytorch.__init__ import kmeans

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"

'''
#K-means sample
    #N, D = x.shape  # Number of samples, dimension of the ambient space
    #c = x[:K, :].clone()  # Simplistic initialization for the centroids
    #x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    #c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids
# K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
'''

#data example
#data_size, dims, num_clusters = 155, 512, 20    
#x = np.random.randn(data_size, dims) / 6
#x = torch.from_numpy(x)

#--- kmeans cluster ---
'''
    arg:
        X: prediction vector
        cluster_centers: number of cluster centroids
        device: training on CPU or GPU
        distance: for distance cost measurement
'''
#ref: https://subhadarship.github.io/kmeans_pytorch/chapters/example/example/
def torch_kmeans(data:torch.Tensor, num_clusters:int=31, device=torch.device('cuda:0'), tqdm_track=True):
    cluster_labels, cluster_centroids = kmeans(
        X=data, num_clusters=num_clusters, distance='euclidean', device=device, tqdm_flag=tqdm_track, iter_limit=0,
    )
    return cluster_labels, cluster_centroids

def compute_variance(
    data, cluster_labels, centroids, alpha=10, debug=False, num_class=None
):
    """compute variance for proto
    Args:
        data (torch.Tensor): data with shape [n, dim] 
        cluster_labels (torch.Tensor): cluster labels of [n]
        centroids (torch.Tensor): cluster centroids [k, ndim]
        alpha (int, optional): Defaults to 10.
        debug (bool, optional): Defaults to False.
    Returns:
        [type]: [description]
    """

    k = len(centroids) if num_class is None else num_class
    phis = torch.zeros(k)
    for c in range(k):
        cluster_points = data[cluster_labels == c]
        c_len = len(cluster_points)
        if c_len == 0:
            phis[c] = -1
        elif c_len == 1:
            phis[c] = 0.05
        else:
            phis[c] = torch.sum(torch.norm(cluster_points - centroids[c].cuda(), dim=1)) / (
                c_len * np.log(c_len + alpha)
            )
            if phis[c] < 0.05:
                phis[c] = 0.05

    if debug:
        print("size-phi:", end=" ")
        for i in range(k):
            size = (cluster_labels == i).sum().item()
            print(f"{size}[phi={phis[i].item():.3f}]", end=", ")
        print("\n")

    return phis # Variance clusters






#---for MemoryBank---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Original MemoryBank derived from https://github.com/zhengzangw/PCS-FUDA/blob/1b82dd088e5475b45688bec44830f3e96ae65d32/pcs/models/memorybank.py#L8
# Unavailable in our implementation due to error of function in use case got NoneType instead of Torch.tensor since MemoryBank.__init__
'''
class MemoryBank(object):
    """
    For efficiently computing the background vectors.
    
        Args: three main mode: MemoryBank{create, index, save}
            as_tensor <-- _bank <-- _create at __init__ (i.e., as_tensor == _create)
            at_idxs == index vector from memory bank
            update == update new feature vector save to memory bank
    """

    def __init__(self, size, dim):
        """generate random memory bank
        Args:
            size (int): length of the memory bank
            dim (int): dimension of the memory bank features
            device_ids (list): gpu lists
        """
        self._bank = self._create(size, dim)

    def _create(self, size, dim):
        """generate randomized features
        """
        # initialize random weights
        mb_init = torch.rand(size, dim, device=torch.device("cuda:0"))
        std_dev = 1.0 / np.sqrt(dim / 3)
        mb_init = mb_init * (2 * std_dev) - std_dev
        # L2 normalize so that the norm is 1
        mb_init = F.normalize(mb_init)
        return mb_init.detach()  # detach so its not trainable

    def as_tensor(self):
        return self._bank

    def at_idxs(self, idxs):
        return torch.index_select(self._bank, 0, idxs)

    def update(self, indices, data_memory):
        data_dim = data_memory.size(1)
        data_memory = data_memory.detach()
        indices = indices.unsqueeze(1).repeat(1, data_dim)

        self._bank = self._bank.scatter_(0, indices, data_memory)

# memory bank calculation function
def updated_new_data_memory(indices, outputs, memory_bank, m=0.1):
    """Compute new memory bank in indices by momentum
    Args:
        indices: indices of memory bank features to update
        outputs: output of features
        domain (str): 'source', 'target'
    """
    data_memory = torch.index_select(memory_bank, 0, indices)

    outputs = F.normalize(outputs, dim=1)
    new_data_memory = data_memory * m + (1 - m) * outputs
    return F.normalize(new_data_memory, dim=1)
'''

#New simple-prepared MemoryBank
#Derived by P.Jirayu reffered in https://colab.research.google.com/drive/1kHAG2Tu54Ihw0K4aHS6BOob8HMOTpfJS#scrollTo=465aurmC8jVz
def MemoryBank_create(size, dim, device):
    mb_init = torch.rand(size, dim, device=device)
    std_dev = 1.0 / np.sqrt(dim / 3)
    mb_init = mb_init * (2 * std_dev) - std_dev
    # L2 normalize so that the norm is 1
    mb_init = F.normalize(mb_init)
    #print("MemoryBank: {} \nMemoryBank shape: {} \nMemoryBank type: {}"\
    #  .format(mb_init, mb_init.shape, type(mb_init))) # shape: torch.Size([155, 512]), type: <class 'torch.Tensor'>
    return mb_init.detach()

def MemoryBank_idxs(MemoryBank, idxs):
    '''
        Args: Indexing to row/column in a focused tensor
        indices=torch.tensor([1])       # shape: torch.Size([1, 512])
        indices=torch.tensor([0,1])     # shape: torch.Size([2, 512])
        indices=torch.tensor([0,1,10])  # shape: torch.Size([3, 512])
    '''
    return torch.index_select(MemoryBank.cpu().detach(), 0, idxs) #dim{0:horizontal, 1:vertical}

# memory bank calculation function
def updated_new_data_memory(indices, outputs, memory_bank, m=0.1):
    """Compute new memory bank in indices by momentum
    Args:
        indices: indices of memory bank features to update
        outputs: output of features
        domain (str): 'source', 'target'
    """
    data_memory = torch.index_select(memory_bank.cpu().detach(), 0, indices)
    outputs = F.normalize(outputs, dim=1).cpu().detach()
    #print("shape of memory bank: {} / output: {}".format(data_memory.shape, outputs.shape))
    new_data_memory = data_memory * m + (1 - m) * outputs
    return F.normalize(new_data_memory, dim=1)

#update to previous mb
def MemoryBank_update(main_data_memory, indices, new_data_memory):
    '''
        Args: torch.scatter_
        self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    '''
    main_data_memory = main_data_memory.cpu()
    data_dim = new_data_memory.size(1)
    data_memory = new_data_memory.detach()
    indices = indices.unsqueeze(1).repeat(1, data_dim).long()
    # print(type(main_data_memory), type(indices), type(data_memory))
    main_data_memory = main_data_memory.scatter_(0, indices, data_memory)

    return main_data_memory





#---CrossSelf loss---
def _compute_I2C_loss(outputs, cluster_centroids, k_list, n_kmeans, batch_size, t=0.05):
    #init
    loss = torch.Tensor([0]).cuda()

    for each_k_idx, k in enumerate(k_list):
        centroids = cluster_centroids[each_k_idx].cuda()
        phi = t

        p = contrastive_sim(outputs, centroids, tao=phi)
        z = torch.sum(p, dim=-1)  # [bs]
        p = p / z.unsqueeze(1)  # [bs, k]

        cur_loss = -torch.sum(p * torch.log(p)) / batch_size

        loss = loss + cur_loss

    loss /= n_kmeans

    return torch.mean(loss)






#---InSelf (Prototypical Contrastive learning; PC) loss---
def _compute_proto_loss(outputs, indices, cluster_labels, cluster_centroids, cluster_phi, k_list, n_kmeans, t=0.05):
    """Loss PC in essay (part of In-domain Prototypical Contrastive Learning)
        Instance: outputs from memory bank
        Prototype: cluster from feature extractor
    """
    #init
    loss = torch.Tensor([0]).cuda()
        
    for each_k_idx, k in enumerate(k_list):
        # clus info
        labels = cluster_labels[each_k_idx]
        centroids = cluster_centroids[each_k_idx].cuda()
        phis = cluster_phi[each_k_idx]
        #print("Memory cluster for each vector index---> \ncluster_labels:{} \ncluster_centroids:{} \ncluster_variance:{}".format(
        #            labels, centroids.shape, phis))

        # batch info
        #batch_labels = labels[indices]
        #batch_centroids = centroids[batch_labels]
        #if loss_type == "fix":
        #    batch_phis = t
        #else:
        #batch_phis = phis[batch_labels]

        # calculate similarity
        dot_exp = torch.exp(
            #torch.sum(outputs * batch_centroids, dim=-1) / batch_phis
            torch.sum(outputs * centroids, dim=-1) / phis
        )

        assert not torch.isnan(outputs).any()
        #assert not torch.isnan(batch_centroids).any()
        assert not torch.isnan(centroids).any()
        assert not torch.isnan(dot_exp).any()

        # calculate Z
        #all_phi = t if is_fix else 
        all_phi = phis.unsqueeze(0).repeat(outputs.shape[0], 1)
        #z = contrastive_sim_z(outputs, centroids, tao=all_phi)
        z = contrastive_sim_z(outputs, centroids, tao=all_phi.cuda())

        # calculate loss
        p = dot_exp / z

        loss = loss - torch.sum(torch.log(p)) / p.size(0)

    loss /= n_kmeans

    return torch.mean(loss)



# Mutual Information Maximization (MIM)
class MomentumSoftmax:
    def __init__(self, num_class, m=1.):
        self.softmax_vector = torch.zeros(num_class).detach() + 1.0 / num_class
        self.m = m
        self.num = m

    def update(self, mean_softmax, num=1):
        self.softmax_vector = (
            (self.softmax_vector * self.num) + mean_softmax * num
        ) / (self.num + num)
        self.num += num

    def reset(self):
        # print(self.softmax_vector)
        self.num = self.m
        D = torch.mm(W, W.transpose(0, 1))
        D_mask = 1 - torch.eye(self.num_class).cuda()
        return torch.sum(D * D_mask).item()