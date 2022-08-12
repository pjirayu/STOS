'''
all-in-one package pytorch for importing

import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
'''

import torch
import torch.nn as nn

def adjust_learning_rate(optimizer, epoch, args):
    # decayed lr by 10 every 20 epochs
    if (epoch+1)%20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.rate

from PIL import Image
def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))

#cudnn.benchmark = True

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot

#utils library
#https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/f42f4786ecaf94f8c2e537c11648d90ecf66b9dc/utils.py#L21

def accuracy(output, target):

    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    accuracy = correct/total

    return accuracy

def imshow(image_tensor, mean, std, title=None):
    """
    Imshow for normalized Tensors.
    Useful to visualize data from data loader
    """

    image = image_tensor.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes

def adjust_learning_rate(optimizer, epoch, lr_starting, step1=30, step2=60, step3=90):
    """decrease the learning rate at n1, n2 and n3 step"""
    lr = lr_starting
    if epoch >= step1:
        lr /= 10.
    if epoch >= step2:
        lr /= 10.
    #if epoch >= step3:
    #    lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Stoper():

    def __init__(self, early_step):
        self.max = 0
        self.cur = 0
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1

    def judgesave(self):
        if self.cur > self.max:
            return True
        else:
            return False

    def judge(self):
        if self.cur > self.max:
            self.max = self.cur
            self.maxindex = self.curindex
        if self.curindex - self.maxindex >= self.early_step:
            return True
        else:
            return False

    def showfinal(self):
        result = "AUC {}\n".format(self.max)
        print(result)
        return self.max

def print_cuda_statistics(nvidia_smi=True, output=print):
    output(f"Python VERSION: {sys.version}")
    output(f"pytorch VERSION: {torch.__version__}")
    output(f"CUDA VERSION: {torch.version.cuda}")
    output(f"CUDNN VERSION: {torch.backends.cudnn.version()}")
    output(f"Device NAME: {torch.cuda.get_device_name(0)}")
    output(f"Number CUDA Devices: {torch.cuda.device_count()}")
    output(f"Available devices: {torch.cuda.device_count()}")
    output(f"current CUDA Device: {torch.cuda.current_device()}")

    if nvidia_smi:
        print("nvidia-smi:")
        call(
            [
                "nvidia-smi",
                "--format=csv",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
            ]
        )


def onehotembedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 