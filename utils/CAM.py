import os
import cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as trans
import numpy as np
import matplotlib.pyplot as plt

#On-Available
def return_CAM(feature_conv, weight, class_idx):
    """
    return_CAM generates the CAMs and up-sample it to 224x224
    arguments:
    feature_conv: the feature maps of the last convolutional layer
    weight: the weights that have been extracted from the trained parameters
    class_idx: the label of the class which has the highest probability
    """
    size_upsample = (256, 256)
    
    # we only consider one input image at a time, therefore in the case of 
    # VGG16, the shape is (1, 512, 7, 7)
    bz, nc, h, w = feature_conv.shape 
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))# -> (512, 49)
        #print("Weight_idx shape : ", weight[idx].shape,"beforeDot shape: ", beforeDot.shape)
        cam = np.matmul(weight[idx], beforeDot) # -> (1, 512) x (512, 49) = (1, 49)
        cam = cam.reshape(h, w) # -> (7 ,7)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

#---for CAM (renew1)---
#import numpy as np
#from PIL import Image
#from matplotlib import cm
#from matplotlib import pyplot as plt

#import torchvision
#import torchvision.transforms as transforms

#def _normalizer(denormalize=False):
#    MEAN = [0.485, 0.456, 0.406]
#    STD = [0.229, 0.224, 0.225]    
    
#    if denormalize:
#        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
#        STD = [1/std for std in STD]
    
#    return transforms.Normalize(mean=MEAN, std=STD)

#def _transformer(imsize=None, cropsize=None):
#    transformer = []
#    if imsize:
#        transformer.append(transforms.Resize(imsize))
#    if cropsize:
#        transformer.append(transforms.CenterCrop(cropsize))
#    transformer.append(transforms.ToTensor())
#    transformer.append(_normalizer())
#    return transforms.Compose(transformer)

#def imload(path, imsize=None, cropsize=None):
#    transformer = _transformer(imsize=imsize, cropsize=cropsize)
#    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)

#def imsave(path, tensor):
#    denormalize = _normalizer(denormalize=True)    
#    if tensor.is_cuda:
#        tensor = tensor.cpu()
#    tensor = torchvision.utils.make_grid(tensor)
#    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)    
#    return None

#def imshow(tensor):
#    denormalize = _normalizer(denormalize=True)    
#    if tensor.is_cuda:
#        tensor = tensor.cpu()    
#    tensor = torchvision.utils.make_grid(denormalize(tensor.squeeze()))
#    image = torchvision.transforms.functional.to_pil_image(tensor)
#    return image

#def array_to_cam(arr):
#    cam_pil = Image.fromarray(np.uint8(cm.gist_earth(arr)*255)).convert("RGB")
#    return cam_pil

#def blend(image1, image2, alpha=0.75):
#    return Image.blend(image1, image2, alpha)

#---GradCAM---
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#from torchvision import models
from skimage.io import imread
from skimage.transform import resize    

# Not available
# ref: https://medium.com/the-owl/gradcam-in-pytorch-7b700caa79e5
class GradCamModel(nn.Module):
    def __init__(self, feature_extractor, embedding, classifier, att_type: str = 'bsfusion'):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        #self.pretrained = models.resnet50(pretrained=True)
        self.pretrained = feature_extractor
        self.embedding = embedding
        self.classifier = classifier
        #selecting attention layer
        if att_type=='bsfusion': finalconv_name = 'fusion'
        elif att_type=='sam': finalconv_name = 'sam'
        elif att_type=='finalbam': finalconv_name = 'bam4'
        elif att_type=='n': finalconv_name='layer4'
        else: raise ValueError('At least one att_type in require [bsfusion, sam, finalbam, n]')

        self.layerhook.append(self.pretrained._modules.get(finalconv_name).register_forward_hook(self.forward_hook()))
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
    
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self,x):
        out = self.pretrained(x)
        out = self.embedding(out)
        out = self.classifier(out, None, mode='eval')
        return out, self.selected_out

# Not Available
# ref: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
from torchvision.models import resnet18, resnet50
class GradcamNet(nn.Module):
    def __init__(self, feature_extractor, embedding, classifier, att_type:str='bsfusion'):
        super(GradcamNet, self).__init__()

        #Resnet18 from pytorch
        self.resnet18 = resnet18(pretrained=True)
        
        # get the pretrained DenseNet201 network
        self.net = feature_extractor
        
        # disect the network to access its attention layer
        if att_type=='bsfusion': finalconv_name = 'fusion'
        elif att_type=='sam': finalconv_name = 'sam'
        elif att_type=='finalbam': finalconv_name = 'bam4'
        elif att_type=='n': finalconv_name='layer4'
        else: raise ValueError('At least one att_type in require [bsfusion, sam, finalbam, n]')
        self.features_conv = self.net._modules.get(finalconv_name)
        
        # add the average global pool
        # self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # get the classifier of the resnet18
        self.embedding = embedding
        self.classifier = classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.net(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # don't forget the pooling
        #x = self.global_avg_pool(x)
        #x = x.view((1, 512))
        x = self.embedding(x)
        x = self.classifier(x, None, mode='eval')
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)