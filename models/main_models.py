import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torchvision import models
from models.BasicModule import BasicModule

import math
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from .bam import *
from .cbam import *
from .sam import *
from .featurefusion import *

#--------------------FADA arch (baseline)--------------------------
class DCD(BasicModule):
    def __init__(self,h_features=64,input_features=128):
        super(DCD,self).__init__()
        self.fc1=nn.Linear(input_features,h_features)
        self.fc2=nn.Linear(h_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        out=F.relu(self.fc2(out))
        return F.softmax(self.fc3(out),dim=1)


class Classifier(BasicModule):
    def __init__(self,input_features=64, num_classes=10):
        super(Classifier,self).__init__()
        self.fc=nn.Linear(input_features, num_classes)

    def forward(self,input):
        #print("Flatten layer input tensor shape: ", input.shape)
        return F.softmax(self.fc(input),dim=1)
        #print("output layer tensor shape: ", out.shape)

#input size required: 28x28
class Encoder(BasicModule):
    def __init__(self):
        super(Encoder,self).__init__()

        self.conv1=nn.Conv2d(1,6,5)
        #self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,64)

    def forward(self,input):
        out=F.relu(self.conv1(input))
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)

        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)

        return out


#------------------------For object recognition----------------------------
# alexnet backbone model: https://github.com/jindongwang/transferlearning/blob/master/code/feature_extractor/for_image_data/backbone.py
class AlexNet_Encoder(nn.Module):
    def __init__(self, pretrained : bool = True, num_classes: int = 10, att_type: str = 'bam'):
        super(AlexNet_Encoder, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        #self.classifier = nn.Sequential()
        #for i in range(num_classes):
        #    self.classifier.add_module(
        #        "classifier"+str(i), model_alexnet.classifier[i])
        #self.__in_features = model_alexnet.classifier[num_classes].in_features

        if att_type=='finalbam':
            self.bam4 = BAM(9216)
            self.sam = None
        elif att_type=='sam':
            self.sam = Self_Attention(9216, 'relu')
            self.bam4 = None
        elif att_type=='orcat':
            self.bam4 = BAM(9216)
            self.sam = Self_Attention(9216, 'relu')
            self.fusion = FeatureFusionModule(in_chan=9216*2, out_chan=9216).to('cuda:0')
        else:
            self.sam =  None
            self.bam4 = None

    def forward(self, x):
        x = self.features(x)    # input= b x 3 x 227 x 227
        #print(x.shape)
        if self.bam4 is not None and self.sam is None:
            out = self.bam4(x)
            x = out[0]
        elif self.sam is not None and self.bam4 is None:
            out = self.sam(x)
            x = out[0]
        elif self.bam4 is not None and self.sam is not None:
            out_bam4 = self.bam4(x)
            out_sam = self.sam(x)
            x = self.fusion(out_bam4[0], out_sam[0])

        x = x.view(x.size(0), 256*6*6)  # flattening from b x 256 x 6 x 6 to 9216
        #x = x.view(x.size(0), -1) # optional
        #x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features



#resnet-backbone model: https://github.com/thuml/Transfer-Learning-Library/blob/master/common/vision/models/resnet.py

#input size required: 224x224, block.expansion == 1 for resnet18, 34
class ResNet18_Encoder(nn.Module):
    def __init__(self, pretrained : bool = True, require_flatten: bool = True, att_type: str = 'bam'):
        super(ResNet18_Encoder, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool

        self.layer1 = model_resnet18.layer1
        #self.layer1.weight.data.normal_(0, 0.01)
        #self.layer1.bias.data.fill_(0.0)

        self.layer2 = model_resnet18.layer2
        #self.layer2.weight.data.normal_(0, 0.01)
        #self.layer2.bias.data.fill_(0.0)

        self.layer3 = model_resnet18.layer3
        #self.layer3.weight.data.normal_(0, 0.01)
        #self.layer3.bias.data.fill_(0.0)

        self.layer4 = model_resnet18.layer4
        #self.layer4.weight.data.normal_(0, 0.005)
        #self.layer4.bias.data.fill_(0.0)

        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features
        self.require_flatten = require_flatten

        if att_type=='bam':
            self.bam1 = BAM(64 * 1)
            self.bam2 = BAM(128 * 1)
            self.bam3 = BAM(256 * 1)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        if att_type=='finalbam':
            self.bam4 = BAM(512 * 1)
            self.sam = None
        elif att_type=='sam':
            self.sam = Self_Attention(512, 'relu')
            self.bam4 = None
        elif att_type=='orcat':
            self.bam4 = BAM(512 * 1)
            self.sam = Self_Attention(512, 'relu')
            self.fusion = FeatureFusionModule(in_chan=1024, out_chan=512).to('cuda:0')
        else:
            self.sam =  None
            self.bam4 = None

        if att_type=='cbam':
            self.cbam1 = CBAM(64 * 1, 16)
            self.cbam2 = CBAM(128 * 1, 16)
            self.cbam3 = CBAM(256 * 1, 16)
            self.cbam4 = CBAM(512 * 1, 16)
        else:
            self.cbam1, self.cbam2, self.cbam3, self.cbam4 = None, None, None, None

    def forward(self, input):
        #residual = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)          # D(layer1 == ch_att1 == spar_att1) ==  64, 56, 56
        
        if self.bam1 is not None:
            out1, f_att1, ch_att1, spar_att1 = self.bam1(out)
            out = self.layer2(out1)
        else:
            out = self.layer2(out)      # D(layer2 == ch_att2 == spar_att2) ==  128, 28, 28

        if self.bam2 is not None:
            out2, f_att2, ch_att2, spar_att2 = self.bam2(out)
            out = self.layer3(out2)     # D(layer3 == ch_att3 == spar_att3) ==  256, 14, 14
        else:
            out = self.layer3(out)

        if self.bam3 is not None:
            out3, f_att3, ch_att3, spar_att3 = self.bam3(out)
            out = self.layer4(out3)    # D(layer4) ==  512, 7, 7
        else:
            out = self.layer4(out)

        if self.bam4 is not None and self.sam is None:
            out = self.bam4(out)        # output from bam4 (2nd-4th): f_att, ch_att, spar_att
        elif self.sam is not None and self.bam4 is None:
            out = self.sam(out)         # output from sam (2nd): p_att
        elif self.bam4 is not None and self.sam is not None:
            out_bam4 = self.bam4(out)
            out_sam = self.sam(out)
            out = self.fusion(out_bam4, out_sam)    #attn_fusion(normal_feature_resnet) == 512, 7, 7
            #print("post-orcat feature", out.shape)

        #if self.bam4 is not None:
        #    out4, f_att4, ch_att4, spar_att4 = self.bam4(out)

        #print("out before flattening: ", out.shape)
        if self.require_flatten==True:
            out = self.avgpool(out)     #avgpool(attn_feature) = 512, 1, 1
            out = out.view(out.size(0), -1)
        else:
            pass
        #print("feature flattening: ", out.view(out.size(0), -1).shape)
        #f_ch_att = [ch_att1, spar_att1, ch_att2, spar_att2, ch_att3, spar_att3]
        #return out, f_ch_att
        #f_att = [f_att1, f_att2, f_att3, f_att4]
        #refined_feature = [out1, out2, out3, out4]
        return out

    def __output_features(self):
        return self.__in_features


class ResNet34_Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34_Encoder, self).__init__()
        model_resnet34 = models.resnet34(pretrained=True)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool
        self.__in_features = model_resnet34.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


#input size required: 224x224, block.expansion == 4 for resnet20, 101, 152
class ResNet50_Encoder(nn.Module):
    def __init__(self, pretrained : bool = True, require_flatten: bool = True, att_type: str = 'bam'):
        super(ResNet50_Encoder, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features
        self.require_flatten = require_flatten

        if att_type=='bam':
            self.bam1 = BAM(64 * 4)
            self.bam2 = BAM(128 * 4)
            self.bam3 = BAM(256 * 4)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        if att_type=='finalbam':
            self.bam4 = BAM(512 * 4)
            self.sam = None
        elif att_type=='sam':
            self.sam = Self_Attention(2048, 'relu')
            self.bam4 = None
        elif att_type=='orcat':
            self.bam4 = BAM(512 * 4)
            self.sam = Self_Attention(2048, 'relu')
            self.fusion = FeatureFusionModule(in_chan=4096, out_chan=2048).to('cuda:0')
        else:
            self.sam =  None
            self.bam4 = None

        if att_type=='cbam':
            self.cbam1 = CBAM(64 * 4, 16)
            self.cbam2 = CBAM(128 * 4, 16)
            self.cbam3 = CBAM(256 * 4, 16)
            self.cbam4 = CBAM(512 * 4, 16)
        else:
            self.cbam1, self.cbam2, self.cbam3, self.cbam4 = None, None, None, None

    def forward(self, out):
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)          # 256, 56, 56
        #out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        if self.bam1 is not None:
            out1, f_att1, ch_att1, spar_att1 = self.bam1(out)
            out = self.layer2(out1)
        else:
            out = self.layer2(out)      # D(layer2 == ch_att2 == spar_att2) ==  512, 28, 28

        if self.bam2 is not None:
            out2, f_att2, ch_att2, spar_att2 = self.bam2(out)
            out = self.layer3(out2)     # D(layer3 == ch_att3 == spar_att3) ==  1024, 14, 14
        else:
            out = self.layer3(out)

        if self.bam3 is not None:
            out3, f_att3, ch_att3, spar_att3 = self.bam3(out)
            out = self.layer4(out3)    # D(layer4) ==  2048, 7, 7
        else:
            out = self.layer4(out)

        if self.bam4 is not None and self.sam is None:
            out = self.bam4(out)            # output from bam4 (2nd-4th): f_att, ch_att, spar_att
        elif self.sam is not None and self.bam4 is None:
            out = self.sam(out)      # output from sam (2nd): p_att
        elif self.bam4 is not None and self.sam is not None:
            out_bam4 = self.bam4(out)
            out_sam = self.sam(out)
            out = self.fusion(out_bam4, out_sam)

        #print("out before flattening: ", out.shape)
        if self.require_flatten==True:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
        else:
            pass
        #print("out after flattening: ", out.view(out.size(0), -1).shape)
        return out

    def output_num(self):
        return self.__in_features


class ResNet101_Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet101_Encoder, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.__in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet152_Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet152_Encoder, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)
        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool
        self.__in_features = model_resnet152.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet_Classifier(BasicModule):
    #def __init__(self, fc=512, num_classes=1000):
    #    super(ResNet_Classifier,self).__init__()
    #    self.fc1=nn.Linear(fc, 512)    # 512 for resnet18, 34  _   2048 for resent50, 101, 152
    #    self.fc2=nn.Linear(512, num_classes)      # n=10 for hand   _   n=31 for office-31

    def __init__(self, num_classes=1000, require_flatten=False):
        super(ResNet_Classifier,self).__init__()
        self.fc=nn.Linear(512, num_classes)    # 512 for resnet18, 34  _   2048 for resent50, 101, 152
        self.fc.weight.data.normal_(0, 0.005)  #mean, std
        self.require_flatten = require_flatten

    def forward(self,input):
        #print("Flatten layer input tensor shape: ", input.shape)
        if self.require_flatten == True:
            out = input.view(input.size(0), -1)
            out = F.softmax(self.fc(out), dim=1)
        else:
            out = F.softmax(self.fc(input), dim=1)
        #print("output layer tensor shape: ", out.shape)
        return out

#input size required: 512 for resnet18, 2048 for resnet50
#--------------------------------------------------------------------------------------------------------
# /// softtriple ///
'''
    Arg: 
        PyTorch Implementation for Our ICCV'19 Paper: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling"
        Referred to in https://github.com/idstcv/SoftTriple
        @inproceedings{qian2019striple,
          author    = {Qi Qian and
                       Lei Shang and
                       Baigui Sun and
                       Juhua Hu and
                       Hao Li and
                       Rong Jin},
          title     = {SoftTriple Loss: Deep Metric Learning Without Triplet Sampling},
          booktitle = {{IEEE} International Conference on Computer Vision, {ICCV} 2019},
          year      = {2019}
        }
'''
class SoftTriple_Embedding(nn.Module):
    def __init__(self, flatten=512, dim=64):
        super(SoftTriple_Embedding, self).__init__()
        self.embedding = nn.Linear(flatten, dim)
        self.embedding.weight.data.normal_(0, 0.5)
        #self.embedding.weight.data.normal_(0,0.005)

    def forward(self, input):
        adaptiveAvgPoolWidth = input.shape[2]
        x = F.avg_pool2d(input, kernel_size=adaptiveAvgPoolWidth)   #H x W ==> 1 x 1
        #print("feature after avg_pool2d: ", x.view(x.size(0), -1).shape)   #[31, 512]
        x = x.view(x.size(0), -1)   # flattening
        #print("feature flattening: ", x.view(x.size(0), -1).shape) #[31, 512]
        #---Note: avgpool2d --> size(-1) == flattening
        fc = self.embedding(x)              #fc = [31, 64]
        x = F.normalize(fc, p=2, dim=1)
        #print("feature normalized flattening: ", x.view(x.size(0), -1).shape)  #[31, 1000]
        return x


# --- combination of classification with softtriple ---
class SoftTriple_Classifier(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(SoftTriple_Classifier, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = nn.Parameter(torch.Tensor(dim, cN*K))
        self.fc.data.normal_(0, 0.005)  #mean, std
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target, mode='triplet'):
        #computing softed similarity
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        
        #softtriple output; S(example;x_i, prob; j)
        if mode=='triplet':
            marginM = torch.zeros(simClass.shape).cuda()
            marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
            outputs = self.la*(simClass-marginM)
        elif mode=='non-triplet':
            outputs = self.la*(simClass)
        else:
            raise ValueError("error module assessment mode")
        #print("(shape) centers:{}, simInd:{}, simStruc:{}, prob:{}, simClass:{}, outputs:{}".format(
        #    centers.shape, simInd.shape, simStruc.shape, prob.shape, simClass.shape, outputs.shape))
        return outputs


#For loss calc. from SoftTirple CE classification
class CE_softtriple_criterion(nn.Module):
    def __init__(self, tau, dim, cN, K):
        super(CE_softtriple_criterion, self).__init__()
        self.tau = tau
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K)).cuda()
        self.fc.data.normal_(0,0.005)   #mean, std
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, output_classifier, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean'):
        centers = F.normalize(self.fc, p=2, dim=0)
        lossClassify = F.cross_entropy(output_classifier, target, weight, size_average, ignore_index, reduce, reduction)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify + self.tau*reg
        else:
            return lossClassify


#--------------------------------------------------------------------------------------------------------

# /// Domain classification (GRL-based) ///
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

#--default simple reversal layer--
'''
class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()
        self.Domain_Classifier = nn.Sequential()
        #output feature resnet layer4
        self.Domain_Classifier.add_module('d_fc1', nn.Linear(512, 100))
        self.Domain_Classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.Domain_Classifier.add_module('d_lrelu1', nn.LeakyReLU(0.2))
        #self.Domain_Classifier.add_module('d_relu1', nn.ReLU(True))

        self.Domain_Classifier.add_module('d_fc2', nn.Linear(100, 2))
        #self.Domain_Classifier.add_module('d_fc2', nn.Linear(100, 31))
        self.Domain_Classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, feature, alpha):
        #feature = feature.view(-1, 512)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.Domain_Classifier(reverse_feature)
        return domain_output
'''
class Domain_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_size:int=256, domainclassifier_head:str='dann', num_classes:int=10):
        super(Domain_Classifier, self).__init__()
        self.Domain_Classifier = nn.Sequential()
        #output feature resnet layer4
        self.Domain_Classifier.add_module('d_fc1', nn.Linear(in_channels, hidden_size))
        self.Domain_Classifier.add_module('d_bn1', nn.BatchNorm1d(hidden_size))
        self.Domain_Classifier.add_module('d_lrelu1', nn.LeakyReLU(0.2))

        self.Domain_Classifier.add_module('d_fc2', nn.Linear(hidden_size, hidden_size//2))
        self.Domain_Classifier.add_module('d_bn2', nn.BatchNorm1d(hidden_size//2))
        self.Domain_Classifier.add_module('d_lrelu2', nn.LeakyReLU(0.2))

        #self.Domain_Classifier.add_module('d_fc2', nn.Linear(100, 31))
        if domainclassifier_head=='dann':
            self.Domain_Classifier.add_module('d_fc3', nn.Linear(hidden_size//2, 2))
            self.Domain_Classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
            #self.Domain_Classifier.add_module('d_sigmoid', nn.Sigmoid())
            self.Domain_Classifier.apply(init_weights)
        elif domainclassifier_head=='mdd':
            self.Domain_Classifier.add_module('d_fc3', nn.Linear(hidden_size//2, num_classes))

    def forward(self, feature_flatten, alpha):
        #feature = feature.view(-1, 512)
        reverse_feature = ReverseLayerF.apply(feature_flatten, alpha)
        domain_output = self.Domain_Classifier(reverse_feature)
        return domain_output

#--------------------

#---for MME, PCS---
def grad_reverse(x, lambd=1.0):
    return ReverseLayerF.apply(x, lambd)

class CosineClassifier(nn.Module):
    def __init__(self, num_class=31, inc=512, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=True, eta=0.1):
        self.normalize_fc()
        #print("input shape: ",x.shape)  #[31, 512]
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x)
        x_out = x_out / self.temp

        return x_out

    def normalize_fc(self):
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, eps=1e-12, dim=1)

    @torch.no_grad()
    def compute_discrepancy(self):
        self.normalize_fc()
        W = self.fc.weight.data

#-------------


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)





####################### All deploying model on above ###########################
####################### All testing model in below #############################







#ensemble_model
'''
# Combine models: https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383/2
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x
'''

class Plain_Ensemble_Models(nn.Module):
    def __init__(self, classifier, embedding, encoder, mode='en_em_cls'):
        super(Plain_Ensemble_Models, self).__init__()
        self.classifier = classifier
        self.embedding = embedding
        self.encoder = encoder
        self.mode = mode
        
    def forward(self, inputs):
        if self.mode == 'en_em':
            outputs = self.embedding(self.encoder(inputs))
        elif self.mode == 'em_cls':
            outputs = self.classifier(self.embedding(inputs))
        elif self.mode == 'em_cls':
            outputs = self.classifier(self.embedding(self.encoder(inputs)))
        else:
            print(" Please select one either ensmeble type from 'en_em', 'em_cls', 'en_em_cls' ")
        
        return outputs

# ----------------------

#///////Film layer that performs per-channel affine transformation (test)//////
class CatFilm(nn.Module):
    """Film layer that performs per-channel affine transformation."""
    def __init__(self, planes):
        super(CatFilm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, planes))
        self.beta = nn.Parameter(torch.zeros(1, planes))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return gamma * x + beta


class BasicBlockFilm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.film1 = CatFilm(planes)
        self.film2 = CatFilm(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
#//////////////////////////////////////////////////////////////////////////////





#------------ Maintaining Discrimination and Fairness for Class Incremental Learning ---------------
# code: https://github.com/hugoycj/Incremental-Learning-with-Weight-Aligning
'''
# Exemplary module
class WALinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sub_num_classes = self.out_features//5
        self.WA_linears = nn.ModuleList()
        self.WA_linears.extend([nn.Linear(self.in_features, self.sub_num_classes, bias=False) for i in range(5)])

    def forward(self, x):
        out1 = self.WA_linears[0](x)
        out2 = self.WA_linears[1](x)
        out3 = self.WA_linears[2](x)
        out4 = self.WA_linears[3](x)
        out5 = self.WA_linears[4](x)

        return torch.cat([out1, out2, out3, out4, out5], dim = 1)
    
    def align_norms(self, step_b):
        # Fetch old and new layers
        new_layer = self.WA_linears[step_b]
        old_layers = self.WA_linears[:step_b]
        
        # Get weight of layers
        new_weight = new_layer.weight.cpu().detach().numpy()
        for i in range(step_b):
            old_weight = np.concatenate([old_layers[i].weight.cpu().detach().numpy() for i in range(step_b)])
        print("old_weight's shape is: ",old_weight.shape)
        print("new_weight's shape is: ",new_weight.shape)

        # Calculate the norm
        Norm_of_new = np.linalg.norm(new_weight, axis=1)
        Norm_of_old = np.linalg.norm(old_weight, axis=1)
        assert(len(Norm_of_new) == 20)
        assert(len(Norm_of_old) == step_b*20)
        
        # Calculate the Gamma
        gamma = np.mean(Norm_of_new) / np.mean(Norm_of_old)
        print("Gamma = ", gamma)

        # Update new layer's weight
        updated_new_weight = torch.Tensor(gamma * new_weight).cuda()
        print(updated_new_weight)
        self.WA_linears[step_b].weight = torch.nn.Parameter(updated_new_weight)
'''

#For Weight-Alignment Normalization Embedding Module
def feature_align_norms(unused_w_old, unused_w_new, axis=None):
       
        # Get weight of layers
        w_old = unused_w_old.cpu().detach().numpy()
        w_new = unused_w_new.cpu().detach().numpy()

        # Calculate the norm
        Norm_of_w_old = np.linalg.norm(w_old, axis)
        Norm_of_w_new = np.linalg.norm(w_new, axis)
        #assert(len(Norm_of_w_X1) == dim)
        #assert(len(Norm_of_w_X2) == dim)
        
        # Calculate the Gamma
        #gamma_X1 = np.mean(Norm_of_w_X1) / np.mean(Norm_of_w_X2)
        gamma_new = np.mean(Norm_of_w_new) / np.mean(Norm_of_w_old)
        #print("Gamma X1=N/A X2={}".format(gamma_new))

        # Update new layer's weight
        #updated_new_w_X1 = torch.Tensor(gamma_X1 * w_X1).cuda()
        updated_new_weightalign = torch.Tensor(gamma_new * w_new).cuda()

        return updated_new_weightalign


#---clustering---
def extract_vector(path):
    resnet_feature_list = []

    for im in glob.glob(path):

        im = cv2.imread(im)
        im = cv2.resize(im,(224,224))
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = my_new_model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())

    return np.array(resnet_feature_list)






