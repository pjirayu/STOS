import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np
import pandas as pd
import dataloader
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os.path, os
import tqdm
import higher
import json
import scipy
import cv2
import time

from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from utils import kd, mmd, entropy, HoMM, coral, fourier_transform, transform_img, bsp, mixup, SCDA, mcc, mme, pcs, rmse
from models import main_models, featurefusion
from models.BasicModule import BasicModule
#from torch.utils.tensorboard import SummaryWriter

#hyper-parameter
parser=argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='resnet50',choices=['resnet18','resnet50', 'alexnet'])   #our work run on resnet-18 and -50 only
parser.add_argument('--n_epoches',type=int,default=100)
parser.add_argument('--n_target_samples',type=int,default=5)    #K-shot determines a few source-target instance set in adaptation task
parser.add_argument('--batch_size',type=int,default=31)                 # bs = num_classes (according to the limitation of HoMM method and fair on eval. metrics)
parser.add_argument('--mini_batch_size_g_h',type=int,default=31)        # bs = num_classes (according to the limitation of HoMM method and fair on eval. metrics)
parser.add_argument('--data_type',type=str,default='office31', choices=['office31', 'officehome', 'imageclef', 'menace', 'crystallization'])    # available on office-31 only in this test repo
parser.add_argument('--source',type=str,default='amazon', choices=['amazon', 'dslr','webcam','synthetic',
                                                                   'art','clipart','product','realworld',
                                                                   'b', 'c', 'i', 'p',
                                                                   'virtual', 'physical',
                                                                   'vir_polymer', 'phy_sugar'])
parser.add_argument('--target',type=str,default='webcam', choices=['amazon', 'dslr','webcam','synthetic',
                                                                   'art','clipart','product','realworld',
                                                                   'b', 'c', 'i', 'p',
                                                                   'virtual', 'physical',
                                                                   'vir_polymer', 'phy_sugar'])     # available on only amazon, webcam, and dslr in this test repo if merging synthetic domain from modern-office31 is able to do so.
parser.add_argument('--adjust_lr',type=bool,default=False)

#regularizing domain-specific representation & statistic criterion
parser.add_argument('--tf_inv_loss', default='spectralcoral', type=str, choices=['coral','logcoral','n','HoMM3','HoMM4','KHoMM3','KHoMM4','linaKHoMM3', 'linaHoMM3', 'mmd','mcc','dan','mdd','robustcoral','spectralcoral','mkspectralcoral'],
                    help='statistic criterion loss (choosing one from parser)')
parser.add_argument('--tf_module_loss', default='n', type=str, choices=['dann','mdd','mme','pcs','n'],
                    help='need gradient reversal layer or none')
parser.add_argument('--robust_order', default=6, type=int, help='a number of iteration for matrix-loop computing (available for robust/spectral coral only)')

#model preparation & data augmentation
parser.add_argument('--mutation', default='n', type=str, choices=['r','n'],
                    help='Mutating source instance as target style')
parser.add_argument('--mutation_style', default='mixup', type=str, choices=['fft','blend','replace','mixup','cutmix','cutmixup','mixvary'],
                    help='style for trying blending source & target instances')
parser.add_argument('--da_type', default='UDA', type=str, choices=['UDA','SDA'],
                    help='target instances with none of any target label(UDA) or the full target labels(SDA)')
parser.add_argument('--att_type', default='orcat', type=str, choices=['bam','finalbam','orcat','sam','n'],
                    help='finalbam (4th layer) to attention, bam1st-3rd layer, sam, orcat(i.e., 4th layer of bam-sam in fusion), or none')
parser.add_argument('--alpha_mix', default=0.2, type=float, help='weight for mixing up between source & target instance in pre-processing determined interval of [0, 1]')


#args follows as FullSoftTripletLoss
#Our framework implement SimpleSoftTripletLoss (w/o regularizer term), we will use regularization on transfer learning loss instead.
parser.add_argument('--dim', default=64, type=int,
                    help='the dimensionality of embeddings (pre-output layer), i.e., n_dim = K_center times to n_classes')
parser.add_argument('--flatten', default=512, type=int, choices=[512, 2048],
                    help='flattening from encoder to st_embedding: resnet18,34 for 512 and resnet50, 101, 152 for 2048')
#for softmax operator term (pulling smooth softmax)
parser.add_argument('--la', default=5, type=float, #default=20 from default vanilla ST paper
                    help='lambda; λ is predefined scaling factor for SoftMaxOperator in training process')
parser.add_argument('--gamma', default=0.1, type=float, #default=0.1 from default vanilla ST paper
                    help='gamma; γ is predefined scaling factor for similarity between the example; x_i and the class; c')
parser.add_argument('--margin', default=0.01, type=float,   #default=0.01 from default vanilla ST paper
                    help='a margin δ can reserve the large margin property on the original triplet constraints')
#for controlling regularizer term
parser.add_argument('--tau', default=0., type=float,       #default= 0.2 --- We would improve the model in domain-invariant regularization term instead.
                    help='tau to control regularizer (weight softtriplet improvement); R(.)')
parser.add_argument('--C', default=31, type=int,           #N(classes)---31:office31---65:officehome---12:imageclef----6:menace---5:crystallization
                    help='C for clustering assignment (==num_class)')
parser.add_argument('--K', default=5, type=int,           
                    help='K for defining number of centers in a class (multiple centers a class > 1)')

#---visualization---
#parser.add_argument('--is_cm', default=True, type=bool, help='Confusion matrix required')
#parser.add_argument('--is_log', default=True, type=bool, help='Validation metrics saved as in log files (acc. & loss)')

opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

#pre-requisite
torch.backends.cudnn.enabled = True
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

 #-----------------------Summary writer (acc, loss) on log---------------
#logdir = 'output/log_eval'
#writer = SummaryWriter(logdir)

#----------------------------pretrain g and h for step 1------------------
#train_dataloader=dataloader.amazon_dataloader(batch_size=opt['batch_size'], image_size=224)
#test_dataloader=dataloader.amazon_dataloader(batch_size=opt['batch_size'], image_size=224)

#---model and image preparation---
if opt['model']=='alexnet': image_size=227
elif opt['model']=='resnet50': image_size=224
elif opt['model']=='resnet18': image_size=224
else: image_size=224

if opt['model']=='resnet18':
    encoder=main_models.ResNet18_Encoder(pretrained=True, require_flatten=False, att_type=opt['att_type']).to(device)
    st_embedding=main_models.SoftTriple_Embedding(flatten=512, dim=opt['dim']).to(device)
    st_classifier=main_models.SoftTriple_Classifier(opt['la'], opt['gamma'], opt['tau'], opt['margin'], opt['dim'], opt['C'], opt['K']).to(device)
    #st_classifier=main_models.ResNet_Classifier(num_classes=opt['C'], require_flatten=True)

    #Reversal gradient method
    if opt['tf_module_loss']=='dann': domain_critic=main_models.Domain_Classifier(in_channels=512).to(device) 
    elif opt['tf_module_loss']=='mdd': domain_critic=main_models.Domain_Classifier(in_channels=512,
                                                                             domainclassifier_head='mdd',
                                                                             num_classes=opt['C']
                                                                             ).to(device)
    elif opt['tf_module_loss'] in {'mme', 'pcs'}:
        #print("Use cosine classifier...")
        cosine_classifier = main_models.CosineClassifier(num_class=opt['C'], inc=512, temp=0.05).to(device) 
        MB_X1 = MB_X2 = pcs.MemoryBank_create(size=2*opt['C']*opt['n_target_samples'],  #bs==C/n_iters==C*shot, i.e., 5-shot,
                                                      dim=512,
                                                      device=device)    #initialize memory bank state as ramdomized
    else: []

elif opt['model']=='resnet50':
    encoder=main_models.ResNet50_Encoder(pretrained=True, require_flatten=False, att_type=opt['att_type']).to(device)
    st_embedding=main_models.SoftTriple_Embedding(flatten=2048, dim=opt['dim']).to(device)
    st_classifier=main_models.SoftTriple_Classifier(opt['la'], opt['gamma'], opt['tau'], opt['margin'], opt['dim'], opt['C'], opt['K']).to(device)
    #st_classifier=main_models.ResNet_Classifier(num_classes=opt['C'], require_flatten=True)

    #Reversal gradient method
    if opt['tf_module_loss']=='dann': domain_critic=main_models.Domain_Classifier(in_channels=2048).to(device) 
    elif opt['tf_module_loss']=='mdd': domain_critic=main_models.Domain_Classifier(in_channels=2048,
                                                                             domainclassifier_head='mdd',
                                                                             num_classes=opt['C']
                                                                             ).to(device)
    elif opt['tf_module_loss'] in {'mme', 'pcs'}:
        cosine_classifier = main_models.CosineClassifier(num_class=opt['C'], inc=2048, temp=0.05).to(device) 
        MB_X1 = MB_X2 = pcs.MemoryBank_create(size=2*opt['C']*opt['n_target_samples'],     
                                                      dim=2048,
                                                      device=device)
    else: []

elif opt['model']=='alexnet':
    encoder=main_models.AlexNet_Encoder(pretrained=True, att_type=opt['att_type']).to(device)
    st_embedding=main_models.SoftTriple_Embedding(flatten=9216, dim=opt['dim']).to(device)
    st_classifier=main_models.SoftTriple_Classifier(opt['la'], opt['gamma'], opt['tau'], opt['margin'], opt['dim'], opt['C'], opt['K']).to(device)
    #st_classifier=main_models.ResNet_Classifier(num_classes=opt['C'], require_flatten=True)
    if opt['tf_module_loss']=='dann': domain_critic=main_models.Domain_Classifier(in_channels=9216).to(device) 
    elif opt['tf_module_loss']=='mdd': domain_critic=main_models.Domain_Classifier(in_channels=9216,
                                                                             domainclassifier_head='mdd',
                                                                             num_classes=opt['C']
                                                                             ).to(device)
    elif opt['tf_module_loss'] in {'mme', 'pcs'}:
        cosine_classifier = main_models.CosineClassifier(num_class=opt['C'], inc=9216, temp=0.05).to(device) 
        MB_X1 = MB_X2 = pcs.MemoryBank_create(size=2*opt['C']*opt['n_target_samples'],     #bs==C/n_iters==C*shot, i.e., 5-shot,
                                                      dim=9216,
                                                      device=device)
    else: []

# all loss
st_criterion = main_models.CE_softtriple_criterion(opt['tau'], opt['dim'], opt['C'], opt['K']).to(device)       # All hyper-parameters compute for weight-reg at vanilla ST baseline that include in this custom criterion from original paper (i.e., CE + weight-reg).
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
loss_fn_nonreduc = torch.nn.CrossEntropyLoss(reduction='none')
loss_fn_bcelogits = torch.nn.BCEWithLogitsLoss(reduction='mean')
loss_fn_bcelogits_nonreduc = torch.nn.BCEWithLogitsLoss(reduction='none')
loss_fn_bce = torch.nn.BCELoss(reduction='mean')
loss_fn_bce_nonreduc = torch.nn.BCELoss(reduction='none')

#-----------------------few shot exenplars------------------------
if opt['data_type']=='office31':
    #n-shot sampling from a fully source loader
    X_s,Y_s = dataloader.sample_data_office31(mode=opt['source'], image_size=image_size)
    #X_t,Y_t = dataloader.sample_data_office31(mode=opt['target'], image_size=image_size)
    #n-scarce-shot at a target loader
    #X_s,Y_s = dataloader.create_few_samples_office31(opt['n_target_samples'], mode=opt['source'], image_size=image_size)
    X_t,Y_t = dataloader.create_few_samples_office31(opt['n_target_samples'], mode=opt['target'], image_size=image_size)


#initial value
tf_inv_loss_loss_log = minibatch_loss_X1_log = source_eval_acc_log = target_eval_acc_log = epoch_log = []
accr_best = 0
util_weight = 0
alpha_weight = 0
lr=1e-3
#fft prep
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (3,1,1))
mean_img = torch.zeros(1, 1)

avgpool2d_7x7 = nn.AvgPool2d(7)  


#print("start training to adapt source-target domain in 3rd step only...")
print("*** starting of domain-specific adaptive training ***\n[number of target samples:{}]\n[Image-augmentation:{}+alpha:{}]\n[descriptor:{},k={}]\n[transfer_cost:{}+order(*for robustCORAL/spectralCORAL only):{}]\n[attention:{}]".format(
opt['n_target_samples'],
opt['mutation_style'] if opt['mutation']=='r' else 'direct inter-domain pure-image in adaptation task',
opt['alpha_mix'] if opt['mutation']=='r' else 'N/A',
"SoftTriple" if opt['K']>1 else "Vanilla classifier",
opt['K'] if opt['K']>1 else 1,
opt['tf_inv_loss'] if opt['tf_module_loss']=='n' else opt['tf_module_loss'],
opt['robust_order'],
opt['att_type']))


#----------------------training optimizer-----------------------------
#SGD got better result than Adam
optimizer_g_encoder=torch.optim.SGD(list(encoder.parameters()),\
                                         lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer_h_embedding=torch.optim.SGD(list(st_embedding.parameters()),\
                                           lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer_h_classifier=torch.optim.SGD(list(st_classifier.parameters()),\
                                            lr=lr, momentum=0.9, weight_decay=5e-4)
if opt['tf_module_loss'] in {'mdd', 'dann'}:
    optimizer_domain_classifier=torch.optim.SGD(list(domain_critic.parameters()),\
                                                    lr=lr, momentum=0.9, weight_decay=5e-4)
elif opt['tf_module_loss'] in {'mme', 'pcs'}:
    optimizer_cosine_classifier=torch.optim.SGD(list(cosine_classifier.parameters()),\
                                                    lr=lr, momentum=0.9, weight_decay=5e-4)


#---to eval and test dataloader---
if opt['source']=='amazon': source_dl=dataloader.amazon_dataloader(batch_size=opt['batch_size'], image_size=image_size)
elif opt['source']=='webcam': source_dl=dataloader.webcam_dataloader(batch_size=opt['batch_size'], image_size=image_size)
elif opt['source']=='dslr': source_dl=dataloader.dslr_dataloader(batch_size=opt['batch_size'], image_size=image_size)

if opt['target']=='amazon': target_dl=dataloader.amazon_dataloader(batch_size=opt['batch_size'], image_size=image_size)
elif opt['target']=='webcam': target_dl=dataloader.webcam_dataloader(batch_size=opt['batch_size'], image_size=image_size)
elif opt['target']=='dslr': target_dl=dataloader.dslr_dataloader(batch_size=opt['batch_size'], image_size=image_size)

epoch = 0
for epoch in tqdm.tqdm(range(opt['n_epoches'])):

    encoder.train()
    st_embedding.train()
    st_classifier.train()
    if opt['tf_module_loss'] in {'dann','mdd'}:
        domain_critic.train()
    elif opt['tf_module_loss'] in {'mme','pcs'}:
        cosine_classifier.train()

    if opt['adjust_lr']==True:
    #decaying lr 10 times every 40 epoch
        lr=1e-2; s1=5; s2=50
        utils.adjust_learning_rate(optimizer_g_encoder, epoch, lr_starting=lr, step1=s1, step2=s2)
        utils.adjust_learning_rate(optimizer_h_embedding, epoch, lr_starting=lr, step1=s1, step2=s2)
        utils.adjust_learning_rate(optimizer_h_classifier, epoch, lr_starting=lr, step1=s1, step2=s2)
        if opt['tf_module_loss'] in {'dann', 'mdd'}:
            utils.adjust_learning_rate(optimizer_domain_classifier, epoch, lr_starting=lr, step1=s1, step2=s2)
        #elif opt['tf_module_loss'] in {'mme', 'pcs'}:
            utils.adjust_learning_rate(optimizer_cosine_classifier, epoch, lr_starting=lr, step1=s1, step2=s2)
        print("new learning rate...{}".format(lr/10.)) if epoch == s1 else None; print("new learning rate...{}".format(lr/10.)) if epoch == s2 else None

    # few data sampling
    if opt['data_type']=='office31':
        groups, groups_y = dataloader.sample_groups_office31(X_s,Y_s,X_t,Y_t,seed=opt['n_epoches']+epoch)

    #for discrimination
    G1, G2, G3, G4 = groups
    Y1, Y2, Y3, Y4 = groups_y
    #for training src-trg model
    groups_2 = [G1, G3]
    groups_y_2 = [Y1, Y3]
    #iter_plain_training
    n_iters = 2 * len(G1) # default: G2 / e.g., at Office-31, n_iters = 310 <--- 2domain x 5*31
    index_list = torch.randperm(n_iters)  

    mini_batch_size_g_h = opt['mini_batch_size_g_h'] #data only contains G2 and G4 ,so decrease mini_batch
    X1 = []
    #X1_mu = []
    X2 = []
    ground_truths_y1 = []
    ground_truths_y2 = []
    avg_time = []
    offset = 0 #for memory bank storing

    index = 0
    for index in range(n_iters):
        
        #--------------------init trade-off & value------------------------
        # control value changing for each training iteration
        util_weight = (epoch + 1)/opt['n_epoches']
        # control value changing for each training iteration
        p = float(index + epoch * n_iters) / (opt['n_epoches'] * n_iters)
        alpha_weight = (2 / (1 + np.exp(-10 * p)))-1
        #-----------------------------------------------------------------
        # two groups sampling
        ground_truth=index_list[index]//len(G1)
        #print(ground_truth, index_list[index], len(G1) * ground_truth)
        x1, x2 = groups_2[ground_truth][index_list[index] - len(G1) * ground_truth]
        y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G1) * ground_truth]
        # direct once group sampling
        #x1, x2 = groups_2[0][index_list[index] - len(G1) * ground_truth]
        #y1, y2 = groups_y_2[0][index_list[index] - len(G1) * ground_truth]

        if opt['mutation']=='r':
        #//////////build a fourier transform that translate source images to the target style//////////
            x1_copy = x1.clone()
        
            if mean_img.shape[-1] < 2:
                C, H, W = x1.shape
                mean_img = IMG_MEAN.repeat(1,H,W)

            x1_np, x2_np, mean_img_np = x1.numpy(), x2.numpy(), mean_img.numpy()
            #-------------------------------------------------------------------#
            if opt['mutation_style']=='fft':
                # 1. source to target, target to target
                src_transformed, x1_transformed_RGB = fourier_transform.FDA_source_to_target_np( x1_np, x2_np, beta= 1e-3)
                trg_transformed = x2_np
                # 2. subtract mean
                src_transformed = src_transformed - mean_img_np                                 # src, src_lbl
                trg_transformed = trg_transformed - mean_img_np                                 # trg, trg_lbl
                #x1_in_x2, x2_in_x2 = [], []
                x1, x2 = torch.from_numpy(src_transformed).float(), torch.from_numpy(trg_transformed).float()
              
            elif opt['mutation_style']=='blend':
                #alpha_mix = 0
                src_transformed = transform_img.blending_images(x1_np, x2_np, alpha_mix)
                trg_transformed = src_transformed
                x1, x2 = torch.from_numpy(src_transformed).float(), torch.from_numpy(trg_transformed).float()

            elif opt['mutation_style']=='replace':
                src_transformed = x2_np
                trg_transformed = x2_np
                x1, x2 = torch.from_numpy(src_transformed).float(), torch.from_numpy(trg_transformed).float()

            elif opt['mutation_style']=='mixup':
                x1, ground_truths_y1 = mixup.mixup_domain(x1, x2, ground_truths_y1,
                                                          alpha = opt['alpha_mix'])     # opt['alpha_mix'] - (0.15 * alpha_weight)

            elif opt['mutation_style']=='mixvary':
                #mixup
                if opt['data_type']=='office31':
                    vary = opt['alpha_mix']-(0.15 * alpha_weight)

                x1, ground_truths_y1 = mixup.mixup_domain(x1, x2, ground_truths_y1,
                                                          alpha = vary)     # opt['alpha_mix'] - (0.15 * alpha_weight)

            elif opt['mutation_style']=='cutmix':
                x1, ground_truths_y1 = mixup.cutmix_domain(x1, x2, ground_truths_y1, alpha = opt['alpha_mix'])

            elif opt['mutation_style']=='cutmixup':
                x1, ground_truths_y1 = mixup.cutmix_domain(x1, x2, ground_truths_y1, alpha = opt['alpha_mix'])
                x1, ground_truths_y1 = mixup.mixup_domain(x1, x2, ground_truths_y1, alpha = opt['alpha_mix'])
            #-------------------------------------------------------------------#
        else:
            pass
        #//////////build a image mutation that translate source images to the target style//////////

        #---GRL---
        #s_domain_label = torch.zeros(mini_batch_size_g_h)
        #t_domain_label = torch.ones(mini_batch_size_g_h)
        s_domain_label = torch.zeros(mini_batch_size_g_h, 2)
        t_domain_label = torch.ones(mini_batch_size_g_h, 2)
        
        # y1=torch.LongTensor([y1.item()])
        # y2=torch.LongTensor([y2.item()])
        X1.append(x1)
        X2.append(x2)
        ground_truths_y1.append(y1)
        ground_truths_y2.append(y2)

        
        if (index+1)%mini_batch_size_g_h==0:

            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths_y1 = torch.LongTensor(ground_truths_y1)
            ground_truths_y2 = torch.LongTensor(ground_truths_y2)
            #---GRL---
            s_domain_label = s_domain_label.long()
            t_domain_label = t_domain_label.long()
 
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths_y1 = ground_truths_y1.to(device)
            ground_truths_y2 = ground_truths_y2.to(device)
            #---GRL---
            s_domain_label = s_domain_label.to(device)
            t_domain_label = t_domain_label.to(device)

            #optimizer_g_h.zero_grad()
            optimizer_g_encoder.zero_grad()            
            optimizer_h_embedding.zero_grad()       
            optimizer_h_classifier.zero_grad()
            if opt['tf_module_loss'] in {'dann','mdd'}:
                optimizer_domain_classifier.zero_grad()
            elif opt['tf_module_loss'] in {'mme','pcs'}:
                optimizer_cosine_classifier.zero_grad()

            #Feature Extraction
            encoder_X1 = encoder(X1)
            encoder_X2 = encoder(X2)

            #---self-flattening (for other purpose, not use in catecorization)---     
            f_avg_X1, f_avg_X2 = avgpool2d_7x7(encoder_X1), avgpool2d_7x7(encoder_X2)
            enc_X1_flattened = f_avg_X1.view(f_avg_X1.size(0), -1)
            enc_X2_flattened = f_avg_X2.view(f_avg_X2.size(0), -1)
            #X_cat = torch.cat([enc_X1_flattened,enc_X2_flattened], 1)
            #--------------------------------------------------------------------

            embed_X1 = st_embedding(encoder_X1)     #[bs, C*n_shot] ---> [31, 155], [12, 60]
            embed_X2 = st_embedding(encoder_X2)
            y_pred_X1 = st_classifier(embed_X1, ground_truths_y1, mode='triplet')       #[bs, C] ---> [31, 31], [12, 12]
            y_pred_X2 = st_classifier(embed_X2, None, mode='non-triplet')

            ###############################################################################################

            minibatch_loss_X1 = st_criterion(y_pred_X1, ground_truths_y1, reduction="mean")

            #---DA type for transferring knowledge to target domain---
            #Unsup'd DA (All protocol is running this state)
            if opt['da_type']=='UDA':
                minibatch_loss_X2 = 1e-20

            #Sup'd DA
            elif opt['da_type']=='SDA':
                minibatch_loss_X2 = st_criterion(y_pred_X2, ground_truths_y2, reduction="mean")

            #################################################################################################
            #---Module domain-adaptive learning---
            if opt['tf_module_loss']=='dann':
                #domain classification
                #domain_pred=discriminator(X_cat)
                s_domain_output = domain_critic(enc_X1_flattened, alpha_weight)
                t_domain_output = domain_critic(enc_X2_flattened, alpha_weight)
                #---GRL---
                s_domain_label, t_domain_label = s_domain_label.float(), t_domain_label.float()

                loss_s_domain = loss_fn_bcelogits(s_domain_output, s_domain_label)
                loss_t_domain = loss_fn_bcelogits(t_domain_output, t_domain_label)
                #loss_s_domain = loss_fn_bcelogits(s_domain_output, s_domain_label)
                #loss_t_domain = loss_fn_bcelogits(t_domain_output, t_domain_label)
                tf_module_loss = alpha_weight * (loss_s_domain + loss_t_domain)

            elif opt['tf_module_loss']=='mdd':
                #---one-hot transformation by prediction vector
                y_max_src = torch.argmax(y_pred_X1, 1)
                y_max_tar = torch.argmax(y_pred_X2, 1)
                y_max_src_onehot = onehotembedding(y_max_src, num_classes=y_max_src.data.shape[0]).to(self.device)
                y_max_tar_onehot = onehotembedding(y_max_tar, num_classes=y_max_tar.data.shape[0]).to(self.device)
                #---aux_class_classifier_head
                s_domain_output = self.aux_class_classifier(enc_X1_flattened, alpha_weight)
                t_domain_output = self.aux_class_classifier(enc_X2_flattened, alpha_weight)
                s_domain_pred, _ = torch.max(s_domain_output, 1)  # [n_classes]
                t_domain_pred, _ = torch.max(t_domain_output, 1)
                #---output criteria from two modules
                #print(s_domain_pred.shape, y_max_src_onehot.shape)  # [bs], [bs, n_cls]
                loss_s_domain = self.class_loss_func(y_max_src_onehot,
                                                     s_domain_pred.to(dtype=torch.long)
                                                     )
                log_t_domain_label = torch.log(torch.clamp(1. - F.softmax(y_max_tar_onehot, dim=1), max=1.))
                loss_t_domain = -F.nll_loss(log_t_domain_label,
                                            t_domain_pred.to(dtype=torch.long),
                                            reduction='mean',
                                            )
                src_weight = 2. # weights in set of {2, 3, 4} following in their MDD studying
                tf_module_loss = (src_weight * loss_s_domain) + loss_t_domain

            elif opt['tf_module_loss']=='mme':
                #tf_module_loss_X1 = mme.adentropy(cosine_classifier, enc_X1_flattened, lamda=0.1)
                tf_module_loss_X2 = mme.adentropy(cosine_classifier, enc_X2_flattened, lamda=0.1)
                tf_module_loss = tf_module_loss_X2
            else:
                tf_module_loss = 1e-20

            ###############################################################################################     
            #---Domain-invariant representation learning (Stat & feature metrics)

            #---the proposed Spectral-CORAL---
            #**************************************
            if opt['tf_inv_loss'] == 'spectralcoral':
                tf_inv_loss = 1e2 * coral.spectralCORAL(y_pred_X1, y_pred_X2, order=opt['robust_order'])
            #**************************************

            elif opt['tf_inv_loss'] == 'robustcoral':
                tf_inv_loss = 1e2 * coral.RobustCORALloss(y_pred_X1, y_pred_X2, order=opt['robust_order']) 

            elif opt['tf_inv_loss'] == 'coral':
                tf_inv_loss = 1e2 * coral.CORALloss(y_pred_X1, y_pred_X2)

            elif opt['tf_inv_loss'] == 'logcoral':
                tf_inv_loss = coral.LogCORALloss(y_pred_X1, y_pred_X2)  #the worst eff

            elif opt['tf_inv_loss'] == 'HoMM3':
                tf_inv_loss, _, _ = 1e2 * HoMM.HoMM3_loss(y_pred_X1, y_pred_X2)

            elif opt['tf_inv_loss'] == 'HoMM4':
                tf_inv_loss, _, _ = 1e2 * HoMM.HoMM4_loss(y_pred_X1, y_pred_X2)

            elif opt['tf_inv_loss'] == 'KHoMM3':
                _, HR_Xs, HR_Xt = HoMM.HoMM3_loss(y_pred_X1, y_pred_X2)
                tf_inv_loss = 1e2 * HoMM.KernelHoMM_loss(HR_Xs, HR_Xt, sigma=1e-5)

            elif opt['tf_inv_loss'] == 'KHoMM4':
                _, HR_Xs, HR_Xt = HoMM.HoMM4_loss(y_pred_X1, y_pred_X2)
                tf_inv_loss = 1e2 * HoMM.KernelHoMM_loss(HR_Xs, HR_Xt, sigma=1e-5)

            elif opt['tf_inv_loss'] == 'mmd':
                loss_f_mmd = mmd.mmdloss(F.softmax(enc_X1_flattened, dim=1), F.softmax(enc_X2_flattened, dim=1))
                tf_inv_loss = loss_f_mmd

            elif opt['tf_inv_loss'] == 'dan':
                loss_f_mmd = 0.5 * mmd.mmdloss(F.softmax(enc_X1_flattened, dim=1), F.softmax(enc_X2_flattened, dim=1))
                loss_e_mdd = 0.5 * mmd.mmdloss(F.softmax(embed_X1, dim=1), F.softmax(embed_X2, dim=1))
                loss_o_mmd = 0.5 * mmd.mmdloss(F.softmax(y_pred_X1, dim=1), F.softmax(y_pred_X2, dim=1))
                tf_inv_loss = loss_f_mmd + loss_e_mdd + loss_o_mmd

            elif opt['tf_inv_loss'] == 'mcc':
                mcc_loss = mcc.MinimumClassConfusionLoss(temperature=2.) # used fix temperature following prototype
                tf_inv_loss = mcc_loss(y_pred_X2)

            #--test--Pairwise Discrepancy Distribution 
            #elif opt['tf_inv_loss'] == 'scda':
            #    tf_inv_loss = loss_pdd = SCDA.get_PDD_loss(y_pred_X1, y_pred_X2,  ground_truths_y1, pslabel_X1, temp=1.0, threshold=0.8)

            elif opt['tf_inv_loss'] == 'n':
                tf_inv_loss = 1e-20

            ###############################################################################################
            #loss monitoring
            #print("[Epoch %d/%d] [loss_CE_X1: %f] [loss_src_domain: %f] [loss_trg_domain: %f] [tf_module_loss: %f] [tf_inv_loss: %f] [alpha: %f]"\
            #   % (epoch+1, opt['n_epoches'], minibatch_loss_X1.item(), loss_s_domain.item(), loss_t_domain.item(), tf_module_loss.item(), tf_inv_loss.item(), util_weight))\
            #   if (index+1)%(opt['C']-1) else None

            ###############################################################################################
            #composite cost function
            #VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
            loss_sum = minibatch_loss_X1 + minibatch_loss_X2
            loss_sum += tf_module_loss
            loss_sum += tf_inv_loss
            ###############################################################################################
            loss_sum.backward()

            optimizer_g_encoder.step()            
            optimizer_h_embedding.step()       
            optimizer_h_classifier.step()
            if opt['tf_module_loss'] in {'dann','mdd'}:
                optimizer_domain_classifier.step()
            elif opt['tf_module_loss'] in {'mme','pcs'}:
                optimizer_cosine_classifier.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []

            minibatch_loss_X1_avg = tf_inv_loss_loss_avg = 0.
            #print("raw loss pre-avg (CE/stat): ",minibatch_loss_X1, tf_inv_loss)
            minibatch_loss_X1_avg += minibatch_loss_X1
            tf_inv_loss_loss_avg += tf_inv_loss

    #---loss log---
    #minibatch_loss_X1_log = np.append(minibatch_loss_X1_log, minibatch_loss_X1_avg.cpu().detach().numpy() / (len(G1)+len(G3)))
    #if opt['tf_inv_loss']=='n':
    #    tf_inv_loss_loss_log = np.append(tf_inv_loss_loss_log, tf_inv_loss_loss_avg / (len(G1)+len(G3)))
    #else:
    #    tf_inv_loss_loss_log = np.append(tf_inv_loss_loss_log, tf_inv_loss_loss_avg.cpu().detach().numpy() / (len(G1)+len(G3)))
    #print("post avg & loss stored by CE/stat: ", tf_inv_loss_loss_avg, tf_inv_loss_loss_log)

    #---------------------------val (source)--------------------------
    acc = 0
    idx = 0
    step = 0
    #X_test = []
    #Y_test = []
    #ground_truth_test = []
    
    encoder.eval()
    st_embedding.eval()
    st_classifier.eval()
    
    for idx, (data,labels) in enumerate(source_dl):

        data = data.to(device)
        labels = labels.to(device)

        encoded_test_src = encoder(data)
        y_pred_src=st_classifier(st_embedding(encoded_test_src), None, mode='non-triplet')
        _, preds = torch.max(y_pred_src, 1)
        acc += (preds == labels).float().mean().item()

    accuracy_s = round(acc / float(len(source_dl)), 4)
        
    #print("3rd step (eval) source [%s]----Epoch %d/%d  accuracy: %.4f "\
    #    % (opt['source'], epoch + 1, opt['n_epoches'], accuracy_s))

    #---log source eval (for plot & cm)---
    #source_eval_acc_log = np.append(source_eval_acc_log, accuracy_s*100)
    

    #--------------------------val (target)--------------------------
    acc = 0
    idx = 0
    step = 0
    stored_lbs = stored_preds = torch.empty(0, dtype=torch.float64).to(device)
    cm_target = []
    for idx, (data,labels) in enumerate(target_dl):
        
        data = data.to(device)
        labels = labels.to(device)

        encoded_test_trg = encoder(data)
        y_pred_trg=st_classifier(st_embedding(encoded_test_trg), None, mode='non-triplet')
        _, preds = torch.max(y_pred_trg, 1)
        acc += (preds == labels).float().mean().item()

        #---for confusion matrix---   
        #stored_lbs = torch.cat((stored_lbs, labels), 0)
        #stored_preds = torch.cat((stored_preds, preds), 0)
                
    accuracy_t = round(acc / float(len(target_dl)), 4)
   
    #print("3rd step (eval) target [%s]----Epoch %d/%d  accuracy: %.4f "\
    #    % (opt['target'], epoch + 1, opt['n_epoches'], accuracy_t))

    #---log target eval (for plot & cm)---
    #target_eval_acc_log = np.append(target_eval_acc_log, accuracy_t*100)

    #---demo displaying new better model (higher accuracy)---
    if accuracy_t > accr_best:

        print("adapting (valid) source [%s]----Epoch %d/%d  accuracy: %.4f"\
        % (opt['source'], epoch + 1, opt['n_epoches'], accuracy_s))
        print("adapting (valid) target [%s]----Epoch %d/%d  accuracy: %.4f"\
        % (opt['target'], epoch + 1, opt['n_epoches'], accuracy_t))
        print('===*** Better target %acc (valid-test) adapting from {} to {} ... saving model ***==='\
            .format(opt['source'], opt['target']))
        
        #---save checkpoint---
        '''
        n_way_k_shot = 'varyway5shot'
        baseline = "SoftTriple" if opt['K']>1 else "Vanilla classifier"
        domain_crit = opt['tf_inv_loss'] if opt['tf_module_loss']=='n' else opt['tf_module_loss']
        BasicModule.save(encoder,\
            path='output/checkpoint/best_encoder_{}2{}_{}_{}_{}({}).pth'\
            .format(opt['source'],\
            opt['target'],\
            baseline,\
            opt['att_type'],\
            domain_crit,\
            n_way_k_shot))
        BasicModule.save(st_embedding,\
            path='output/checkpoint/best_embedding_{}2{}_{}_{}_{}({}).pth'\
            .format(opt['source'],\
            opt['target'],\
            baseline,\
            opt['att_type'],\
            domain_crit,\
            n_way_k_shot))
        BasicModule.save(st_classifier,\
            path='output/checkpoint/best_classifier_{}2{}_{}_{}_{}({}).pth'\
            .format(opt['source'],\
            opt['target'],\
            baseline,\
            opt['att_type'],\
            domain_crit,\
            n_way_k_shot))
        '''

        #To limit announcement
        accr_best = accuracy_t
        accuracy_s = accuracy_t = 0 #reset
      
    encoder.train()
    st_embedding.train()
    st_classifier.train()

torch.cuda.empty_cache()

#---end of training---
print("*** the end of domain-specific adaptive training ***\n[number of target samples:{}]\n[Image-augmentation:{}+alpha:{}]\n[descriptor:{},k={}]\n[transfer_cost:{}+order(*for robustCORAL/spectralCORAL only):{}]\n[attention:{}]".format(
    opt['n_target_samples'],
    opt['mutation_style'] if opt['mutation']=='r' else 'direct inter-domain pure-image in adaptation task',
    opt['alpha_mix'] if opt['mutation']=='r' else 'N/A',
    "SoftTriple" if opt['K']>1 else "Vanilla classifier",
    opt['K'] if opt['K']>1 else 1,
    opt['tf_inv_loss'] if opt['tf_module_loss']=='n' else opt['tf_module_loss'],
    opt['robust_order'],
    opt['att_type']))
    
