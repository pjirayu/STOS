import time
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils import data
from random import choice

#------------------------------ hard-written dataloader (MNIST-SVHN) ------------------------------
#return MNIST dataloader

def mnist_dataloader(batch_size=256,train=True):

    dataloader=DataLoader(
    datasets.MNIST('data/handwritten/mnist',train=train,download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])),
    batch_size=batch_size,shuffle=True)

    return dataloader

def svhn_dataloader(batch_size=4,train=True):
    dataloader = DataLoader(
        datasets.SVHN('data/handwritten/svhn', split=('train' if train else 'test'), download=True,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
        batch_size=batch_size, shuffle=False)

    return dataloader

def sample_data_handwritten():
    dataset=datasets.MNIST('D:/D10907801_PJ/2_research/Prototype-FADACIL-Pytorch_R0/data/handwritten/',train=True,download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
    n=len(dataset)

    X=torch.Tensor(n,1,28,28)
    Y=torch.LongTensor(n)

    inds=torch.randperm(len(dataset))
    for i,index in enumerate(inds):
        x,y=dataset[index]
        X[i]=x
        Y[i]=y
    return X,Y

def create_target_samples_handwritten(n=1):
    dataset=datasets.SVHN('D:/D10907801_PJ/2_research/Prototype-FADACIL-Pytorch_R0/data/handwritten/svhn', split='train', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]))
    X,Y=[],[]
    classes=10*[n]

    i=0
    while True:
        if len(X)==n*10:
            break
        x,y=dataset[i]
        if classes[y]>0:
            X.append(x)
            Y.append(y)
            classes[y]-=1
        i+=1

    assert (len(X)==n*10)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))

#------------------------------------------------------------------------------------------------


# only available on Office-31 dataset
#-------------------------------- objects dataloader (OFFICE31) ---------------------------------

def amazon_dataloader(batch_size=16, image_size=224):
    dataloader = DataLoader(
            dataset=datasets.ImageFolder('data/office31/amazon/images',
                        transform=transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.79235075407833078, 0.78620633471295642, 0.78417965306916637], [0.27691643643313618, 0.28152348841965347, 0.28287296762830788])   #amazon
                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])       #ImageNet
                            #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])      #for gemeral normalized color image
                        ])),
        batch_size=batch_size, shuffle=True)

    return dataloader

def webcam_dataloader(batch_size=16, image_size=224):
    dataloader = DataLoader(
            dataset=datasets.ImageFolder('data/office31/webcam/images',
                        transform=transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(), 
                            transforms.Normalize([0.61197983011509638, 0.61876474000372972, 0.61729662103473015], [0.22763857108616978, 0.23339382150450594, 0.23722725519031848])       #webcam
                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])       #ImageNet
                            #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])      #for general normalized color image
                        ])),
        batch_size=batch_size, shuffle=True)

    return dataloader

def dslr_dataloader(batch_size=16, image_size=224):
    dataloader = DataLoader(
            dataset=datasets.ImageFolder('data/office31/dslr/images',
                        transform=transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      #ImageNet
                            #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])      #for general normalized color image
                        ])),
        batch_size=batch_size, shuffle=True)

    return dataloader

def synthetic_dataloader(batch_size=16, image_size=224):
    dataloader = DataLoader(
            dataset=datasets.ImageFolder('data/office31/synthetic/images',
                        transform=transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      #ImageNet
                            #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])      #for general normalized color image
                        ])),
        batch_size=batch_size, shuffle=True)

    return dataloader

# --- For 2nd step ---
def sample_data_office31(mode="amazon", image_size=224):

    if mode == "amazon":
        path = 'data/office31/amazon/images'
        mean = [0.79235075407833078, 0.78620633471295642, 0.78417965306916637]
        std = [0.27691643643313618, 0.28152348841965347, 0.28287296762830788]
    elif mode == "dslr":
        path = 'data/office31/dslr/images'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif mode == "webcam":
        path = 'data/office31/webcam/images'
        mean = [0.61197983011509638, 0.61876474000372972, 0.61729662103473015]
        std = [0.22763857108616978, 0.23339382150450594, 0.23722725519031848]
    elif mode == 'synthetic':
        path = 'data/office31/synthetic/images'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        print("selecting dataset either amazon, dslr, webcam or synthetic (newly added in modern office-31")

    dataset=datasets.ImageFolder(path,
                    transform=transforms.Compose([
                        transforms.Resize((image_size,image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ]))
    n=len(dataset)

    X=torch.Tensor(n,3,image_size,image_size)
    Y=torch.LongTensor(n)

    inds=torch.randperm(len(dataset))
    for i,index in enumerate(inds):
        x,y=dataset[index]
        X[i]=x
        Y[i]=y

    return X,Y


def create_few_samples_office31(n=1, mode="webcam", image_size=224):             #n = shot

    if mode == "amazon":
        path = 'data/office31/amazon/images'
        mean = [0.79235075407833078, 0.78620633471295642, 0.78417965306916637]
        std = [0.27691643643313618, 0.28152348841965347, 0.28287296762830788]
    elif mode == "dslr":
        path = 'data/office31/dslr/images'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif mode == "webcam":
        path = 'data/office31/webcam/images'
        mean = [0.61197983011509638, 0.61876474000372972, 0.61729662103473015]
        std = [0.22763857108616978, 0.23339382150450594, 0.23722725519031848]
    elif mode == 'synthetic':
        path = 'data/office31/synthetic/images'
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        print("selecting dataset either amazon, dslr, webcam or synthetic (newly added in modern office-31")

    dataset=datasets.ImageFolder(path,
                transform=transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]))
    X,Y=[],[]
    classes=31*[n]

    i=0
    while True:
        if len(X)==n*31:
            break
        x,y=dataset[i]
        if classes[y]>0:
            X.append(x)
            Y.append(y)
            classes[y]-=1
        i+=1

    assert (len(X)==n*31)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))   

#---------------------

# --- For few-shot transfer step ---
"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes

G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""
def create_groups_office31(X_s,Y_s,X_t,Y_t,seed=1):
    #change seed so every time we get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    n=X_t.shape[0] #31*shot
    #print("n for target samples: ",range(n))

    #shuffle order
    classes = torch.unique(Y_s)
    classes=classes[torch.randperm(len(classes))]

    class_num=classes.shape[0]  #31
    shot=(n//class_num)
    #print("n/class_num/n shot for target samples: ",n,class_num,shot)

    def s_idxs(c):
        idx=torch.nonzero(Y_s.eq(int(c)))
        return idx[torch.randperm(len(idx))][:shot].squeeze()
    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix=torch.stack(source_idxs)
    target_matrix=torch.stack(target_idxs)

    G1, G2, G3, G4 = [], [], [], []
    Y1, Y2, Y3, Y4 = [], [], [], []

    #i = random.randint(1,31)
    # varying-way k-shot
    for i in range(31):     #default: range=31
        for j in range(shot):

            #G1: a pair of pic comes from same domain ,same class
            G1.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
            Y1.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))

            #G2: a pair of pic comes from different domain,same class
            G2.append((X_s[source_matrix[i][j]],X_s[source_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]],Y_s[source_matrix[i][j]]))

            #G3: a pair of pic comes from same domain, different classes
            G3.append((X_s[source_matrix[(i+1) % 31][j]],X_t[target_matrix[i % 31][j]]))
            Y3.append((Y_s[source_matrix[(i+1) % 31][j]],Y_t[target_matrix[i % 31][j]]))

            #G4: a pair of pic comes from different domain, different classes
            G4.append((X_s[source_matrix[i % 31][j]],X_t[target_matrix[i % 31][j]]))
            Y4.append((Y_s[source_matrix[i % 31][j]],Y_t[target_matrix[i % 31][j]]))


    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]

    #print("len: ",len(groups), len(groups_y))

    g=0
    #make sure we sampled enough samples
    for g in groups:
        assert(len(g)==n)
    return groups,groups_y


def sample_groups_office31(X_s,Y_s,X_t,Y_t,seed=1):

    #print("Sampling groups")
    return create_groups_office31(X_s,Y_s,X_t,Y_t,seed=seed)
#------------------------------------------------------------------------------------------------
