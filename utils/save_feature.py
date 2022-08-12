#https://github.com/indy-lab/ProtoTransfer/blob/master/omni-mini/prototransfer/save_features.py

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as f
from main_models import *
import numpy as np

def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)

        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


    #----
    #if __name__ == '__main__':
    #    train_classes = ["n02687172", "n04251144", "n02823428", "n03676483", "n03400231"]
    #    test_classes = ["n03272010", "n07613480", "n03775546", "n03127925", "n04146614"]
    #    trainset = LabelledDataset('miniimagenet', configs.data_path,
    #                               'train', train_classes)
    #    testset = LabelledDataset('miniimagenet', configs.data_path,
    #                              'test', test_classes)
    #    trainloader = DataLoader(trainset, shuffle=False, batch_size=100)
    #    testloader = DataLoader(testset, shuffle=False, batch_size=100)

    #    # Load checkpoint
    #    model = CNN_4Layer(in_channels=3)
    #    load_path = 'prototransfer/checkpoints/protoclr/proto_miniimagenet_conv4_euclidean_1supp_3query_50bs_best.pth.tar'
    #    checkpoint = torch.load(load_path)
    #    model.load_state_dict(checkpoint['model'])
    #    start_epoch = checkpoint['epoch']
    #    print("Loaded checkpoint '{}' (epoch {})"
    #          .format(load_path, start_epoch))

    #    model.cuda()
    #    model.eval()
    #    print('----------------- Save train features ------------------------')
    #    save_features(model, trainloader, 'plots/featuresProtoCLR_mini-ImageNet_train.hdf5')
    #    print('----------------- Save test features ------------------------')
    #    save_features(model, testloader, 'plots/featuresProtoCLR_mini-ImageNet_test.hdf5')