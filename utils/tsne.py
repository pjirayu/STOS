# ref https://github.com/zhjscut/Bridging_UDA_SSL/blob/e0be6742f1203bb983261e3e1e57d34e1e03299d/common/utils/analysis/__init__.py#L7
import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=20)  #default: s=2
    plt.savefig(filename)

'''
# ref https://github.com/zhjscut/Bridging_UDA_SSL/blob/e0be6742f1203bb983261e3e1e57d34e1e03299d/examples/domain_adaptation/classification/mdd.py#L19
# analysis the model
if args.phase == 'analysis':
    # extract features from both domains
    feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
    source_feature = collect_feature(train_source_loader, feature_extractor, device)
    target_feature = collect_feature(train_target_loader, feature_extractor, device)
    # plot t-SNE
    tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
    tsne.visualize(source_feature, target_feature, tSNE_filename)
    print("Saving t-SNE to", tSNE_filename)
    # calculate A-distance, which is a measure for distribution discrepancy
    A_distance = a_distance.calculate(source_feature, target_feature, device)
    print("A-distance =", A_distance)
    return
'''