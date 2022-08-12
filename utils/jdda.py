import torch

def InstanceDiscriminativeLoss(f_src, batch_size):
    """
        Args:
        f_src (int): source_model.fc4 output feature
        feat_dim (int): feature dimension.
    """
    norm = lambda x: torch.sum(torch.square(x), 1)
    F0 = torch.transpose(norm(torch.unsqueeze(f_src, dim=2) - torch.transpose(f_src)))  #calculate pair-wise distance of Xs
    margin0 = 0
    margin1 = 100
    F0=torch.pow(torch.maximum(0.0, F0-margin0), 2)
    F1=torch.pow(torch.maximum(0.0, margin1-F0), 2)
    intra_loss=torch.mean(torch.matmul(F0, W))
    inter_loss=torch.mean(torch.matmul(F1, 1.0-W))
    discriminative_loss = (intra_loss+inter_loss) / (batch_size * batch_size)

    return discriminative_loss

#self.CalDiscriminativeLoss(method="InstanceBased")
#self.CalDomainLoss(method="CORAL")