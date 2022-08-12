import torch

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + 1e-6)
        return loss

# solved nan
#eps = 1e-6
#loss = torch.sqrt(criterion(x, y) + eps)
