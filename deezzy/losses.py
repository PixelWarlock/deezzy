import torch
from deezzy.modules.softargmax import SoftArgMax

class AscendingMeanLoss(torch.nn.Module):
    def __init__(self):
        super(AscendingMeanLoss, self).__init__()
        self.softargmax = SoftArgMax(beta=10000., keepdim=False)
        self.bceloss = torch.nn.BCELoss()

    def forward(self, x):
        f,g,_ = x.size()
        mean = x[...,0]
        logits_indicies = self.softargmax(mean)
        targets_indicies = torch.ones(f) * (g-1)
        return self.bceloss(logits_indicies, targets_indicies)