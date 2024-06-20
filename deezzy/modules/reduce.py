import torch

class ReduceAndCopy(torch.nn.Module):
    def __init__(self):
        super(ReduceAndCopy, self).__init__()
    
    def forward(self,x):
        b = x.size()[0]
        reduced = torch.mean(x, dim=0).unsqueeze(dim=0)
        return torch.cat([reduced for _ in range(b)], dim=0)
