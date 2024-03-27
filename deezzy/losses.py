import torch
from deezzy.modules.softargmax import SoftArgMax

class AscendingMeanLoss(torch.nn.Module):
    def __init__(self):
        super(AscendingMeanLoss, self).__init__()

    def get_chunks(self, x):
        _,g,_ = x.shape
        n = g - 1 
        mean = x[...,0]
        chunks = list()
        for i in range(n):
            chunks.append(mean[:, i:i+2].T)
        return chunks

    
    def compute_loss(self, chunk:torch.Tensor):
        div = chunk[:-1]/chunk[1:]
        log = torch.log(div)
        loss = torch.nn.functional.relu(log).sum()
        return loss

    def forward(self, x):
        _,g,_ = x.shape
        if g<=1:
            raise ValueError("Doesn't make sense")
        elif g==2:
            mean = x[...,0].T
            return self.compute_loss(chunk=mean)
        else:
            chunks = self.get_chunks(x=x)
            losses=torch.zeros(len(chunks))
            for i,chunk in enumerate(chunks):
                losses[i] = self.compute_loss(chunk=chunk)
            return losses.sum()
        
    
class SquashingVarianceLoss(torch.nn.Module):
    def __init__(self):
        super(SquashingVarianceLoss, self).__init__()

    def forward(self, x):
        num_features, granularity, num_params = x.size()
        var = x.view(num_features*granularity,num_params)[...,1].view(num_features,granularity)
        factor = num_features*granularity*num_params
        return var.sum()/factor

