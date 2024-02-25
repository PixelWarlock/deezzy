import torch

class Gaussian(torch.nn.Module):
    def __init__(self, univariate:bool):
        super(Gaussian, self).__init__()
        self.univariate=univariate

    def forward(self, x:torch.Tensor, representation:torch.Tensor):
        if self.univariate:
            batch_size, num_features, granularity, _ = representation.size()
            mean = representation.view(batch_size, num_features*granularity,2)[...,0].view(batch_size,num_features,granularity)
            var = representation.view(batch_size, num_features*granularity,2)[...,1].view(batch_size,num_features,granularity)
            z = (x.unsqueeze(dim=-1) - mean)/var
            return torch.exp(-0.5 * torch.pow(z,2))
        else:
            pass
        