import torch

class Gaussian(torch.nn.Module):
    def __init__(self,
                 in_features:int,
                 multivariate:bool,
                 granularity:int=None,
                 num_classes:int=None):
        super(Gaussian, self).__init__()

        if multivariate is True and granularity is not None:
            print("WARNING! You've specified granularity parameter but it does not play a part when using multivariate option.")
        if multivariate is True and num_classes is None:
            raise ValueError("You must specify number of classes when using multivariate.")
        
        low=0
        high=0
        if multivariate:
            ...
        else:
            self.mean = torch.nn.Parameter(torch.FloatTensor(in_features, granularity).uniform_(low,high),requires_grad=True)
            self.variance = torch.nn.Parameter(torch.FloatTensor(in_features, granularity).uniform_(low,high),requires_grad=True)

        self.multivariate = multivariate
        
    def forward(self, x:torch.Tensor, representation:torch.Tensor):
        covariance_matrix = torch.eye(self.mean.size(0))
        inverse_covariance = torch.inverse(covariance_matrix)
        exponent_term = -0.5 * torch.sum((x - self.mean).mm(inverse_covariance) * (x - self.mean), dim=1)
        return torch.exp(exponent_term)
        