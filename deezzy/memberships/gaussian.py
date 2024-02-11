import torch

class Gaussian(torch.nn.Module):
    def __init__(self, multivariate:bool):
        super(Gaussian, self).__init__()
        self.multivariate=multivariate

    def forward(self, x:torch.Tensor, representation:torch.Tensor):
        covariance_matrix = torch.eye(self.mean.size(0))
        inverse_covariance = torch.inverse(covariance_matrix)
        exponent_term = -0.5 * torch.sum((x - self.mean).mm(inverse_covariance) * (x - self.mean), dim=1)
        return torch.exp(exponent_term)
        