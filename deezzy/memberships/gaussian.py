import torch

class Gaussian(torch.nn.Module):
    def __init__(self, univariate:bool):
        super(Gaussian, self).__init__()
        self.univariate=univariate

    def forward(self, x:torch.Tensor, representation:torch.Tensor):
        batch_size, num_features = x.size()
        if self.univariate:
            # Extract mean and variance for univariate Gaussian
            mean = representation[..., 0]  # Extract mean from the last dimension
            var = representation[..., 1]   # Extract variance from the last dimension

            # Expand dimensions to match batch_vectors
            mean = mean.unsqueeze(-1).expand(batch_size, num_features, representation.size(1))
            var = var.unsqueeze(-1).expand(batch_size, num_features, representation.size(1))

            # Calculate univariate Gaussian
            prob = torch.exp(-0.5 * ((x.unsqueeze(1) - mean) ** 2) / var) / torch.sqrt(2 * torch.pi * var)
            prob = prob.prod(dim=2)  # Multiply along the Gaussian components axis

        else:
            # Extract mean and covariance for multivariate Gaussian
            mean = representation[..., 0]  # Extract mean from the last dimension
            cov = representation[..., 1]   # Extract covariance from the last dimension

            # Calculate multivariate Gaussian
            diff = x.unsqueeze(1) - mean.unsqueeze(2)
            inv_cov = torch.inverse(cov)
            exponent = -0.5 * torch.einsum('bif,bfij,bfj->bfi', diff, inv_cov, diff)
            prob = torch.exp(exponent) / torch.sqrt(torch.det(2 * torch.pi * cov))

        return prob
        