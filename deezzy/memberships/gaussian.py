import torch

class Gaussian(torch.nn.Module):
    def __init__(self, univariate:bool):
        super(Gaussian, self).__init__()
        self.univariate=univariate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_inverse_cov_matrix(self, variance:torch.Tensor):
        b,c,g,f = variance.size()
        # view(...).to(self.device) 
        covariance_matrix = torch.pow((torch.cat([torch.eye(f) for _ in range(b*c*g)]).view(b,c,g,f,f) * variance.unsqueeze(dim=-1)),2)
        return torch.inverse(covariance_matrix)

    def forward(self, x:torch.Tensor, representation:torch.Tensor):
        if self.univariate:
            batch_size, num_features, granularity, _ = representation.size()
            mean = representation.view(batch_size, num_features*granularity,2)[...,0].view(batch_size,num_features,granularity)
            var = representation.view(batch_size, num_features*granularity,2)[...,1].view(batch_size,num_features,granularity)
            z = (x.unsqueeze(dim=-1) - mean)/var
            return torch.exp(-0.5 * torch.pow(z,2))
        else:
            batch_size, num_classes,num_gaussians, num_features, num_parameters = representation.size()
            mean = representation.view(batch_size, num_classes*num_gaussians*num_features, num_parameters)[...,0].view(batch_size, num_classes, num_gaussians,num_features)
            var = representation.view(batch_size, num_classes*num_gaussians*num_features, num_parameters)[...,1].view(batch_size, num_classes, num_gaussians,num_features)

            inverse_matrix = self._compute_inverse_cov_matrix(var)
            diff = x.view(batch_size, 1, 1, num_features) - mean
            mm1 = diff.view(batch_size, num_classes, num_gaussians, 1, num_features) @ inverse_matrix
            mm2 = mm1 @ diff.view(batch_size, num_classes, num_gaussians, num_features,1)
            z = -.5 * mm2.view(batch_size, num_classes, num_gaussians)
            return torch.exp(z)
        