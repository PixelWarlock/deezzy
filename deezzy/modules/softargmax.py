import torch

class SoftArgMax(torch.nn.Module):

    def __init__(self, beta:float=1000., dim:int=-1, keepdim=False) -> None:
        """
        SoftArgMax is a function that return the approximation of an indice of a maximum value in a tensor
        Args:
            beta (float, optional): The higher the beta the more precise is the estimation. Defaults to 1000.
        """
        super(SoftArgMax, self).__init__()
        self.beta = beta
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        *_, n = x.shape
        softmax = torch.nn.functional.softmax((self.beta*x), dim=self.dim)
        indices = torch.linspace(0,1,n)
        outputs = torch.sum((n-1)*softmax*indices, dim=self.dim, keepdim=self.keepdim, dtype=torch.float)
        return outputs