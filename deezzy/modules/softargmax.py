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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        *_, n = x.shape
        softmax = torch.nn.functional.softmax((self.beta*x), dim=self.dim)
        indices = torch.linspace(0,1,n) #.to(self.device)
        outputs = torch.sum((n-1)*softmax*indices, dim=self.dim, keepdim=self.keepdim)
        return outputs