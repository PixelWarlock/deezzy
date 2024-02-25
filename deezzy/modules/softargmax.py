import torch

class SoftArgMax(torch.nn.Module):

    def __init__(self, beta:float=100.) -> None:
        """
        SoftArgMax is a function that return the approximation of an indice of a maximum value in a tensor
        Args:
            beta (float, optional): The higher the beta the more precise is the estimation. Defaults to 1000.
        """
        super(SoftArgMax, self).__init__()
        self.beta = beta

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        *_, n = x.shape
        softmax = torch.nn.functional.softmax((self.beta*x), dim=-1)
        indices = torch.linspace(0,1,n)
        outputs = torch.sum((n-1)*softmax*indices, dim=-1, keepdim=False, dtype=torch.float)
        return outputs