import torch

class LinearReluDroput(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, drop_rate:float=0.3):
        super(LinearReluDroput, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate)
        )
    
    def forward(self,x):
        return self.module(x)

