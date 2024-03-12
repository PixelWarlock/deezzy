import math
import torch

class LinearRelu(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super(LinearRelu, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.ReLU()
        )
    
    def forward(self,x):
        return self.module(x)

class LinearReluDropout(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, drop_rate:float=0.3):
        super(LinearReluDropout, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop_rate)
        )
    
    def forward(self,x):
        return self.module(x)
    
class LinearSigmoid(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super(LinearSigmoid, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.Sigmoid()
        )
    
    def forward(self,x):
        return self.module(x)
    
class LinearFeatureHead(torch.nn.Module):
    def __init__(self,
                 hidden_size:int,
                 in_features:int,
                 granularity:int):
        super(LinearFeatureHead, self).__init__()

        self.linear_layers = torch.nn.ModuleList()
        head_shape = math.prod((in_features, granularity,2))
        
        factor = 1./granularity
        for _ in range(in_features):
            for _ in range(granularity):
                for p in range(2):
                    linear_layer = torch.nn.Linear(in_features=hidden_size, out_features=1) #LinearSigmoid(in_features=hidden_size, out_features=1)
                    if p == 0:
                        linear_layer.weight.data.fill_(factor)
                        factor +=factor
                    sigmoid_layer = torch.nn.Sigmoid()
                    self.linear_layers.append(
                        torch.nn.Sequential(
                            linear_layer,
                            sigmoid_layer
                        )
                    )
        """
        for _ in range(head_shape):
            self.linear_layers.append(LinearSigmoid(in_features=hidden_size, out_features=1))
        """
        self.in_features = in_features
        self.granularity = granularity

    def forward(self, x):
        batch_size = x.size(0)
        outputs = []
        for layer in self.linear_layers:
            z = layer(x)
            outputs.append(z.unsqueeze(dim=1))
        return torch.cat(outputs, dim=1).view(batch_size, self.in_features, self.granularity, 2)
    
class LinearClassHead(torch.nn.Module):
    def __init__(self,
                 hidden_size:int,
                 in_features:int,
                 num_classes:int,
                 num_gaussians:int):
        super(LinearClassHead, self).__init__()

        self.linear_layers = torch.nn.ModuleList()
        head_shape = math.prod((num_classes, num_gaussians, in_features, 2))
        for _ in range(head_shape):
            self.linear_layers.append(LinearSigmoid(in_features=hidden_size, out_features=1))

        self.in_features = in_features
        self.num_gaussians = num_gaussians
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.size(0)
        outputs = []
        for layer in self.linear_layers:
            z = layer(x)
            outputs.append(z.unsqueeze(dim=1))
        return torch.cat(outputs, dim=1).view(batch_size, self.num_classes, self.num_gaussians, self.in_features, 2)

