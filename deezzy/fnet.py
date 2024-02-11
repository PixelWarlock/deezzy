import torch
from deezzy.modules.linear import LinearFeatureHead, LinearClassHead

class Fnet(torch.nn.Module):
    def __init__(self, 
                 backbone:torch.nn.Module,
                 in_features:int,
                 granularity:int,
                 num_gaussians:int,
                 num_classes:int
                 ):
        super(Fnet, self).__init__()

        hidden_size = backbone(torch.ones(in_features)).size(dim=-1) # get the size of an output of the backbone
        self.backbone = backbone
        self.feature_head = LinearFeatureHead(hidden_size=hidden_size,
                                              in_features=in_features,
                                              granularity=granularity)
        
        self.class_head = LinearClassHead(hidden_size=hidden_size,
                                          in_features=in_features,
                                          num_classes=num_classes,
                                          num_gaussians=num_gaussians)


    def forward(self, x):
        z = self.backbone(x)
        
        # compute representation with which we will fuziffy features
        fgp = self.feature_head(z)

        # compute representation with which we will turn adjectives of features into class probabilities
        cmfp = self.class_head(z)

