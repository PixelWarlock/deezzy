import torch
import numpy as np

def loop_univariate_gaussian(x,fgp):
    def univariate_gauss(value, mean, var):
        z =  (value - mean)/var
        return np.exp(-0.5*(z**2))
    
    batch_size, num_features, granularity, parametes = fgp.shape
    outputs = np.zeros((batch_size, num_features, granularity))

    for sample_iterator in range(batch_size):
        sample = x[sample_iterator]
        for feature_iterator in range(num_features):
            feature = sample[feature_iterator]
            for granularity_iterator in range(granularity):
                mean = fgp[sample_iterator, feature_iterator, granularity_iterator, 0]
                var = fgp[sample_iterator, feature_iterator, granularity_iterator, 1]
                gauss = univariate_gauss(value=feature, mean=mean, var=var)
                outputs[sample_iterator, feature_iterator, granularity_iterator] = gauss

    return outputs



def tensor_univariate_gaussian(x, fgp):
    x = torch.tensor(x)
    fgp = torch.tensor(fgp)
    batch_size, num_features, granularity, parameters = fgp.size()

    # reshaping 
    mean = fgp.view(batch_size, num_features*granularity,2)[...,0].view(batch_size,num_features,granularity)
    var = fgp.view(batch_size, num_features*granularity,2)[...,1].view(batch_size,num_features,granularity)
    z = (x.unsqueeze(dim=-1) - mean)/var
    return torch.exp(-0.5 * torch.pow(z,2))

def loop_multivariate_gaussian():
    pass

def tensor_mutlivariate_gaussian():
    pass

if __name__ == "__main__":
    batch_size=1
    num_features = 5
    granularity = 20
    num_classes = 2
    num_gaussians = 2
    

    x = np.random.random((batch_size, num_features))
    fgp = np.random.random((batch_size, num_features, granularity, 2))

    univariate_loop = loop_univariate_gaussian(x, fgp).astype(np.float32)
    univariate_tensor = np.array(tensor_univariate_gaussian(x, fgp)).astype(np.float32)

    if (univariate_loop==univariate_tensor).all():
        print("Univariate test passed")
    else:
        print("Univariate test not passed!")