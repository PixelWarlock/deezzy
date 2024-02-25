import torch
import numpy as np
from math import prod

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


def loop_multivariate_gaussian(x, cmfp):
    def compute_covariance_matrix(variance_vector):
        n = len(variance_vector)
        return np.eye(n) * np.power(variance_vector, 2)
    
    batch_size, num_classes,num_gaussians, _,_ = cmfp.shape
    outputs = np.zeros((batch_size, num_classes,num_gaussians))

    for sample_iterator in range(batch_size):
        for class_iterator in range(num_classes):
            for gaussian_iterator in range(num_gaussians):

                    mean = cmfp[sample_iterator, class_iterator, gaussian_iterator, :, 0]
                    var = cmfp[sample_iterator, class_iterator, gaussian_iterator, :, 1]
                    #print(f"Mean | Sample: {sample_iterator} | C: {class_iterator} | G: {gaussian_iterator}: \n\n",mean)
                    #print(f"Var | Sample: {sample_iterator} | C: {class_iterator} | G: {gaussian_iterator}: \n",var)
                    feature_vector = x[sample_iterator, :]
                    
                    covariance_matrix = compute_covariance_matrix(variance_vector=var)
                    inverse_matrix = np.linalg.inv(covariance_matrix)
                    diff = feature_vector - mean
                    result = np.exp(-0.5*np.dot(np.dot(diff.T, inverse_matrix), diff))
                    outputs[sample_iterator, class_iterator, gaussian_iterator] = result
    return outputs


def tensor_mutlivariate_gaussian(x, cmfp):
    def inverse_cov_matrix(var):
        b,c,g,f = var.size()
        covariance_matrix = torch.pow((torch.cat([torch.eye(f) for _ in range(b*c*g)]).view(b,c,g,f,f) * var.unsqueeze(dim=-1)),2)
        return torch.inverse(covariance_matrix)
    
    x = torch.tensor(x)
    cmfp = torch.tensor(cmfp)

    batch_size, num_classes,num_gaussians, num_features, num_parameters = cmfp.size()
    mean = cmfp.view(batch_size, num_classes*num_gaussians*num_features, num_parameters)[...,0].view(batch_size, num_classes, num_gaussians,num_features)
    var = cmfp.view(batch_size, num_classes*num_gaussians*num_features, num_parameters)[...,1].view(batch_size, num_classes, num_gaussians,num_features)

    inverse_matrix = inverse_cov_matrix(var)
    diff = x.view(batch_size, 1, 1, num_features) - mean
    mm1 = diff.view(batch_size, num_classes, num_gaussians, 1, num_features) @ inverse_matrix
    mm2 = mm1 @ diff.view(batch_size, num_classes, num_gaussians, num_features,1)
    exp = -.5 * mm2.view(batch_size, num_classes, num_gaussians)
    return torch.exp(exp)

    


if __name__ == "__main__":
    batch_size=6
    num_features = 6
    granularity = 20
    num_classes = 6
    num_gaussians = 6
    x = np.random.random((batch_size, num_features))
    cmfp = np.random.random((batch_size, num_classes, num_gaussians, num_features, 2))
    fgp = np.random.random((batch_size, num_features, granularity, 2))

    univariate_loop = loop_univariate_gaussian(x, fgp).astype(np.float32)
    univariate_tensor = np.array(tensor_univariate_gaussian(x, fgp)).astype(np.float32)

    if (univariate_loop==univariate_tensor).all():
        print("Univariate test passed")
    else:
        print("Univariate test not passed!")

    multivariate_loop = loop_multivariate_gaussian(x=x, cmfp=cmfp).astype(np.float32)
    multivariate_tensor = np.array(tensor_mutlivariate_gaussian(x=x, cmfp=cmfp)).astype(np.float32)

    if (multivariate_loop==multivariate_tensor).all():
        print("Multivariate test passed")
    else:
        print("Multivariate test not passed!")

