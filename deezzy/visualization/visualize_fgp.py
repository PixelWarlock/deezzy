import os
import torch
import matplotlib.pyplot as plt
from deezzy.memberships.gaussian import Gaussian

COLORS = [
    'b',
    'g',
    'r',
]

def get_figure_grid(f:int):
    if f %2==0:
        n=int(f/2)
        return n, n
    else:
        nrows=f-2
        ncols=2
    return nrows, ncols

def get_memberships(fgp, domain):
    gaussian = Gaussian(univariate=True)
    batched_fgp = torch.cat([fgp.unsqueeze(dim=0) for _ in range(domain.size()[0])], dim=0)
    memberships = gaussian(domain, batched_fgp)
    return memberships

def select_feature(memberships, feature_id):
    return memberships[:,feature_id,:]

def visualize(fgp:torch.Tensor):
    f,g,_= fgp.shape
    
    domain = torch.cat([torch.arange(0.,1.01, 0.01).unsqueeze(dim=0) for _ in range(f)], dim=0).T
    memberships = get_memberships(fgp, domain)

    nrows, ncols = get_figure_grid(f)
    fig, axes = plt.subplots(nrows, ncols)
    axes = axes.ravel()

    for i in range(f):
        feature_memberships = select_feature(memberships=memberships, feature_id=i)
        axes[i].set_title(f"Feature: {i}")
        axes[i].grid(True)
        for m in range(g):
            axes[i].plot(torch.arange(0.,1.01, 0.01).detach().numpy(),feature_memberships[:,m].detach().numpy(), COLORS[m])

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    fgp = torch.load("outputs/iris_representations/fgp/epoch_499.pt")
    visualize(fgp=fgp)