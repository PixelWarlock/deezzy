import os
import torch
import numpy as np
from deezzy.fnet import Fnet
from deezzy.losses import AscendingMeanLoss
from deezzy.modules.linear import LinearRelu, LinearReluDropout

#torch.manual_seed(0)

class XorDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        """
        Data is int format:
        x0, x1, y
        """
        self.data = {
            (0,0):0,
            (0,1):1,
            (1,0):1,
            (1,1):0
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample, label = list(self.data.items())[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
def main():
    in_features = 2
    granularity = 2
    num_of_gaussians = 2
    num_classes = 2
    learning_rate = 0.001
    batch_size=4
    epochs=500

    save_dir = os.path.join(os.getcwd(), "outputs/xor_representations")
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'fgp'))
        os.mkdir(os.path.join(save_dir, 'cmfp'))

    dataset = XorDataset()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False) #, num_workers=8

    backbone = torch.nn.Sequential(
        LinearReluDropout(in_features=in_features, out_features=256),
        LinearReluDropout(in_features=256, out_features=256),
        LinearReluDropout(in_features=256, out_features=256)
    )
    model = Fnet(backbone=backbone,
                 in_features=in_features,
                 granularity=granularity,
                 num_gaussians=num_of_gaussians,
                 num_classes=num_classes)

    class_criterion = torch.nn.BCELoss()
    ascending_mean_criterion = AscendingMeanLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
    
        losses = list()
        for inputs, target in dataloader:

            optimizer.zero_grad()
            logits, fgp, cmfp = model(inputs)
            criterion_loss = class_criterion(logits, target)
            am_loss = ascending_mean_criterion(fgp)
            loss = criterion_loss + am_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            torch.save(fgp, os.path.join(save_dir, f"fgp/epoch_{epoch}.pt"))
            torch.save(cmfp, os.path.join(save_dir, f"cmfp/epoch_{epoch}.pt"))

        print(f"Epoch: {epoch} | Loss: {np.mean(losses)}")

if __name__ == "__main__":
    main()