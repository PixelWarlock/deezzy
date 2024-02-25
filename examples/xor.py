import torch
import numpy as np
from deezzy.fnet import Fnet
from deezzy.modules.linear import LinearRelu

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

    dataset = XorDataset()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    backbone = torch.nn.Sequential(
        LinearRelu(in_features=in_features, out_features=128),
        LinearRelu(in_features=128, out_features=128),
        LinearRelu(in_features=128, out_features=128)
    )
    model = Fnet(backbone=backbone,
                 in_features=in_features,
                 granularity=granularity,
                 num_gaussians=num_of_gaussians,
                 num_classes=num_classes)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
    
        losses = list()
        for inputs, target in dataloader:

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch: {epoch} | Loss: {np.mean(losses)}")

if __name__ == "__main__":
    main()