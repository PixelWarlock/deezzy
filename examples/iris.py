import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from deezzy.fnet import CategoricalFnet
from deezzy.losses import AscendingMeanLoss, SquashingVarianceLoss
from deezzy.modules.linear import LinearReluDropout, LinearRelu
from deezzy.knowledge.knowledge_extractor import KnowledgeExtractor

torch.manual_seed(15)

class IrisDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe):
        self.scaler = MinMaxScaler()
        self.x = self.scale(x=dataframe.iloc[:, 1:5].to_numpy())
        self.y = self.encode(y=dataframe.iloc[:, -1].to_numpy())

    def scale(self, x):
        return self.scaler.fit_transform(X=x)
    
    def encode(self, y):
        unique = np.unique(y)        
        mapping = {k:v for v,k in enumerate(unique)}
        labels=[]
        for label in y:
            labels.append(mapping[label])
        return torch.nn.functional.one_hot(torch.tensor(labels))
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        inputs = self.x[idx]
        targets = self.y[idx]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def main():
    in_features = 4
    granularity = 3
    num_of_gaussians = 10
    num_classes = 3
    learning_rate = 0.001
    batch_size=150
    epochs=1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(os.getcwd(), "outputs/iris_representations")
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'fgp'))
        os.mkdir(os.path.join(save_dir, 'cmfp'))

    dataset = IrisDataset(dataframe=pd.read_csv("./examples/iris_dataset.csv"))
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    backbone = torch.nn.Sequential(
        LinearRelu(in_features=in_features, out_features=256),
        LinearRelu(in_features=256, out_features=256),
        LinearRelu(in_features=256, out_features=256),
        LinearRelu(in_features=256, out_features=256)
    )
    model = CategoricalFnet(backbone=backbone,
                 in_features=in_features,
                 granularity=granularity,
                 num_gaussians=num_of_gaussians,
                 num_classes=num_classes)
    
    #model = model.to(device=device)
    
    class_criterion = torch.nn.CrossEntropyLoss()
    ascending_mean_criterion = AscendingMeanLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
    
        losses = list()
        for inputs, targets in dataloader:

            #inputs=inputs.to(device)
            #targets=targets.to(device)

            optimizer.zero_grad()
            logits, fgp, cmfp = model(inputs)

            criterion_loss = class_criterion(logits, targets)
            am_loss = ascending_mean_criterion(fgp)

            loss = criterion_loss + am_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            torch.save(fgp, os.path.join(save_dir, f"fgp/epoch_{epoch}.pt"))
            torch.save(cmfp, os.path.join(save_dir, f"cmfp/epoch_{epoch}.pt"))

        print(f"Epoch: {epoch} | Loss: {np.mean(losses)}")

    KnowledgeExtractor(fgp=fgp.detach(), cmfp=cmfp.detach())()
    
if __name__ == "__main__":
    main()