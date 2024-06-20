import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from deezzy.fnet import CategoricalFnet
from deezzy.losses import AscendingMeanLoss, SparseCategoricalLoss
from deezzy.modules.linear import LinearRelu, LinearLeakyRelu
from deezzy.knowledge.knowledge_extractor import KnowledgeExtractor

torch.manual_seed(15)

class IrisDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X=dataframe.iloc[:, 1:5].to_numpy())
        self.x = self.scale(x=dataframe.iloc[:, 1:5].to_numpy())
        self.y = self.encode(y=dataframe.iloc[:, -1].to_numpy())

    def scale(self, x):
        return self.scaler.transform(X=x)
    
    def encode(self, y):
        unique = np.unique(y)        
        #mapping = {k:v for v,k in enumerate(unique, start=1)}
        mapping = {k:v for v,k in enumerate(unique)}
        labels=[]
        for label in y:
            labels.append(mapping[label])
        #return torch.tensor(labels) 
        return torch.nn.functional.one_hot(torch.tensor(labels))
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        inputs = self.x[idx]
        targets = self.y[idx]
        return torch.from_numpy(inputs).type(torch.float32), targets.type(torch.float32) #torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
def calculate_accuracy(logits, targets):
    with torch.no_grad():
        #y = targets.type(torch.int).detach().numpy()
        #yhat = logits.type(torch.int).detach().numpy()
        y = torch.argmax(targets, dim=-1).type(torch.int).detach().numpy()
        yhat = torch.argmax(logits, dim=-1).type(torch.int).detach().numpy()
        accuracy = accuracy_score(y_true=y, y_pred=yhat)
    return accuracy

def main():
    in_features = 4
    granularity = 2
    num_of_gaussians = 8
    num_classes = 3
    learning_rate = 0.0001
    batch_size=150
    epochs=200

    save_dir = os.path.join(os.getcwd(), "outputs/iris_representations")
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'fgp'))
        os.mkdir(os.path.join(save_dir, 'cmfp'))

    dataset = IrisDataset(dataframe=pd.read_csv("./examples/iris_dataset.csv"))
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    backbone = torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=128),
        torch.nn.Linear(in_features=128, out_features=128),
        torch.nn.Linear(in_features=128, out_features=128),

    )
    model = CategoricalFnet(backbone=backbone,
                 in_features=in_features,
                 granularity=granularity,
                 num_gaussians=num_of_gaussians,
                 num_classes=num_classes)
        
    class_criterion = torch.nn.CrossEntropyLoss()
    #class_criterion = SparseCategoricalLoss()
    ascending_mean_criterion = AscendingMeanLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
    
        losses = list()
        accuracies = list()
        for inputs, targets in dataloader:
            
            optimizer.zero_grad()
            logits, fgp, cmfp = model(inputs)

            criterion_loss = class_criterion(logits, targets)
            am_loss = ascending_mean_criterion(fgp)

            loss = criterion_loss + am_loss 

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            accuracies.append(calculate_accuracy(logits=logits, targets=targets))

            torch.save(fgp, os.path.join(save_dir, f"fgp/epoch_{epoch}.pt"))
            torch.save(cmfp, os.path.join(save_dir, f"cmfp/epoch_{epoch}.pt"))

        print(f"Epoch: {epoch} | Loss: {np.mean(losses)} | Accuracy: {np.mean(accuracies)}")

    KnowledgeExtractor(fgp=fgp.detach(), cmfp=cmfp.detach(), scaler=dataset.scaler)() #
    
if __name__ == "__main__":
    main()