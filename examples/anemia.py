import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from deezzy.fnet import CategoricalFnet
from deezzy.modules.linear import LinearRelu
from deezzy.losses import AscendingMeanLoss, SquashingVarianceLoss
from deezzy.knowledge.knowledge_extractor import KnowledgeExtractor

#torch.manual_seed(15)

class AnemiaDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe):
        self.scaler = MinMaxScaler()

        dataframe.drop_duplicates(inplace=True)
        dataframe = dataframe.drop(dataframe[dataframe['HGB'] < 0].index[0])
        dataframe = dataframe.drop(dataframe[dataframe['MCV'] < 0].index[0])

        x,y = self.split(dataframe=dataframe)
        self.scaler.fit(X=x)
        self.x = self.scale(x=x)
        self.y = self.encode(y=y)

    def split(self, dataframe):
        x = dataframe.drop(['Diagnosis'], axis=1).to_numpy()
        y = dataframe['Diagnosis'].to_numpy()
        return x,y

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
        y = torch.argmax(targets, dim=-1).type(torch.int).detach().cpu().numpy()
        yhat = torch.argmax(logits, dim=-1).type(torch.int).detach().cpu().numpy()
        accuracy = accuracy_score(y_true=y, y_pred=yhat)
    return accuracy

def main():
    in_features = 14
    granularity = 2
    num_of_gaussians = 20
    num_classes = 9
    learning_rate = 0.0001
    batch_size=1281
    epochs=25000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(os.getcwd(), "outputs/anemia_representations")
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'fgp'))
        os.mkdir(os.path.join(save_dir, 'cmfp'))

    dataset = AnemiaDataset(dataframe=pd.read_csv("./examples/diagnosed_cbc_data_v4.csv"))
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    backbone = torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=256),
        torch.nn.Linear(in_features=256, out_features=256),
        torch.nn.Linear(in_features=256, out_features=256),
    )
    model = CategoricalFnet(backbone=backbone,
                 in_features=in_features,
                 granularity=granularity,
                 num_gaussians=num_of_gaussians,
                 num_classes=num_classes) #.to(device=device)
        
    class_criterion = torch.nn.CrossEntropyLoss()
    ascending_mean_criterion = AscendingMeanLoss()
    squishing_var_criterion = SquashingVarianceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
    
        losses = list()
        accuracies = list()
        for inputs, targets in dataloader:
            
            #inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, fgp, cmfp = model(inputs)

            criterion_loss = class_criterion(logits, targets)
            am_loss = ascending_mean_criterion(fgp)
            #sv_loss = squishing_var_criterion(fgp)

            loss = criterion_loss + am_loss #+ sv_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            accuracies.append(calculate_accuracy(logits=logits, targets=targets))

            torch.save(fgp, os.path.join(save_dir, f"fgp/epoch_{epoch}.pt"))
            torch.save(cmfp, os.path.join(save_dir, f"cmfp/epoch_{epoch}.pt"))

        print(f"Epoch: {epoch} | Loss: {np.mean(losses)} | Accuracy: {np.mean(accuracies)}")

    KnowledgeExtractor(fgp=fgp.detach(), cmfp=cmfp.detach(), scaler=dataset.scaler)()
    
if __name__ == "__main__":
    main()