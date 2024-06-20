import pandas as pd
from deezzy.knowledge.membership import Membership
from deezzy.knowledge.feature import Feature

class RulesRunner:

    def __init__(self, features_file:str, rules_file:str):
        
        self.parse_features(features_file=features_file)
        with open(rules_file, 'r') as f:
            self.rules = f.read()

    def parse_features(self, features_file):
        dataframe = pd.read_csv(features_file)
        num_features = len(dataframe['Feature index'].unique())
        features = []
        for i in range(num_features):
            subset = dataframe[dataframe['Feature index'] == i]
            memberships = []
            granularity = len(subset)
            for m,row in enumerate(subset.iterrows()):
                memberships.append(Membership(index=m, begining=row[1].Begining, end=row[1].End, num_gaussians=granularity))
            features.append(Feature(index=i, memberships=memberships))
        self.features = features
            
    def __call__(self, sample):
        for value, feature in zip(sample, self.features):
            feature(value)
        
        for feature in self.features:
            exec(f"Feature_{feature.index} = feature")

        exec(self.rules)
        return locals()['CATEGORY']

if __name__ == "__main__":
    feature_file = './results/features.csv'
    rules_file = './results/rules.txt'
    rules_runner = RulesRunner(features_file=feature_file, rules_file=rules_file)

    sample = [5.1,3.5,1.4,0.2]
    print(rules_runner(sample=sample))