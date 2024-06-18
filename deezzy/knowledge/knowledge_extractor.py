import os
import torch
import itertools
import numpy as np
from deezzy.knowledge.feature import Feature
from deezzy.memberships.gaussian import Gaussian
from deezzy.knowledge.membership import Membership


class KnowledgeExtractor:
    def __init__(self,
                 fgp:torch.tensor,
                 cmfp:torch.tensor):
        
        self.fgp = fgp
        self.cmfp = cmfp
        self.univariate = Gaussian(univariate=True)
        self.multivariate = Gaussian(univariate=False)
        
    def get_sliced_array(self, array:np.ndarray, current_gaussian:int):
        return array[:,current_gaussian:current_gaussian+2]

    def get_memberships(self, array:np.ndarray, num_gaussians:int, domain:np.ndarray):
        memberships = []
        last_end_point = None
        for m in range(num_gaussians):
            if m == 0:
                start_point = domain[0]
            else:
                start_point = last_end_point

            if m==num_gaussians-1:
                end_point = domain[-1]
                memberships.append(Membership(index=m, begining=start_point, end=end_point, num_gaussians=num_gaussians))
            else:
                sliced_array = self.get_sliced_array(array=array, current_gaussian=m)
                end_index = np.where(sliced_array[:,1] > sliced_array[:,0])[0][0]
                end_point = domain[end_index]

                memberships.append(Membership(index=m, begining=start_point, end=end_point, num_gaussians=num_gaussians))
                last_end_point=end_point
                
        return memberships

    def get_features(self, 
                     domain:torch.tensor, 
                     num_features:int, 
                     num_gaussians:int,
                     names:list=None):
        
        batched_fgp = torch.cat([self.fgp.unsqueeze(dim=0) for _ in range(domain.size()[0])], dim=0)
        ff_array = self.univariate(domain,batched_fgp).numpy()
        domain = domain.numpy()

        features = []
        for feature_id in range(num_features):
            # for each feature find a range where specific gaussian applies (its membership is higher than other gaussian)
            memberships = self.get_memberships(array=ff_array[:,feature_id,:], num_gaussians=num_gaussians, domain=domain[:,feature_id])
            if names is not None:
                name = names[feature_id]
                feature = Feature(index=feature_id, memberships=memberships, name=name)
            else:
                feature = Feature(index=feature_id, memberships=memberships)
            features.append(feature)
        return features

    def construct_statement(self, adjectives:np.array, assigment:int):
        num_features = len(adjectives)
        #statement = "if "
        statement = ""
        for f in range(num_features):
            statement += f'Feature_{f}.adjective == "{adjectives[f]}"'
            if f < num_features-1:
                statement += " and "
            else:
                statement += "\n"

        statement += f"    CATEGORY={assigment}\n"
        return statement
    
    def explain_features(self, features:list):
        statemets = ""
        for feature in features:
            statemets += f"Feature: {feature.index}\n"
            for membership in feature.memberships:
                statemets += f"Is {membership} when is in-between <{'{0:0.4f}'.format(membership.begining)},{'{0:0.4f}'.format(membership.end)}>\n"
        return statemets
    
    def group_statements(self, statements):

        grouped_statements = ""
        
        lines = statements.split('\n')
        category_dict = {}
        for line_index,line in enumerate(lines):
            if "CATEGORY" in line:
                category=int(line.split("=")[1])
                category_dict[line_index] = category

        # check where category is the same
        categories = np.unique(list(category_dict.values()))
        for c in categories:
            category_statements = np.where(list(category_dict.values())==c)[0]
            category_lines = np.take(list(category_dict.keys()), category_statements)
            explain_lines_indexes = category_lines - 1
            explain_lines = [lines[eli] for eli in explain_lines_indexes]

            till_last_explain_line = explain_lines[:-1]
            grouped_statements += " or ".join(till_last_explain_line)
            grouped_statements += f" or {explain_lines[-1]}:\n    CATEGORY={c}\n"

        return grouped_statements

    def explain(self, features:list):

        print(50*'-')
        print(self.explain_features(features=features))
        print(50*'-')
        
        adjectives_values = []
        adjectives_names = []

        #cartesian product of number of featres X number of gaussians
        cartesian_product = list(itertools.product(*features))
        for product in cartesian_product:
            adjectives_values.append([p.index for p in product])
            adjectives_names.append([repr(p) for p in product])
            
        adjectives = torch.tensor(adjectives_values)
        b = adjectives.size()[0]
        batched_cmfp = torch.cat([self.cmfp.unsqueeze(dim=0) for _ in range(b)], dim=0)
        assigments = torch.max(torch.sum(self.multivariate(adjectives, batched_cmfp), dim=-1), dim=-1).indices.numpy()
        adjectives = adjectives.numpy()

        statements = ""
        for adj, assigment in zip(adjectives_names, assigments):
            statements += self.construct_statement(adjectives=adj, assigment=assigment)

        return self.group_statements(statements)

    def __call__(self, step:float = 0.01):
        f,g,_ = self.fgp.shape
        domain = torch.cat([torch.arange(start=0.0, end=1.+step, step=step).unsqueeze(dim=0) for _ in range(f)], dim=0).T
        features = self.get_features(domain=domain, num_features=f, num_gaussians=g)
        knowledge = self.explain(features=features)
        print(knowledge)
    
if __name__ == "__main__":
    fgp_path = "outputs/iris_representations/fgp/epoch_499.pt"
    cmfp_path = "outputs/iris_representations/cmfp/epoch_499.pt"

    fgp =  torch.load(fgp_path).detach().cpu()
    cmfp = torch.load(cmfp_path).detach().cpu()

    KnowledgeExtractor(fgp=fgp,
                       cmfp=cmfp)()


