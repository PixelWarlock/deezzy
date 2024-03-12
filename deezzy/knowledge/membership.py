import numpy as np

class Membership:
    def __init__(self,
                 index:int,
                 begining:np.array,
                 end:np.array,
                 num_gaussians:int):
        
        self.index = index
        self.begining = begining
        self.end = end
        self.mapping = self.make_mapping(num_gaussians=num_gaussians)

    def make_mapping(self, num_gaussians):
        if num_gaussians == 1:
            raise("Doesn't make sense")
        
        if num_gaussians%2!=0:
            mapping = ['small', 'medium', 'high']
        else:
            mapping = ['small', 'high']
        
        leftover = num_gaussians - len(mapping)
        if leftover == 0:
            return dict(zip(list(range(len(mapping))), mapping)) 
        else:
            add = leftover/2
            for _ in range(add):
                start_element = f'VERY_{mapping[0]}'
                end_element = f'VERY_{mapping[-1]}'
                mapping.insert(0, start_element)
                mapping.insert(len(mapping), end_element)
        
        return dict(zip(list(range(len(mapping))), mapping)) 
    
    def __repr__(self) -> str:
        return  self.mapping[self.index] #f"{self.index}"