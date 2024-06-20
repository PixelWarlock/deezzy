
class Feature:
    def __init__(self,
                 index:int,
                 memberships:list,
                 name:str=None):
        
        self.index = index
        self.memberships = memberships
        if name is not None:
            self.name = name

    def __repr__(self) -> str:
        return self.name if hasattr(self, 'name') else f"{self.index}"
    
    def __iter__(self):
        return iter(self.memberships)
    
    def __call__(self, value):
        for membership in self.memberships:
            if membership.begining <= value <= membership.end:
                self.adjective = str(membership)