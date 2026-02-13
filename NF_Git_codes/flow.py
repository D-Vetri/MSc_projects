
#Creating the Flow model

import torch
import torch.nn as nn
import numpy as np
from quar import affine
from Mobius import mobius,MobiusFlow


def flow(config):
    return Flow(config)

_permuter = torch.tensor(
    [[0,1,2],[1,2,0],[2,0,1],[0,1,2],[1,2,0],[2,0,1]] # 6 may not be needed
)                                                     # as we are not using 6x6 uncons.W



class Flow(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.layers = config.layers
        self.conditions = config.condition
        self._permute = _permuter
        

        if self.conditions:
            self.feature_dim = config.feature_dim
        else:
            self.feature_dim=None
    
        layers = []
    # for now we are going for last_affine layer(i.e in generative direction), 
    # can add as a config later
       
        layers.append(affine(config,self.feature_dim))
        for i in range(self.layers):
            a = mobius(config,self.feature_dim)
            if a !=None:
                layers.append(a)
            a = affine(config,self.feature_dim)
            if a!=None and (i != self.layers-1):
                layers.append(a)
        print("Flow layers= ",len(layers))
        self.layers=nn.ModuleList(layers)

    def forward(self, rotations,feature=None,inverse=False,draw=False):
        if inverse:
            return self.inverse(rotations,feature,draw)
        permute = self._permute.to(rotations.device)

        ldjs = 0
        exchange_count = 0

        if not self.conditions:
            feature=None
        
        for i in range(len(self.layers)): #permute sent is just one row vector
            rotations,ldj = self.layers[i](
                rotations,permute[exchange_count%6],feature
            )
            ldjs +=ldj
            if (isinstance(self.layers[i],MobiusFlow)):
                exchange_count+=1
        
        return rotations,ldjs
    
    def inverse(self, rotations, feature=None,draw=False):
        permute = self._permute.to(rotations.device)

        ldjs = 0
        exchange_count = len(self.layers)

        if not self.conditions:
            feature=None
        
        for i in range(len(self.layers))[::-1]: # reversing the range
            if(isinstance(self.layers[i],MobiusFlow)):
                exchange_count -=1
            rotations,ldj = self.layers[i].inverse(
                rotations,permute[exchange_count%6],feature
            )

            ldjs += ldj
    
        return rotations,ldjs





