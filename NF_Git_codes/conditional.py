from torch import nn
import torch
import numpy as np
import os
class ConditionalTransform(nn.Module):
    """
    The conditional transformation is done with ReLU and Linear MLP layers
    A total of 4 layers
    Ni = input dimension
    No = output dimension
    Nh = inter neuron layers: Default = 64
    First layer and last layer is linear
    """
    def __init__(self,Ni,No,Nh=64):
        super().__init__()
        layers = []
        self.first_lay = nn.Linear(Ni,Nh) # could do nn.sequential as well, but not trying it now,also the authors
        layers.append(nn.ReLU())          #use a custom first and last layer in the forward. Hence it could affect
        layers.append(nn.Linear(Nh, Nh))  #the working of Sequential
        layers.append(nn.ReLU())
        layers.append(nn.Linear(Nh, Nh))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(Nh, Nh))   # 4 layers of MLP
        self.last_relu = nn.ReLU() # conditional finish such that all positive values are taken and negatives are zeros
        self.last_lay = nn.Linear(Nh,No)
        self.layers = nn.ModuleList(layers)

    def forward(self,x): # forward function for nn.Module
        x = self.first_lay(x) #linear scale to MLP hyper-dimension(with n,64 dim)
        x1 = x.clone()
        for layer in self.layers:
            x = layer(x)
        x = self.last_relu(x+x1)
        return self.last_lay(x)
        



