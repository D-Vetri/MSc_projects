import os
from os.path import join,basename,dirname,abspath
import json
import logging
import shutil
import csv
import torch
import numpy as np

class TrainClock(object):
    """
    Provides clock for tracking epochs and iterations while training
    """

    def __init__(self, schedulers=None):
        self.epoch = 0
        self.minibatch = 0
        self.iteration = 0 
        self.schedulers = schedulers

    def tick(self):
        self.minibatch+=1
        self.iteration += 1
    
    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        clock_dict = dict(
            epoch = self.epoch,
            minibatch = self.minibatch,
            iteration = self.iteration
        )
        return clock_dict
    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.iteration = clock_dict['iteration']

        return clock_dict

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def requires_grad(xs,req=False):
    if not(isinstance(xs,tuple) or isinstance(xs,list)):
        xs = tuple(xs)
    for x in xs:
        x.requires_grad_(req)


def dict_get(dicts, key,default,default_device='cuda'):
    v = dicts.get(key)
    default_tensor = torch.tensor([default]).float().to(default_device)
    if v is None or v.nelement()==0:
        return default_tensor
    else:
        return v
    
def assertion(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()

def softplus(x):
    return torch.log(1. + torch.exp(x))
def softplus_inv(x):
    return torch.log(-1. + torch.exp(x))
def softmax(x):
    ex = torch.exp(x - torch.max(x))
    return ex / torch.sum(ex)



        