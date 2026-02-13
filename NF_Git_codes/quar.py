'''
For quarternion transformation, we will use the unconstrained 4x4 matrix.
Hence no abalations from the original SO3 NF paper is needed.
For our case, where we deal with cubic crystals, we go with the conditions
used for SymSOL I dataset. 
'''


import torch
from torch import nn
import numpy as np
import pytorch3d.transforms as pytf
from conditional import ConditionalTransform as ct
# from scipy import linalg as la

def affine(config,feature_dim,first_layer_conditional=False):
    if first_layer_conditional:
        return conditional16Trans(feature_dim)
    if config.condition:
        return conditional16Trans(feature_dim)
    else:
        return unconditional16Trans()

def my_det_3_3(A):
    det00 = A[..., 1, 1]*A[..., 2, 2]-A[..., 1, 2]*A[..., 2, 1]
    det01 = A[..., 1, 2]*A[..., 2, 0]-A[..., 1, 0]*A[..., 2, 2]
    det02 = A[..., 1, 0]*A[..., 2, 1]-A[..., 1, 1]*A[..., 2, 0]
    return det00*A[..., 0, 0]+det01*A[..., 0, 1]+det02*A[..., 0, 2] #why all +ve?


def my_det_4_4(A):
    det00 = A[..., 0, 0]*my_det_3_3(A[..., 1:, [1, 2, 3]])
    det01 = A[..., 0, 1]*my_det_3_3(A[..., 1:, [0, 2, 3]])
    det02 = A[..., 0, 2]*my_det_3_3(A[..., 1:, [0, 1, 3]])
    det03 = A[..., 0, 3]*my_det_3_3(A[..., 1:, [0, 1, 2]])
    return det00-det01+det02-det03

def calculate_16(mat, rotation):
    quat = pytf.matrix_to_quaternion(rotation)
    quat = mat @ quat.reshape(-1, 4, 1)
    length = quat.norm(dim=-2, keepdim=True)
    t_rotation = pytf.quaternion_to_matrix((quat/length).reshape(-1, 4)) #WQ/||Wq|| # Normalizing
    return t_rotation, my_det_4_4(mat).abs().log()-4*length.reshape(-1).log()

# since we are working on just the effective kind of quarternion transformation
# defining just one class from the parent SO3 NF code is sufficient(i.e no abalations)
# Thus, unconstrained 4x4 Transformation. We define both the condition and unconditional
# classes

class conditional16Trans(nn.Module):
    '''
    Class for 4x4 quarternion transformation with conditions.
    '''
    def __init__(self, feature_dim):
        super().__init__()
        self.NN = ct(feature_dim,16)
    
    def forward(self,rotations,permute=None,feature=None):
        mat  = self.NN(feature).reshape(-1,4,4) + \
            torch.eye(4,device=rotations.device).unsqueeze(0)
        return calculate_16(mat,rotations)
    
    def inverse(self, rotations, permute=None,feature=None):
        mat = self.NN(feature).reshape(-1,4,4) +\
            torch.eye(4,device=rotations.device).unsqueeze(0)
        mat = torch.linalg.inv(mat)
        return calculate_16(mat,rotations)

class unconditional16Trans(nn.Module):
    def __init__(self):
        super().__init__()
        self.mat = nn.Parameter(torch.eye(4).unsqueeze(0)+\
                                torch.rand(1,4,4)*1e-3) 
        #random unconstrained matrix. 

    def forward(self, rotations, permute=None,feature=None):
        mat = self.mat
        return calculate_16(mat,rotations)

    def inverse(self, rotations,permute=None,feature=None):
        mat = torch.linalg.inv(self.mat)
        return calculate_16(mat,rotations)
      

    

