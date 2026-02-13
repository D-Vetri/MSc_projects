import torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
import numpy as np
import os

class ConditionalLoader(Dataset):
    def __init__(self,phase,config):
        super().__init__()
        self.phase = phase
        self.config = config
        self.project_dir = config.project_dir
        self.data_path = os.path.join(self.project_dir,'data',config.data)

        data_file  = f'{self.data_path}/conditional/{config.data.capitalize()}_conditional_{self.phase}.npz'
        data = np.load(data_file)
        
        self.weights,self.SO3 = zip(*data.items())

        self.weights = [int(weights) for weights in self.weights]
        self.SO3 = [torch.tensor(SO3,dtype=torch.float) for SO3 in self.SO3]
        

       
        
    
    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self,index):
        weight = torch.tensor([self.weights[index]]).type(torch.float)
        so3 = self.SO3[index]
        
        
        
        sample = dict(
            condition = weight,
            rot_mat = so3
        )
        
        return sample
         
def custom_collat_fn(batch):
    condition = torch.stack([item['condition'] for item in batch])
    rot_mat = [item['rot_mat'] for item in batch]

    return {'condition':condition,'rot_mat':rot_mat}

def get_conditional_data(phase,config):
    if phase == 'train':
     
        batch_size = config.cond_batch_size
        shuffle = True
    
    elif phase == 'test':
        
        batch_size = config.cond_batch_size // torch.cuda.device_count()
        shuffle = False

    else:
        raise ValueError
    
    dataset = ConditionalLoader(phase,config)
    dloader = DataLoader(dataset,batch_size=batch_size,
                         shuffle=shuffle,num_workers = 2,pin_memory=True,collate_fn=custom_collat_fn)
    return dloader