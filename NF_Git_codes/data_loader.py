import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RotData(Dataset):
    def __init__(self,phase,config):
        self.config = config
        file_dir = config.project_dir
        
        folder = os.path.join(file_dir,'data',config.data) 
        
        
        file = f'{folder}/{config.data}_{phase}.npy'
        # breakpoint()
        print(f'\n{file}')
        try:
            model_data = np.load(file)
        except Exception as e:
            print(f"{e} : Ensure the file is properly named and exists")
            
        print(f'The data size for {phase} is {model_data.shape}')

        self.rotations = torch.from_numpy(model_data).float()
        self.length = self.rotations.shape[0]

    def __len__(self):
        '''we need to define __len__ and __getitem__ as part of the 
           abstract class: Dataset  '''
        return self.length

    def __getitem__(self, index):
        rotation = self.rotations[index]
        sample = dict(
            rot_mat =  rotation,
        )

        return sample
    
def get_dataloader_rot(phase,config):
    if phase.lower() == 'train':
        shuffle = True
        batch_size = config.batch_size
    elif phase.lower() == 'test':
        
        shuffle = False
        batch_size = config.batch_size
        batch_size = config.batch_size// torch.cuda.device_count()
        

    else:
        raise ValueError
    
    dset = RotData(phase,config)
    dloader = DataLoader(dset,batch_size=batch_size,
                         shuffle=shuffle,num_workers=1,pin_memory=True)
    
    return dloader







