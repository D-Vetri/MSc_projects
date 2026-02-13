import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# script to store the Eulers for different weights as sets of npy file of rotations within a single npz

data_path = Path(r"") # add required path
all_rot_mats = {}
def gen_rots_npz(save_path):
    flag = 0
    
    #For our current model, the weights are stored as the file name
    #for more complex data, json files would be better for generating conditional data.
    # The loop can be modified to work on the 'key-value' format that json uses 
                  
    if os.path.isdir(data_path):
        for file in data_path.iterdir():
            name = Path(file).stem
            weight = ''
            for c in reversed(name): #For our current model, the weights are stored as the file name
                if c.isdigit():      # for more complex data, json files would be better for generating conditional data 
                    weight = c + weight
                else:
                    break
            
            eulers = np.loadtxt(file)

            rot_mats = np.zeros((eulers.shape[0],3,3))

            for i in range((eulers.shape[0])):
                rot_mats[i] = (R.from_euler('ZXZ',eulers[i,:],degrees=False).as_matrix())

            all_rot_mats[weight] = rot_mats
            #checking if the files are properly named
            if flag<10:
                print('These are the weights keys:',weight, 'and the file is ',file)
                
                flag+=1

    return all_rot_mats

rot_mats_dataset = gen_rots_npz(None)


np.savez(f'{Path(data_path).stem}.npz',**rot_mats_dataset)