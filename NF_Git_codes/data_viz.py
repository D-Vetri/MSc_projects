import torch 
import main_eval_uncon
import os 
import numpy as np
from config import Config
import sys
sys.path.append('/home/divakar/RotationNormFlow')
import pathlib as pb



con = Config()
file_dir = r"/home/divakar/odf-flow/SO3_NF"
file = os.path.join(file_dir,'data',"45k_cube_rot_mat.npy") 
data = np.load(file)
data = data.reshape(-1,3,3)
print(data.shape)
# config = Config()
# data = torch.from_numpy(data)
# fig = viz.so3_sample_viz(data,
#                          torch.ones(data.size(0)),
#                          ax=None,
#                          fig=None,
#                          scatter_size=config.scatter_size)
# os.makedirs('plot/raw',exist_ok=True)
# save_path = f'plot/raw/base_data.png'
# print("Saving plot to",save_path)
# fig.show()
# fig.savefig(save_path)ls


file_path = os.path.dirname(os.path.abspath(__file__ ))
a =  torch.rand(3,3)
name = 'trail_1'


new_path = os.path.join(con.project_dir,'data','orients.npy')
data = np.load(new_path)
data = torch.from_numpy(data)
