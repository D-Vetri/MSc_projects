import numpy as np
import pandas as pd
import csv
from scipy.spatial.transform import Rotation as r
import random as rd
import os
from pathlib import Path

# Note : The codes currently contain hardcoded directory paths to read and write from simulated data files. To be refactored when to code is developed for deployability.
class Transformers:
    def __init__(self):
        pass
    def euler_to_matrix(self,Eulers,i,all_in_one=False):
        '''
        - Function to convert Euler angles to Rotation matrix.
        - Uses the bunge convention(ZXZ).
        - The exact proper Euler convention and the Formula for Rotation matrix generation
            is taken from Wikipedia.
        - Source:(https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)
        - use numpy to create arrays(not torch.tensor)
        - The Euler angles are to be used in Radians:
            # for α and γ, the range is defined modulo 2π radians. For instance, a valid range could be [−π,π].
            # for β, the range covers π radians (but can not be said to be modulo π). For example, it could be [0,π] or [−π/2,π/2].
            source: https://en.wikipedia.org/wiki/Euler_angles#Signs,_ranges_and_conventions

        Input: Euler angles(1,3) arrays
        ouput: Rotation Matrix(3,3) arrays
        '''
        alpha = Eulers[0] 
        beta = Eulers[1]
        gamma = Eulers[2]
        cos_a,cos_b,cos_g = np.cos(alpha),np.cos(beta),np.cos(gamma)
        sin_a,sin_b,sin_g = np.sin(alpha),np.sin(beta),np.sin(gamma)
        #first Row
        R11 = (cos_a*cos_g) - (cos_b*sin_a*sin_g)
        R12 = (-cos_a*sin_g) - (cos_b*cos_g*sin_a)
        R13 = sin_a*sin_b
        #second row
        R21 = cos_g*sin_a + (cos_a*cos_b*sin_g)
        R22 = (cos_a*cos_b*cos_g) - (sin_a*sin_g)
        R23 = -cos_a*sin_b
        #third row
        R31 = sin_b*sin_g
        R32 = cos_g*sin_b
        R33 = cos_b
        if not all_in_one:
            Rotation_matrix = np.array([[R11,R12,R13],
                                    [R21,R22,R23],
                                    [R31,R32,R33],])
        else:
            Rotation_matrix = self.in_mat_calc(cos_a,cos_b,cos_g,sin_a,sin_b,sin_g)
        if i==0:
            
            return Rotation_matrix,alpha,beta,gamma
        return Rotation_matrix


    def in_mat_calc(self,cos_a,cos_b,cos_g,sin_a,sin_b,sin_g):
        print('Using all_in_one')
        Rot_mat_in =np.array( [[cos_a*cos_g - (cos_b*sin_a*sin_g), (-cos_a)*(sin_g)-(cos_b*cos_g*sin_a), sin_a*sin_b],
                    [cos_g*sin_a + cos_a*cos_b*sin_g, cos_a*cos_b*cos_g - (sin_a*sin_g), (-cos_a)*sin_b],
                    [sin_b*sin_g, cos_g*sin_b, cos_b],
                    ])
        
        return Rot_mat_in


    def perform_euler_to_matrix(self,is_text = False):
        if not is_text:
            Euler_arrays = pd.read_csv(r"",header=None)#add path to read from CSV
        else:
            Euler_arrays = pd.read_csv(r"",header=None,delimiter=',') # add path if the file is text
        Euler_arrays =Euler_arrays.to_numpy()
        Rotation_matrix = np.zeros((Euler_arrays.shape[0],3,3))
        for i in range(Euler_arrays.shape[0]):
            if i==0:
                Rot_mat,a,b,c = self.euler_to_matrix(Euler_arrays[i],i,all_in_one=True)
            else:
                Rot_mat = self.euler_to_matrix(Euler_arrays[i],i)    
            Rotation_matrix[i] = Rot_mat
        return Rotation_matrix
  

    #scipy check:
    def using_scipy(self,Euler_arrays):
        Euler_arrays = np.loadtxt(Euler_arrays,delimiter=',')
        Rotation_matrix2 = r.from_euler('ZXZ',Euler_arrays,degrees=False).as_matrix()
        print(f'\n{Rotation_matrix2.shape} is the shape of the matrix ndarray generated using scipy' )
        return Rotation_matrix2



    def write_rots_csv(self,Rotation_matrix):
        for i in range(Rotation_matrix.shape[0]):
            Rotation_matrix[i] = Rotation_matrix[i].T
        print(Rotation_matrix.shape)
        Rot_mat_write = Rotation_matrix.reshape(Rotation_matrix.shape[0],9)
        def csv_write():
            with open('Rot_comparison.csv','w+') as f:
                pen = csv.writer(f,delimiter=",")
                pen.writerows(Rot_mat_write)

        # csv_write()

    def matrix_to_euler(self,mat):
        alpha = np.arctan(mat[0][2]/(-mat[1][2]))
        beta = np.arccos(mat[2][2])
        gamma = np.arctan(mat[2][0]/mat[2][1])
        euler_angles = np.array([alpha,beta,gamma])
        return euler_angles

def rot_mat_to_euler(file,euler_save_path):
    rot_for_euler = np.load(file)
    
    # rot_for_euler = np.zeros((1,3,3))
    rot_for_euler= rot_for_euler.reshape(-1,3,3)

    # print(rot_for_euler.shape)
    euler_angles = np.zeros((rot_for_euler.shape[0],3))
    for i in range(rot_for_euler.shape[0]):
        euler_angles[i] = r.from_matrix(rot_for_euler[i]).as_euler('ZXZ',degrees=False)
    print(euler_angles.shape)
    np.savetxt(f'{euler_save_path}\\{file.stem}.txt',euler_angles,delimiter=' ')
    # return euler_angles

def rots_to_npy(save_path,inp_rot=None,train=True):
    save_file = os.path.basename(save_path)
    if inp_rot is None:
        rots_npy = pd.read_csv(r"",header=None)
        rots_npy = rots_npy.to_numpy()
        print(rots_npy.shape)
        assert rots_npy.shape[1] == 9,"Ensure only rotational matrices are given as input"
        rots_npy = rots_npy.reshape(-1,3,3)
        for i in range(rots_npy.shape[0]):
            rots_npy[i] = rots_npy[i].T
    else:
        rots_npy = inp_rot
    lim = int(rots_npy.shape[0]*0.2)
    rng_idx = rd.sample(range(rots_npy.shape[0]),k=lim)
    
    if train:
        # test_rots =rots_npy[0:lim+1,...]
        # train_rots = rots_npy[lim:,...]
        # print(test_rots.shape,train_rots.shape)
        idx = 0
        train_rots = np.zeros((rots_npy.shape[0]-lim,3,3))
        test_rots = np.zeros((lim,3,3))
        for i in range(rots_npy.shape[0]):
            if i not in rng_idx:
                train_rots[idx] = rots_npy[i]
                idx+=1
        idx = 0
        for i in rng_idx:
            test_rots[idx] = rots_npy[i]
            idx+=1  
        np.save(f'{save_file}_train.npy',train_rots)
        np.save(f'{save_file}_test.npy',test_rots)
    else:
        np.save(f'{save_file}.npy',rots_npy)
        
        
    

T = Transformers()



def eulers_from_formula(file):
    def mats_to_euls(file):
        rot_mats = np.load(r"")
        print(rot_mats.shape)
        rot_mats = rot_mats.reshape(-1,3,3)
        eulers = []
        for batch in range(rot_mats.shape[0]):
            eulers.append(T.matrix_to_euler(rot_mats[batch]))
        return eulers

    eulers = mats_to_euls(file)
    eulers = np.vstack(eulers)
    print(eulers.shape)
    np.savetxt('copperT4_formula.txt',eulers,delimiter=' ')


def functioncalls():
   
    # Mix_rot = T.using_scipy(Euler_arrays=r"E:\Programming\NF_growth\train_data\Copper_large\CopperT6_Eulers.txt")

    # eulers_from_formula()
    # rots_to_npy(euler_save_path,inp_rot=Mix_rot,train=True)
    rot_mat_to_euler(Path(r""),euler_save_path)
    return None
euler_save_path = r"" # add desired path to save the rotation data in Euler values


# functioncalls()
class NpzGen:

    def __init__(self):
        pass

    def getfiles(self,work_dir):
        files_path = Path(work_dir)
        files =  [file for file in files_path.iterdir() if file.suffix == '.npy']
        return files

    def gen_euls_npz(self,directory):
        matrices = np.load(directory)
        weights,batch_so3 = zip(*matrices.items())
        
        # breakpoint()
        idx = rd.randint(0,len(weights))
        
        print(weights[1])
        euler_check = r.from_matrix(batch_so3[1]).as_euler('ZXZ',degrees=False)
        np.savetxt(f'euls_check_{int(weights[1])}.txt',euler_check)
        
    def gen_cond_train_test(self,directory,data_type,save=False):
        matrices = np.load(directory)
        weights,batch_so3 = zip(*matrices.items())
        test_so3 = {}
        train_so3 = {}
        lim = int(0.2*len(weights))
        rng_indx = rd.sample(weights,k=lim)
        
        for key,_ in matrices.items():
            if key in rng_indx:
                test_so3[key] = matrices[key]
            else:
                train_so3[key] = matrices[key]
        
        if save == True:
            np.savez(f'{data_type.capitalize()}_conditional_train',**train_so3)
            np.savez(f'{data_type.capitalize()}_conditional_test',**test_so3)

        return None
            
        
        # for so3 in batch_so3:
        #     lim = int(0.2*len(so3))
        #     rng_indx = rd.sample(range(int(so3.shape[0])),k=lim)
            
        #     test_mats = np.zeros((lim,3,3))
        #     train_mats = np.zeros((so3.shape[0]-lim,3,3))
        #     idx = 0
        #     for i in rng_indx:
        #         test_mats[idx] = so3[i]
        #         idx +=1
        #     test_so3.append(test_mats)
        #     idx = 0
        #     for i in range(so3.shape[0]):
        #         if i not in rng_indx:
        #             train_mats[idx] = so3[i]
        #             idx+=1
        #     train_so3.append(train_mats)

        return None

npz = NpzGen()

def gen_eulers_files_call():
    os.makedirs(os.path.join(euler_save_path,'euler_data'),exist_ok=True)
    euler_save_path1 = f'{euler_save_path}\\euler_data'

    rot_mat_dir = r""
    rots_files = npz.getfiles(rot_mat_dir)
    for file in rots_files:
        rot_mat_to_euler(file,euler_save_path1)

def gen_conditional_data(data_type):
    data_path = os.makedirs(os.path.join(r"",f'{data_type}'),exist_ok=True)
    npz.gen_cond_train_test(r"",data_type,save=True)
     
# gen_euls_npz(r'E:\Programming\Python Programs\Thesis_work\Goss_conditional.npz')
functioncalls()
# gen_conditional_data('Copper')
# gen_eulers_files_call()