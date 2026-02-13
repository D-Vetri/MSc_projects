import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from flow import flow
import utils
import os
import numpy as np
from tools import viz
import random as rd

def get_trainer(config):
    return Trainer(config)

class Trainer:
    def __init__(self,config):
        self.config = config
        
        self.flow = flow(config)
        self.flow = DataParallel(self.flow)# to split the module across devices.
        # Have to test for how this affects for smaller batches of data

        self.optimizer_flow = optim.Adam(self.flow.parameters(),config.lr)      
        self.clock = utils.TrainClock(schedulers=None)
        # we are not implementing multi step learning rate. The lr is constant 
        # for a training cycle

        #embedding not implemented yet - 27/12/24        
    
    def forward(self, data):
        in_data = data.get('rot_mat') #(batch_size* rotation matrix(here 3*3))        
                                      # The in_data is a list of N,3,3  if conditional
        if self.config.condition:
            condition_vector = data.get('condition')
            feature = torch.empty(0,3,dtype=torch.float)
            for i in range(len(in_data)):
                feature = torch.cat((feature,condition_vector[i].expand(in_data[i].shape[0],3)),dim=0)
            feature = feature.cuda()   
            in_data = torch.cat(in_data,dim=0)
            
        else:
            feature = None   
        in_data = in_data.cuda()
     
        
        flow = self.flow.cuda()
        
        rotations,ldjs = flow(in_data,feature=feature)
        losses_nll = -ldjs

        loss = losses_nll.mean()
        result_dict = dict(
            loss = loss,
            losses_nll = losses_nll,
            rotations = rotations,
            probs = ldjs,
        )
        
        return result_dict
    
    def train_func(self,data):
        
        net_optimizers = self.train()
        result_dict = self.forward(data)
        loss = result_dict['loss']

        for optimizers in net_optimizers:
            optimizers[1].zero_grad()

        loss.backward()

        for optimizers in net_optimizers:
            optimizers[1].step()
        
        return result_dict



    def train(self):
        net_optimizer = []
        self.flow.train()
        net_optimizer.append((self.flow,self.optimizer_flow)) 
        return net_optimizer
        
    def val_func(self,data):
        self.flow.eval()
        if isinstance(self.flow,DataParallel):
            flow = self.flow.module
        else:
            flow = self.flow
        

        with torch.no_grad():
            loss_dict = self.eval_nll(flow,data,feature=None)
        
        return loss_dict
    
    def eval_nll(self,flow,data,feature=None):
        in_data = data.get('rot_mat') #rotation matrices. A list of N,3,3 matrices if
                                      # conditional.
        
        if self.config.condition:
            condition_vector = data.get('condition')
            feature = torch.empty(0,3,dtype=torch.float)
            for i in range(len(in_data)):
                feature = torch.cat((feature,condition_vector[i].expand(in_data[i].shape[0],3)),dim=0)
            feature = feature.cuda()
            in_data = torch.cat(in_data,dim=0)
        in_data = in_data.cuda() 
        
        rotation,eval_ldjs = flow(in_data,feature)
        losses_ll = eval_ldjs
        loss_dict = dict(
            loss = losses_ll.mean(),
            losses= losses_ll,
            rotations = rotation,
            eval_probs = eval_ldjs

        )

        return loss_dict

    def save_state(self,name=None):
        flow_state = self.flow.module.cpu().state_dict()
        save_dict = {
            "clock": self.clock.make_checkpoint(),
            "flow_states": flow_state,
            "flow_Optimizer_states": self.optimizer_flow.state_dict(),
        }
        if name is None:
            save_path = os.path.join(self.config.state_dir,
                                     "save_point_iteration{}.pth".format(self.clock.iteration))
            print(f'saving states at iteration{self.clock.iteration}')

        else:
            save_path = os.path.join(self.config.state_dir,f'{name}.pth')
            print(f'Saving states at iteration {self.clock.iteration}')

        torch.save(save_dict,save_path)
        self.flow.cuda()

    def load_states(self,name=None):
        '''
        Loads the module states. Acts as either checkpoint to continue training
        (this continuing aspect in not currently included(21/11/24), it also serves as the pth 
        file needed for testing and or post-training usage.
        '''
        if os.path.isabs(name):
            load_path = name
        else:
            try:
                load_path = os.path.join(
                    self.config.state_dir,f'{name}.pth'
                )
            except:
                load_path = name
        print(load_path)
        if not os.path.exists(load_path):
            raise ValueError(f'Checkpoint path {load_path} does not exist')
        
        print(f"the saved States path is {load_path}")
        checkpoint = torch.load(load_path,map_location=torch.device('cpu'))
        
        print(f"Loading states from {load_path}")

        self.flow.module.load_state_dict(checkpoint['flow_states'])
        
        self.flow.cuda()
        self.optimizer_flow = optim.Adam(
            self.flow.parameters(),self.config.lr)
        
        self.optimizer_flow.load_state_dict(checkpoint['flow_Optimizer_states'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    def save_val_rots(self,data_save):
        cat_tensors = torch.cat(data_save,dim=0)
        cat_tensors = cat_tensors.cpu()
        tens_arrays = cat_tensors.detach().numpy().reshape(-1,9)
        os.makedirs(f'{self.config.project_dir}/Test_rot_mats',exist_ok=True)
        np.save(f'Test_rot_mats/Test_mats_iternation{self.clock.iteration}.npy',tens_arrays)

    def save_train_rots(self,train_save,inverse_viz =False,frob_norm=False):
        cat_tensors = torch.cat(train_save,dim=0)
        if inverse_viz:
            self.inverse_during_training(cat_tensors)
        if frob_norm:
            norms = self.states_compare(cat_tensors,Viz_memory=False)
        
        
        
       
        if frob_norm:
            return norms
   
    def inverse_during_training(self,data):
        
        if isinstance(self.flow,DataParallel):
            flow = self.flow.module
        else:
            flow = self.flow
        self.flow.eval()
        with torch.no_grad():
            
            # fig1 = viz.so3_sample_viz(data.cpu(),torch.ones(data.shape[0]))
            # os.makedirs(f'{self.config.project_dir}/During_training_data_Viz',exist_ok=True)
            # fig1.savefig(f'During_training_data_Viz/train_Viz_at_{self.clock.iteration}.png')
            
            
            input_rotations = viz.generate_queries(2000,mode='random')
            input_rotations = input_rotations.cuda()
            rotations,probs = flow.inverse(input_rotations,feature=None)
            rotations,probs = rotations.cpu(),-probs.cpu()
            fig = viz.so3_sample_viz(rotations,probs)
            
            tens_arrays = rotations.detach().numpy().reshape(-1,9)

            os.makedirs(f'{self.config.project_dir}/Train_rot_mats',exist_ok=True)
            np.save(f'Train_rot_mats/Train_mats_epoch{self.clock.epoch}.npy',tens_arrays)
            
            if self.config.train_invert % self.config.inv_viz_freq == 0:
                os.makedirs(f'{self.config.project_dir}/Inverted_sample_data_Viz',exist_ok=True)
                fig.savefig(f'Inverted_sample_data_Viz/Viz_at_{self.clock.iteration}.png')

    def states_compare(self,data,query_size=None,Viz_memory=False):
        GT_data = data
        if  query_size == None:
            query_nums = GT_data.size(dim=0)

        else:
            query_nums = query_size

        self.flow.eval()
        flow2 = self.flow.module
        
        
        if Viz_memory:
            with torch.no_grad():
                mem_queries = viz.generate_queries(query_nums,mode='random')
                mem_queries = mem_queries.cuda()

                rotations,ldjs = flow2.inverse(mem_queries,feature=None)
                rotations,probs = rotations.cpu(),-ldjs.cpu()
                
                fig1 = viz.so3_sample_viz(rotations,probs)
                os.makedirs(f'{self.config.project_dir}/Memory_model_states',exist_ok=True)
                fig1.savefig(f'Memory_model_states/Memory_sample_data_viz{self.clock.iteration}.png')
            # simultaneous comparison with loaded states
            
            with torch.no_grad():
                flow2.eval()
                load_path = f"{self.config.state_dir}/save_point_iteration{self.clock.iteration}.pth"
                save_states = torch.load(load_path,map_location='cpu')
                flow2.load_state_dict(save_states['flow_states'])
                flow2.cuda()
            

                mem_queries2 = viz.generate_queries(query_nums,mode='random')
                mem_queries2 = mem_queries2.cuda()

                rot2,probs2 = flow2.inverse(mem_queries2,feature=None)
                rot2,probs2 = rot2.cpu(),-probs2.cpu()

                fig1 = viz.so3_sample_viz(rotations,probs)
                os.makedirs(f'{self.config.project_dir}/Loaded_model_states',exist_ok=True)
                fig1.savefig(f'Loaded_model_states/Loaded_sample_data_viz{self.clock.iteration}.png')

                
                diff_rot = rot2 - GT_data.detach().cpu()
                fro_norm = torch.linalg.norm(diff_rot,dim=(-2,-1))
                fro_norm = fro_norm.mean()
        
        else:
            with torch.no_grad():
                flow2.eval()
                
                rng_index = rd.sample(range(GT_data.size(dim=0)),k=100)
                GT_rng = []
                for i in rng_index:
                    GT_rng.append(GT_data[i,...])
                GT_rng = torch.cat(GT_rng).reshape(-1,GT_data.size(dim=-2),GT_data.size(dim=-1))

                # load_path = f"{self.config.state_dir}/save_point_iteration{self.clock.iteration}.pth"
                # save_states = torch.load(load_path,map_location='cpu')
                # flow2.load_state_dict(save_states['flow_states'])
                # flow2.cuda()
                

                for_rot,prob2 = flow2.forward(GT_rng,feature=None)
                rot2,prob2 = flow2.inverse(for_rot,feature=None)
                rot2 = rot2.cpu()




        
                diff_rot = rot2 - GT_rng.detach().cpu()
                fro_norm = torch.linalg.matrix_norm(diff_rot,ord='fro',dim=(-2,-1))
                fro_norm = fro_norm.mean()
    
       
        return fro_norm