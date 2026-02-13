import numpy as np
import torch
from config import Config
from data_loader import get_dataloader_rot
from torch.nn import DataParallel
from tools import viz
from trainer import get_trainer
from tqdm import tqdm
import os
import pandas as pd
import csv
import time
from datetime import timedelta

def eval():
    config = Config()
    
    trainer = get_trainer(config)
    trainer.load_states(config.test_states)
    if config.eval == 'prob':
        pdf_vis(config,trainer)
    elif config.eval == 'sample':
        sample_vis(config,trainer)
    elif config.eval == 'nll':
        evaluation(config,trainer)

def evaluation(config,trainer):
    test_loader = get_dataloader_rot('test',config)
    testbar = tqdm(test_loader)

    test_loss = []
    for i, data in enumerate(testbar):
        result_dict = trainer.val_func(data)
        loss = result_dict['loss']
        test_loss.append(result_dict['losses'].cpu())

    test_loss = np.concatenate(test_loss) # the individually appended tensors are combined
    test_loss = np.mean(test_loss)
    print(test_loss)

def sample_vis(config, trainer):
    random_rotations = viz.generate_queries(config.num_queries,mode='random')
    random_rotations = random_rotations.cuda()
    if isinstance(trainer.flow,DataParallel):
        flow = trainer.flow.module.cuda()
    else:
        flow = trainer.flow.cuda()
    flow.eval()
    inverted_rotations,ldjs = flow.inverse(random_rotations)
    inverted_rotations = inverted_rotations.cpu()
    log_prob = -ldjs.cpu()
    prob = torch.exp(ldjs)
    norm = prob.norm()

    fig = viz.so3_sample_viz(inverted_rotations,
                             log_prob,
                            #  ax=None,
                            #  fig = None,
                             display_threshold_probability=0,
                             scatter_size=config.scatter_size,
                             )
    os.makedirs('plot/raw',exist_ok=True)
    save_path = f'plot/raw/{config.eval}.png'
    print("Saving plot to",save_path)
    fig.show()
    fig.savefig(save_path)
    
    #Rotation matrices Save
    os.makedirs(f'{config.project_dir}/Generated_Rotation_data',exist_ok=True)

    rot_write=inverted_rotations.detach().numpy().reshape(-1,9)
    # with open(f'Generated_Rotation_data/Sampled_rotation_matrices.csv','w+') as f:
    #     pen = csv.writer(f)        
    #     pen.writerows(rot_write)
    # rot_write2=inverted_rotations.detach().numpy()#.reshape(-1,9)
    # with open(f'Generated_Rotation_data/Sampled_rotation_matrices2.csv','w+') as f:
    #     pen = csv.writer(f)        
    #     pen.writerows(rot_write2)
    np.save(f'Generated_Rotation_data/{config.data}_NF_rotations.npy',rot_write)
        



def pdf_vis(config, trainer):
    inp_rotations = viz.generate_queries(config.num_queries,mode='random')
    inp_rotations = inp_rotations.cuda()
    if isinstance(trainer.flow,DataParallel):
        flow = trainer.flow.module.cuda()
    else:
        flow = trainer.flow.cuda()
    
    flow.eval()
    rotations,ldjs = flow.inverse(inp_rotations)
    rotations = rotations.cpu()
    log_prob = -ldjs.cpu()
    prob = torch.exp(ldjs)
    norm = prob.norm()

    fig = viz.so3_sample_viz(rotations,
                             log_prob,
                             ax=None,
                             fig = None,
                             display_threshold_probability=0,
                             scatter_size=config.scatter_size,
                             )
    os.makedirs('plot/raw',exist_ok=True)
    save_path = f'plot/raw/{config.eval}.png'
    print("Saving plot to",save_path)
    fig.show()
    fig.savefig(save_path)
    

if __name__ == '__main__':
    with torch.no_grad():
        start = time.perf_counter()
        eval()
        
        total = timedelta(seconds=time.perf_counter() - start)
        print(f'total time: {total}')
    


