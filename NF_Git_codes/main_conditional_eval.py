import torch
import os
from trainer import get_trainer
from config import Config
import numpy as np
from data_loader_conditional import get_conditional_data
from torch.nn import DataParallel
from tqdm import tqdm
from tools import viz
import time
from datetime import timedelta

def eval():
    config = Config()

    trainer = get_trainer(config)
    
    trainer.load_states(config.test_states)
    if config.cond_eval == 'sample':
        get_samples(config,trainer)
    if config.cond_eval == 'nll':
        evaluation(config,trainer)
        

def get_samples(config,trainer):

    random_rotations = viz.generate_queries(config.num_queries,mode='random')
    random_rotations = random_rotations.cuda()
    
    if isinstance(trainer.flow,DataParallel):
        flow = trainer.flow.module.cuda()
    else:
        flow = trainer.flow.cuda()
    flow.eval()
    condition = viz.get_condition(config,random_rotations.shape[0])
    generated_rotations,ldjs = flow.inverse(random_rotations,condition)
    generated_rotations = generated_rotations.cpu()
     #Rotation matrices Save
    os.makedirs(f'{config.project_dir}/Generated_conditional_data',exist_ok=True)

    rot_write=generated_rotations.detach().numpy().reshape(-1,9)
    np.save(f'Generated_conditional_data/{config.data}_NF_rotations_cond_{config.weight}.npy',rot_write)

def evaluation(config,trainer):
    test_loader = get_conditional_data('test',config)
    
    testbar = tqdm(test_loader)
    test_loss = []
    for i,data in enumerate(testbar):            
        result_dict = trainer.val_func(data)
        loss = result_dict['loss']
        test_loss.append(result_dict['losses'].cpu())

    test_loss = np.concatenate(test_loss)
    test_mean = np.mean(test_loss)
    print(f'{test_mean} is the evaluated log likelihood')



if __name__ == '__main__':
    with torch.no_grad():
        start = time.perf_counter()
        eval()
        total = timedelta(seconds=time.perf_counter() - start)
        print(f'total time: {total}')
