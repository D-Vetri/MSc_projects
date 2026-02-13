'''
Main script
'''
import torch
import numpy as np
from tqdm import tqdm
from config import Config
from trainer import get_trainer
from data_loader import get_dataloader_rot
from  tools.viz import evol_plot,generate_queries,plot_loss,plot_norm
import time
from datetime import  timedelta
import os
import logging
import warnings
import traceback
warnings.simplefilter(action='ignore', category=FutureWarning)

# setting up logger for error capture

logging.basicConfig(filename='train_log.txt', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)



def main():
    try:
        config = Config()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        #dataloading
        train_loader = get_dataloader_rot('train',config)
        test_loader = get_dataloader_rot('test',config)
        test_iter = iter(test_loader)

        #creating the trainer for training
        trainer = get_trainer(config)
        val_rotations = []
        losses = []
        Frob_norms = []
        test_loss_mean = []
        loss_plots = {'train_loss':[],'test_loss':[]}
        clock = trainer.clock
        # breakpoint()  

        starttime = time.perf_counter()
        logging.info(time.time())
        
        
        while True:
            train_rotations=[]
            pbar = tqdm(train_loader)
            for _, data in enumerate(pbar):
                #training steps
                train_result_dict = trainer.train_func(data)
                loss = train_result_dict["loss"]
                train_rotations.append(train_result_dict['rotations'])
                losses.append(loss.cpu())
                


                pbar.set_description("Epoch[{}][{}]".format(
                    clock.epoch, clock.minibatch))
                
                pbar.set_postfix({"loss":loss.item()})
                

                clock.tick()
                

                if clock.iteration % config.eval_frequency == 0 or clock.iteration == config.max_iter:
                    test_loss = []
                    
                    for i in range(10):
                        try:
                            data = next(test_iter)
                        except:
                            test_iter = iter(test_loader)
                            data = next(test_iter)
                        result_dict = trainer.val_func(data)
                        loss = result_dict["loss"]
                        test_loss.append(result_dict["losses"].cpu())
                        val_rotations.append(result_dict['rotations'])
                    
                    test_loss = np.concatenate(test_loss,0)
                    # trainer.save_val_rots(val_rotations)
                    test_mean = np.mean(test_loss)
                    logging.info(f'Evaluation loss at {clock.iteration}: {test_mean}')
                    
                    test_loss_mean.append(test_mean)
                    
                    
                if config.viz_evolve == True:
                    if clock.iteration % config.evolve_frequency == 0:  
                        viz_rotations = train_result_dict['rotations'].cpu()
                        viz_prob = train_result_dict['probs'].cpu()
                        evol_plot(viz_rotations,viz_prob,clock.iteration)
                        # logging.info('evol_plot_saved')

                       
                        

                if config.rot_save == False:    
                    if clock.iteration % config.save_frequency == 0:
                        trainer.save_state()
                
                # if clock.iteration % config.train_invert == 0:
                #     trainer.save_state()
                #     fro_norms = trainer.save_train_rots(train_rotations,inverse_viz=True,frob_norm=False)
                #     # Frob_norms.append(trainer.states_compare(torch.cat(train_rotations).reshape(-1,3,3),Viz_memory=False))
                #     if fro_norms is not None:
                #         Frob_norms.append(fro_norms)
                    

            clock.tock()
            
            if config.rot_save == True:
                if clock.epoch % config.train_invert == 0:
                    if  clock.epoch <= 500:
                        trainer.save_state()
                        fro_norms = trainer.save_train_rots(train_rotations,inverse_viz=True,frob_norm=False)
                        # Frob_norms.append(trainer.states_compare(torch.cat(train_rotations).reshape(-1,3,3),Viz_memory=False))
                        if fro_norms is not None:
                            Frob_norms.append(fro_norms)
           
                
                
              

            if clock.iteration > config.max_iter or clock.epoch > config.max_epoch:

                logging.info(f'iteration {config.max_iter} reached')
                # logging.info(f"Total train time: {pbar.format_dict['elapsed']:.2f}") 
                logging.info(f"Total Training Time:{timedelta(seconds=time.perf_counter()-starttime)}")
                trainer.save_state()
                trainer.save_train_rots(train_rotations)
                plot_loss(losses,clock.iteration,config,test_loss_mean)
                logging.info(len(Frob_norms))
                if Frob_norms:
                    plot_norm(Frob_norms,config)


                break

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc()) 

if __name__ == "__main__":
    main()








    

