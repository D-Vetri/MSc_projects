
from tqdm import tqdm
import numpy as np
import torch 
from config import Config
import random
from data_loader_conditional import get_conditional_data
from tools.viz import plot_loss
from trainer import get_trainer
import time
from datetime import  timedelta
import os
import logging
import warnings
import traceback
warnings.simplefilter(action='ignore', category=FutureWarning)

# setting up logger for error capture

logging.basicConfig(filename='conditional_train_log.txt', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)


def exp_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

try:
    def main():
        ''''
        main function running on conditional data
        '''
        config = Config()
        exp_seeds(config.seed)
        
        train_loader = get_conditional_data('train',config)
        test_loader = get_conditional_data('test',config)
        test_iter =iter(test_loader)

        trainer = get_trainer(config)
        losses = []
        
        test_loss_mean = []
        
        clock = trainer.clock
        starttime = time.perf_counter()
        logging.info(time.time())
        
        while True:
            pbar = tqdm(train_loader)
            for _,data in enumerate(pbar):
                  
                result_dict = trainer.train_func(data)
                
                loss = result_dict['loss']
                losses.append(loss.cpu())


                pbar.set_description("EPOCH[{}][{}]".format(
                            clock.epoch, clock.minibatch))
                pbar.set_postfix({"loss":loss.item()})
                
                if clock.iteration % config.eval_frequency == 0:
                    test_loss = []
                    for i in range(10):
                        try:
                                data = next(test_iter)
                        except:
                                test_iter = iter(test_loader)
                                data = next(test_iter)
                            
                        result_dict = trainer.val_func(data)
                        loss = result_dict['loss']
                        test_loss.append(result_dict['losses'].cpu())
                    
                    test_loss = np.concatenate(test_loss,0)
                    test_mean = np.mean(test_loss)
                    logging.info(f'Evaluation loss at {clock.iteration}: {test_mean}')

                    test_loss_mean.append(test_mean)



                clock.tick()

                  
                if clock.iteration % config.save_frequency == 0:
                    trainer.save_state()

            clock.tock()
            
            if clock.iteration > config.max_iter:
                logging.info(f'iteration {config.max_iter} reached')
                # logging.info(f"Total train time: {pbar.format_dict['elapsed']:.2f}") 
                logging.info(f"Total Training Time:{timedelta(seconds=time.perf_counter()-starttime)}")
                plot_loss(losses,clock.iteration,config,test_loss_mean)
                break

except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc()) 

if __name__ == '__main__':
     main()
    

