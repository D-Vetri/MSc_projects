import os
import configargparse as cp

class Config(object):
    def __init__(self):
        parser,args = self.parse()

        #setting all the collected arguments as attributes of the config class to be used
        for k,v in sorted(args.__dict__.items()):
            self.__setattr__(k,v)

        #getting the working directory
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        print('The Running directory is:',self.project_dir)
        
        
        state_dir = os.path.join(self.project_dir,'model_states')
        os.makedirs(state_dir,exist_ok=True)
        self.state_dir = state_dir
        
    
    def parse(self):
        parser = cp.ArgumentParser(
            default_config_files=['config_file.yml']
        )
        parser.add_argument('--config', is_config_file=True,
                            help='Config file path')
        self._flow_config_parse(parser)
        self._basic_config(parser)
        self._viz_evol(parser)
        self._conditional(parser)

        args = parser.parse_args()

        return parser,args
    
    def _basic_config(self,parser):
        group = parser.add_argument_group('basic')
        group.add_argument('--seed',type=int,
                           help='Random seed-Default=42')
        group.add_argument('--condition',type=int,default=0,
                           help='set condition to 1 for conditional training')
        group.add_argument('--lr',type=float,default=1e-4,
                           help='The learning rate')
        group.add_argument('--batch_size',type=int,default=128,
                           help='Batch size for training and testing')
        group.add_argument('--eval_frequency',type=int,
                           help='Run evaluation for every n iterations')
        group.add_argument('--max_iter',type=int,default=5000,
                           help='maximum number of training iterations')
        group.add_argument('--eval', type=str, default='sample',
                           help="enter \'sample\' or \'prob\' or \'nll\' for evaluation" )
        group.add_argument('--num_queries',type=int,default=500,
                           help='The number of samples for generation')
        group.add_argument('--scatter_size',type=float,default=1e1,
                           help='scatter size for points in the data plot')
        group.add_argument('--D',type=int,default=3,
                           help='The Spatial dimension. In this project, on D=3 is implemented')
        group.add_argument('--data',type=str,default = 'Goss',
                           help = 'The data folder used for training.')
        group.add_argument('--save_frequency',type=int, default=1000,
                           help='The number of iterations needed to save model states')
        group.add_argument('--test_states',type=str,
                           help = 'The file path of the pth file containing trained model parameters')
        group.add_argument('--max_epoch',type=int, default = 1000,
                           help = 'Max epoch.One epoch is total_data_points/batch_size')
    
    def _flow_config_parse(self,parser):
        group = parser.add_argument_group('flow')

        group.add_argument('--layers',type=int, default=21,
                           help='number of stacked layers-default=21')
        group.add_argument('--segments',type=int,default=64,
                           help='number of combinations of Mobius coupling-default=64')
        
        group.add_argument('--rot',type=str,default='uncon16',
                           help='unconditional rotations or conditional rotations')
        group.add_argument('--feature_dim',type=int,default=0,
                           help='number of features on the conditions')
       
        return group

    def _viz_evol(self,parser):
        group = parser.add_argument_group('viz_evolve')

        group.add_argument('--viz_evolve',type=str,default=False,
                           help='Set to True to visualize training evolution')
        group.add_argument('--evolve_frequency',type=int,default = 1000,
                           help ='Training iteration frequency to visualize evolution')
        group.add_argument('--inv_viz',type=bool,default=False,
                           help='Set to True: training visualization in inverse direction')
        group.add_argument('--gen_query_viz',type=bool,default=False,
                           help='Set to True: Random distribution sample in training visualization')
        group.add_argument('--rot_save',type=bool,default=True,
                           help='Set to True: Rotation matrices saved as csv file')
        group.add_argument('--train_invert',type=int,default=50,
                           help='Epoch frequency to invert learned rotations')
        group.add_argument('--inv_viz_freq',type=int,default=1000,
                           help='The frequency for visualizing inverse direction along training')
    
    def _conditional(self,parser):
        group = parser.add_argument_group('conditional')

        group.add_argument('--weight',type=int,default=None,
                           help='Enter the condition: weights')
        group.add_argument('--cond_eval',type=str,default = 'sample',
                           help='Enter evaluation mode: 1)sample')
        group.add_argument('--cond_batch_size',type=int,default =10,
                           help = 'Batch size for conditional data')