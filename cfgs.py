# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

import os, torch, random
import numpy as np
from types import MethodType

class PATH:

    def __init__(self):

        # dataset root path
        self.DATASET_PATH = 'dataset/mnist/'

        self.LOG_DIR = "output/logs"
        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)

        self.LOG_STATS_DIR = "output/stats_logs"
        if not os.path.exists(self.LOG_STATS_DIR):
            os.mkdir(self.LOG_STATS_DIR)

        self.CKPT_PATH = "output/ckpts"
        if not os.path.exists(self.CKPT_PATH):
            os.mkdir(self.CKPT_PATH)

        self.init_path()

    def init_path(self):

        self.IMAGE_DATA_PATH = {
            'train': self.DATASET_PATH + 'train/',
            'val': self.DATASET_PATH + 'val/',
            'test': self.DATASET_PATH + 'test/',
        }

        self.CIFAR10_PATH = "dataset/cifar10"
        self.CIFAR10_MID_PATH = "dataset/cifar10_outputs"
        if not os.path.exists(self.CIFAR10_MID_PATH):
            os.mkdir( self.CIFAR10_MID_PATH )

class Cfgs(PATH):

    def __init__(self):
        super(Cfgs, self).__init__()

        # Set Devices
        # If use multi-gpu training, set e.g.'0, 1, 2' instead
        self.GPU = '1'

        # Print loss every step
        self.VERBOSE = True

        # Use pin memory
        self.PIN_MEM = True

        # {'train', 'val', 'test'}
        self.SPLIT = 'train'

        # Multi-thread I/O
        self.NUM_WORKERS = 8

        # Set 'external': use external shuffle method to implement training shuffle
        # Set 'internal': use pytorch dataloader default shuffle method
        self.SHUFFLE_MODE = 'external'

        # Resnet Type
        self.RESNET_TYPE = "34"

        # number of layers in each block when using the simple resnet model for Cifar10
        self.N_LAYERS = 5

        # has squeeze & excitation blocks
        self.WITH_SE = True

        # type of SE block (vanilla, parallel, parallel_orthogonal). Default is vanilla.
        self.SE_TYPE = 'vanilla'

        # specifies which blocks in Resnet would have a custom squeeze & excite block
        self.CUSTOM_SE_BLOCKS = '0,1,2,3,4'

        # SE stride size, default is 2 (not used for vanilla SE)
        self.SE_STRIDE = 2

        # reduction ratio (for the squeeze & excitation block)
        self.REDUCTION_RATIO = 16

        # orthogonal loss weight for orthogonality between SE weights
        self.ORTH_WEIGHT = 0.1
        self.ORTHOGONAL  = "none"

        # Model hidden size
        # (512 as default, bigger will be a sharp increase of gpu memory usage)
        self.HIDDEN_SIZE = 512

        # Dropout rate for all dropout layers
        # (dropout can prevent overfitting： [Dropout: a simple way to prevent neural networks from overfitting])
        self.DROPOUT_R = 0.1

        # --------------------------
        # ---- Optimizer Params ----
        # --------------------------

        # The base learning rate
        self.BATCH_SIZE = 128

        # The base learning rate
        self.LR_BASE = 0.1

        # Learning rate decay ratio
        self.LR_DECAY_R = 0.2

        # Learning rate decay at {x, y, z...} epoch
        self.LR_DECAY_LIST = [100, 150]

        # Max training epoch
        self.MAX_EPOCH = 200

        # Adam optimizer betas and eps
        self.OPT_BETAS = (0.9, 0.98)
        self.OPT_EPS = 1e-9

        # ------------ Devices setup
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)

    def parse_to_dict(self, args):
        '''
           puts given arguments into a dictionary with their values
           input: args is given as input by the user
        '''
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType) and getattr(args, arg) is not None:
                args_dict[arg] = getattr(args, arg)

        return args_dict

    def add_args(self, args_dict):
        '''
           populates attributes of this object with the given argument dictionary
        '''
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def __str__(self):
        '''
           prints attributes of this object and discards methods when printing
        '''
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''

