# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

import os
from  cfgs import *
from data.load_data import DataSet, CIFAR10DataSet
import torch.utils.data as Data
from processes import Process
import torch
import argparse, random

# ------------------------------------

def parse_args():
    '''
       Parse input arguments
    '''

    parser = argparse.ArgumentParser(description='Parallel Squeeze & Excitation Args')

    parser.add_argument('--WITH_SE', dest='WITH_SE',
                      action='store_true',
                      help='use the squeeze & excitation module or not',
                      default=False)

    parser.add_argument('--SE_TYPE', dest='SE_TYPE',
                      choices=['vanilla', 'parallel', 'split', 'parallel_orthogonal', 'squeeze_recur'],
                      help='choose the type of SE block you wanna have in the network',
                      default='vanilla', type=str)

    parser.add_argument('--GPU', dest='GPU',
                      choices=['0', '1', '2', '3', '4', '5', '6', '7'],
                      help='choose the gpus to be used',
                      default='0', type=str)

    parser.add_argument('--SE_STRIDE', dest='SE_STRIDE',
                      choices=[2,3,4,5],
                      help='2,3,4,5',
                      default=2, type=int)

    parser.add_argument('--ORTHOGONAL', dest='ORTHOGONAL',
                      help='what is the method of orthogonal regularization if any',
                      default='none', type=str)

    parser.add_argument('--ORTH_WEIGHT', dest='ORTH_WEIGHT',
                      help='orthogonal loss weight',
                      default=0.1, type=float)

    parser.add_argument('--CUSTOM_SE_BLOCKS', dest='CUSTOM_SE_BLOCKS',
                      help='specifies which blocks in Resnet get the custom squeeze & excite block',
                      default='0,1,2,3,4', type=str)

    parser.add_argument('--RESNET_TYPE', dest='RESNET_TYPE',
                      help='type of resnet',
                      default='34', type=str)

    parser.add_argument('--N_LAYERS', dest='N_LAYERS',
                      choices=[3,5,7,9,18],
                      help='3,5,7,9,18',
                      default=5, type=int)

    return parser.parse_args()

# ------------------------------------

def main():

    # argument parser which takes input arguments & parses them
    _g_params = Cfgs()
    args = parse_args()
    args_dict = _g_params.parse_to_dict(args)

    _g_params.add_args(args_dict)
    print('Hyper Parameters:')
    print(_g_params)

    random.seed()

    # Create CIFAR10 train & test datasets
    dataset_train = CIFAR10DataSet(_g_params.CIFAR10_PATH)
    dataset_test = CIFAR10DataSet(_g_params.CIFAR10_PATH, train=False)

    # create a process object
    process = Process(_g_params, dataset_train, dataset_test)

    process.train()

# ------------------------------------

main()
