# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

import os
import numpy as np
import glob, json, torch, time, random
import torch.utils.data as Data
import torch
import pickle
from PIL import Image

# ------------------------------
# ---------- Functions ---------
# ------------------------------

def create_one_hot(lbl, name2id, num_classes):
    '''
       create a one-hot vector for a given label
       input: 
             lbl: label or class
             name2id: dictionary that maps a given label to an index
             num_classes: number of classes or size of the one-hot vector
    '''

    lbl_indx = name2id[lbl]

    lbl_one_hot = torch.zeros(num_classes, dtype=torch.long)

    lbl_one_hot[lbl_indx] = 1

    return lbl_one_hot

def pad4(img):
    '''
       pads input tensor with 4 zeros on the sides
    '''
    return np.pad(img, ((0, 0), (4, 4), (4, 4)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
