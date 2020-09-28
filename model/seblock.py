# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

from model.net_utils import global_average_pooling, create_pads
import torch.nn as nn
import torch.nn.functional as F
import torch

# -------------------------------------------
# -------- Function to pick SE block --------
# -------------------------------------------

def pick_se_block(_g_params, custom_se_block, output_ch):
    '''
       This function picks one of the SE block types to be assigned to the network
       input:
             output_ch: is the number of channels in the output tensor
             custom_se_block: specifies if the block is a custome se-block or vanilla se-block
    '''

    # if WITH_SE is false the self.se_block is identity (No squeeze & excitation block added)
    se_block = nn.Sequential()

    if _g_params and _g_params.WITH_SE: 
        
        # the original squeeze & excitation block from the paper (CVPR 2018)
        if _g_params.SE_TYPE=='vanilla' or (not custom_se_block):
            se_block = nn.Sequential(
                SE_BLOCK(output_ch, _g_params.REDUCTION_RATIO, _g_params.BATCH_SIZE)
            )
        # parallel squeeze & excitation
        elif _g_params.SE_TYPE=='parallel':
            se_block = nn.Sequential(
                PARALLEL_SE_BLOCK(output_ch, _g_params.REDUCTION_RATIO, _g_params.BATCH_SIZE)
            )
        elif _g_params.SE_TYPE=='split':
            se_block = nn.Sequential(
                SPLIT_SE_BLOCK(output_ch, _g_params.REDUCTION_RATIO, _g_params.BATCH_SIZE, stride=_g_params.SE_STRIDE)
            )
        elif _g_params.SE_TYPE=='squeeze_recur':
            se_block = nn.Sequential(
                SR_BLOCK(output_ch, _g_params.BATCH_SIZE)
            )

    return se_block

# -------------------------------------------
# ------ Vanilla Squeeze & Excitation -------
# -------------------------------------------

class SE_BLOCK(nn.Module):

    def __init__(self, channel_size, r, batch_size):

        super(SE_BLOCK, self).__init__()

        self.channel_size = channel_size
        self.batch_size = batch_size

        intermediate_size = self.channel_size // r

        self.fc1 = nn.Linear( self.channel_size, intermediate_size) 
        self.fc2 = nn.Linear( intermediate_size, self.channel_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map):

        # global average pooling
        gap_output = global_average_pooling(feature_map)

        # first fully-connected layer
        fc1_output = self.fc1(gap_output)

        # apply relu
        relu_output = self.relu(fc1_output)

        # second fully-connected layer
        fc2_output= self.fc2(relu_output)

        # apply sigmoid
        weights = torch.sigmoid(fc2_output)
        self.weights = weights

        # scale
        scale_output = torch.mul(feature_map, weights.view(self.batch_size, self.channel_size, 1, 1))

        return scale_output

# -------------------------------------------
# ------ Parallel Squeeze & Excitation ------
# -------------------------------------------

class PARALLEL_SE_BLOCK(nn.Module):
    '''
       Four parallel squeeze & exciation blocks are applied to the input feature map
       given an input feature map each pixel would go into 1 of 4 categories & they would be applied a seprate SE Block as below.
       after applying SE block the results would be merged back
       neighbouring pixels would not go in the same feature map in terms of applying SE blocks.
    '''

    def __init__(self, channel_size, r, batch_size):

        super(PARALLEL_SE_BLOCK, self).__init__()

        # pooling in four different ways: [[1,0],[0,0]], [[0,1],[0,0]], [[0,0],[1,0]], [[0,0],[0,1]]
        self.fixed_pool1 = nn.MaxPool2d(1, stride=2, return_indices=True)
        self.fixed_pool2 = nn.MaxPool2d(1, stride=2, return_indices=True)
        self.fixed_pool3 = nn.MaxPool2d(1, stride=2, return_indices=True)
        self.fixed_pool4 = nn.MaxPool2d(1, stride=2, return_indices=True)

        self.se_block1 = SE_BLOCK(channel_size, r, batch_size)
        self.se_block2 = SE_BLOCK(channel_size, r, batch_size)
        self.se_block3 = SE_BLOCK(channel_size, r, batch_size)
        self.se_block4 = SE_BLOCK(channel_size, r, batch_size)

        self.fixed_unpool1 = nn.MaxUnpool2d(1, stride=2)
        self.fixed_unpool2 = nn.MaxUnpool2d(1, stride=2)
        self.fixed_unpool3 = nn.MaxUnpool2d(1, stride=2)
        self.fixed_unpool4 = nn.MaxUnpool2d(1, stride=2)
        
    def forward(self, feature_map):

        f1 = feature_map[:,:,1:,:]
        f2 = feature_map[:,:,:,1:]
        f3 = feature_map[:,:,1:,1:]

        pool1,indices1 = self.fixed_pool1(feature_map)
        pool2,indices2 = self.fixed_pool2(f1)
        pool3,indices3 = self.fixed_pool3(f2)
        pool4,indices4 = self.fixed_pool4(f3)

        se1 = self.se_block1(pool1)
        se2 = self.se_block2(pool2)
        se3 = self.se_block3(pool3)
        se4 = self.se_block4(pool4)

        unpool1 = self.fixed_unpool1(se1, indices1, output_size=feature_map.size())
        unpool2 = F.pad(self.fixed_unpool2(se2, indices2, output_size=f1.size()), (0,0,1,0), "constant", 0)
        unpool3 = F.pad(self.fixed_unpool3(se3, indices3, output_size=f2.size()), (1,0,0,0), "constant", 0)
        unpool4 = F.pad(self.fixed_unpool4(se4, indices4, output_size=f3.size()), (1,0,1,0), "constant", 0)

        parallel_out = unpool1 + unpool2 + unpool3 + unpool4

        return parallel_out

    def orthogonal_operation(self, method=""):
        # no regularization defined for this class of se-blocks
        print ('Running without any orthogonal regularization applied.')
        return 0

# -------------------------------------------
# --- Parallel Split Squeeze & Excitation ---
# -------------------------------------------

class SPLIT_SE_BLOCK(nn.Module):
    '''
       Multiple parallel squeeze & exciation blocks are applied to the input feature map
       the difference between this method & the PARALLEL_SE_BLOCK is that here the parallel SE blocks are applied to adjacent chuncks of input features.
    '''
    def __init__(self, channel_size, r, batch_size, stride=2):
        '''
           input:
                 split: split value for the feature map on each side (e.g. 2 means 2 on each side (i.e. width & height) of the feature map.
                        making 4 different blocks to apply SE blocks. (split=3 makes 9 feature maps).
                        the squeeze weights are the same for all splits but the excite weights are different
                        this is different from parallel in which both squeeze & excite vectors are different for all splits
        '''

        super(SPLIT_SE_BLOCK, self).__init__()

        self.K = stride*stride
        self.pads = create_pads(stride)
        self.channel_size = channel_size
        self.batch_size = batch_size

        self.fixed_pool = nn.MaxPool2d(1, stride=stride, return_indices=True)

        intermediate_size = self.channel_size // r

        self.squeeze = nn.Sequential( 
                nn.Linear( channel_size, intermediate_size),
                nn.ReLU(inplace=True)
            )

        self.excites = []
        for i in range(0,self.K):
            self.excites.append( nn.Linear( intermediate_size, channel_size) )

        self.excites = nn.ModuleList(self.excites)

        excite_size = self.excites[0].weight.size()
        self.excite_len = excite_size[0]*excite_size[1]

        self.sigmoid = nn.Sigmoid()

        self.fixed_unpool = nn.MaxUnpool2d(1, stride=stride)

    def process_stream(self, split_indx, feature_map):
        '''
           processes each split feature map
           inputs:
                  split_indx:  shows the n-th split of the feature map (n=split_indx)
                  feature_map: is the total feature map 
        '''

        l,u = self.pads[split_indx]

        # the feature map is further split by the generated pads from the create_pads() function into self.pads
        f_map = feature_map[:,:,l:,u:]

        pool,indice = self.fixed_pool(f_map[:])

        # apply squeeze & excite on the split feature map
        gap_pool = global_average_pooling(pool[:])
        squeeze = self.squeeze(gap_pool[:])
        excite = self.sigmoid( self.excites[split_indx](squeeze[:]) )
        se = torch.mul(pool[:], excite[:].view(self.batch_size, self.channel_size, 1, 1))
        unpool= F.pad(self.fixed_unpool(se[:], indice[:], output_size=f_map.size()), (u,0,l,0), "constant", 0)

        return unpool

    def forward(self, feature_map):

        parallel_out = self.process_stream(0, feature_map)

        for i in range(1,self.K):
            parallel_out += self.process_stream(i, feature_map)

        return parallel_out

    def orthogonal_operation(self, method=""):
        '''
           input:
                 method: 
                        0. none
                        1. complete
                        2. simple
                        3. loose
                        4. weak
                        5. subcomplete
        '''

        if "_" in method:
            method, eye_term = method.split("_")
        else:
        # orthogonal regularization
            eye_term = "weye"

        if method=="complete":        
            return self.orthogonal_sentence(0,1,eye_term) + \
                   self.orthogonal_sentence(0,2,eye_term) + \
                   self.orthogonal_sentence(0,3,eye_term) + \
                   self.orthogonal_sentence(1,2,eye_term) + \
                   self.orthogonal_sentence(1,3,eye_term) + \
                   self.orthogonal_sentence(2,3,eye_term)

        if method=="subcomplete":
            return self.orthogonal_sentence(1,2,eye_term) + \
                   self.orthogonal_sentence(1,3,eye_term) + \
                   self.orthogonal_sentence(2,3,eye_term)

        if method=="simple":
            return self.orthogonal_sentence(0,1,eye_term) + \
                   self.orthogonal_sentence(1,2,eye_term) + \
                   self.orthogonal_sentence(2,3,eye_term) + \
                   self.orthogonal_sentence(3,0,eye_term)

        if method=="loose":
            return self.orthogonal_sentence(1,2,eye_term)

        if method=="weak":
            return self.orthogonal_sentence(0,1,eye_term) + \
                   self.orthogonal_sentence(2,3,eye_term)

        # if type is not specified no regularization
        return 0

    def orthogonal_sentence(self, idx1, idx2, eye_term="weye"):
        '''
           idx1: is the index of a weight vector (1 referse to the second weight vector out of 4 in a 4-split)
           idx2: is the index of another weight vector
           eye_term: 
                  if weye is orthogonal regularization.
                  if woeye is orthogonal regularization without the eye matrix deduction.
        '''
        if eye_term=='weye':

            # matrix multiply of two weight matrices (in the excite stage)
            mat_mul = torch.matmul(self.excites[idx1].weight, torch.transpose(self.excites[idx2].weight, 0, 1)) - torch.eye(self.excites[idx1].weight.size(0)).cuda()

            # Frobenius norm of the matrix
            return torch.norm(mat_mul)

        else: #if eye_term=='woeye':

            # matrix multiply of two weight matrices (in the excite stage)
            mat_mul = torch.matmul(self.excites[idx1].weight, torch.transpose(self.excites[idx2].weight, 0, 1))

            # Frobenius norm of matrix
            return torch.norm(mat_mul)

# -------------------------------------------
# ------------ Squeeze & Recur --------------
# -------------------------------------------

class SR_BLOCK(nn.Module):
    '''
       Recurrence in Squeeze & Excitation
       Instead of applying two fc layers to compute SE weights we use a bi-lstm to compute the weights
    '''

    def __init__(self, channel_size, batch_size):

        super(SR_BLOCK, self).__init__()

        self.channel_size = channel_size
        self.batch_size = batch_size

        self.bilstm = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, feature_map):

        # global average pooling
        gap_output = global_average_pooling(feature_map)

        # apply bi-lstm
        bi_output, _ = self.bilstm(gap_output.unsqueeze(2))

        weights = torch.sum(bi_output, dim=2)

        # scale
        scale_output = torch.mul(feature_map, weights.view(self.batch_size, self.channel_size, 1, 1))

        return scale_output
