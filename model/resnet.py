# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch
from model.seblock import SE_BLOCK, PARALLEL_SE_BLOCK, pick_se_block

# ------------------------------
# ----- Initialize Weights -----
# ------------------------------

def initialize_weights(module):

    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

# ------------------------------
# ------- Identity Layer -------
# ------------------------------

class IdentityOptionA(nn.Module):
    def __init__(self, out_ch):
        super(IdentityOptionA, self).__init__()
        self.option_a = lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_ch//4, out_ch//4), "constant", 0)

    def forward(self, x):
        return self.option_a(x)

# ------------------------------
# ------ BottleNeck Layer ------
# ------------------------------

class BottleNeck(nn.Module):

    def __init__(self, input_ch=64, intermediate_ch=64, output_ch=256, stride=1, _g_params=None, custom_se_block=False):

        super(BottleNeck, self).__init__()

        if _g_params.RESNET_TYPE=="34":
            self.bottleneck_core = nn.Sequential(
                nn.Conv2d(in_channels=input_ch, out_channels=output_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=output_ch, out_channels=output_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_ch),
            )
        elif _g_params.RESNET_TYPE=="50+":
            self.bottleneck_core = nn.Sequential(
                nn.Conv2d(in_channels=input_ch, out_channels=intermediate_ch, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(intermediate_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=intermediate_ch, out_channels=intermediate_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(intermediate_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=intermediate_ch, out_channels=output_ch, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(output_ch),
            )

        self.shortcut = nn.Sequential()
        if _g_params.RESNET_TYPE=="34" and (stride!=1 or input_ch!=output_ch):

            # option a projection from the resnet paper
            self.shortcut = IdentityOptionA(output_ch)

        elif stride!=1 or input_ch!=output_ch:

            # option b projection from the resnet paper
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=input_ch, out_channels=output_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
            )

        self.relu = nn.ReLU(inplace=True)

        # Different variations of se block (our own novel Squeeze & Excite blocks)
        self.se_block = pick_se_block(_g_params, custom_se_block, output_ch)

        # if there is an orthogonal weight term in the loss function
        self.orthogonal_weights_enabled = _g_params.ORTHOGONAL
        self.orthogonal_loss = None

    def forward(self, feature_map):

        core_out = self.bottleneck_core(feature_map)

        # apply one of the SE blocks (e.g. vanilla, split, parallel)
        se_out = self.se_block( core_out )

        shortcut_out = se_out + self.shortcut( feature_map )

        if self.orthogonal_weights_enabled!="none":
            self.orthogonal_loss = self.se_block[0].orthogonal_operation(self.orthogonal_weights_enabled)
 
        relu_out = self.relu(shortcut_out)

        return relu_out

# ------------------------------
# ----------- Block ------------
# ------------------------------

# Block of resnet
class Block(nn.Module):

    def __init__(self, num_layers=3, input_ch=64, intermediate_ch=64, output_ch=256, stride=1, _g_params=None, custom_se_block=False):

        super(Block, self).__init__()

        strides = [stride] + [1]*(num_layers-1)

        self.iterative_input_ch = input_ch
        layers = []
        for i in range(num_layers):
            layers.append(
                               BottleNeck(
                                               input_ch=self.iterative_input_ch, 
                                               intermediate_ch=intermediate_ch, 
                                               output_ch=output_ch,
                                               stride=strides[i],
                                               _g_params=_g_params,
                                               custom_se_block=custom_se_block
                                         )
                          )
            self.iterative_input_ch = output_ch
    
        self.layers = nn.Sequential(*layers)
        self.orthogonal_weights_enabled = _g_params.ORTHOGONAL

    def forward(self, feature_map):

        # outputs of each layer
        self.outputs = []
  
        x = feature_map
        for i in range(len(self.layers)):
            x = self.layers[i](x)

            # output of layer i appended to outputs
            self.outputs.append(x[:])

            if self.orthogonal_weights_enabled!="none":
                if i==0:
                    self.orthogonal_loss  = torch.sum(self.layers[i].orthogonal_loss)
                else:
                    self.orthogonal_loss += torch.sum(self.layers[i].orthogonal_loss)

        return x

# ------------------------------
# --------- Resnet50 -----------
# ------------------------------

class ResNet(nn.Module):

    def __init__(self, _g_params=None, 
                       num_stacked=[3,4,6,3], inter_channels=[64,128,256,512], out_channels=[256,512,1024,2048], strides=[1,2,2,2],
                       num_classes=10):

        # If it is resnet-34
        if _g_params.RESNET_TYPE=="34":
            n = _g_params.N_LAYERS
            num_stacked=[n,n,n]
            inter_channels=[16,32,64]
            out_channels=[16,32,64]

        # identify which blocks should have a custom se-block specified by the user
        custom_se_blocks = []
        custom_se_block_ids = set(map(int, _g_params.CUSTOM_SE_BLOCKS.split(",")))
        for block_id in range(len(num_stacked)):
            if block_id in custom_se_block_ids:
                custom_se_blocks.append(True)
            else:
                custom_se_blocks.append(False)

        input_channel = inter_channels[0]

        if _g_params.RESNET_TYPE=="34":
            for i in range(0,len(inter_channels)):
                out_channels[i] = inter_channels[i]

        super(ResNet, self).__init__()

        # If it is resnet-50 or more layers
        if _g_params.RESNET_TYPE=="50+":
            self.initial_block = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=input_channel, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif _g_params.RESNET_TYPE=="34":
            self.initial_block = nn.Sequential(
                nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU(inplace=True)
            )

        blocks = []
        for i in range(len(num_stacked)):
            blocks.append( 
                               Block(num_stacked[i], 
                                       input_ch=input_channel, 
                                       intermediate_ch=inter_channels[i], 
                                       output_ch=out_channels[i],
                                       stride=strides[i],
                                       _g_params=_g_params,
                                       custom_se_block=custom_se_blocks[i]
                                     ) 
                          )
            input_channel = out_channels[i]

        self.blocks = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(out_channels[-1], num_classes)

        # initialize weights
        self.apply(initialize_weights)

        self.orthogonal_weights_enabled = _g_params.ORTHOGONAL

    def get_last_block_outputs(self):

        return self.blocks[-1].outputs

    def forward(self, feature_map):
  
        x = self.initial_block(feature_map)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            if self.orthogonal_weights_enabled!="none":
                if i==0:
                    self.orthogonal_loss  = self.blocks[i].orthogonal_loss
                else:
                    self.orthogonal_loss += self.blocks[i].orthogonal_loss

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.linear(x)

        return x




