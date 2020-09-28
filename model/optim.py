# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

import torch
import torch.optim as Optim

# Optimizer
class AdjustableOptim():

    def __init__(self, _g_params, model, data_size, optimizer_type='sgd', factor=0.1, patience=1, weight_decay=0.0001, lr_base=None):

        self._step = 0

        # make all convs weights with "fixed" in their names non-trainable
        for name, param in model.named_parameters():
            if 'fixed' in name:
                param.requires_grad = False

        self.counter = 0
        self.epochs_decay_list = _g_params.LR_DECAY_LIST

        if lr_base is None:
            self.rate = self.lr_base = _g_params.LR_BASE

        if optimizer_type=="adam":
            self.optimizer = Optim.Adam(
                                         filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=self.lr_base,
                                         betas=_g_params.OPT_BETAS,
                                         eps=_g_params.OPT_EPS
                             )
        elif optimizer_type=="sgd":
            self.optimizer = Optim.SGD(
                                         filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=self.lr_base, 
                                         momentum=0.9,
                                         weight_decay=weight_decay
                             )

        # lowers learning late whenever reaches to a specific epoch specified in the epochs_decay_list (i.e. follows the paper behaviour)
        self.scheduler = Optim.lr_scheduler.MultiStepLR(self.optimizer,  milestones=self.epochs_decay_list)


    def step(self):
        ''' one step of the optimizer '''
        self._step += 1
        self.optimizer.step()

    def scheduler_step(self):
        ''' one step of the scheduler '''
        self.scheduler.step()

    def lr(self):
        ''' 
           returns the learning rate. 
           if there is multiple learning rates for different parameter groups returns the first learning rate 
        '''
        lr = []
        for param_group in self.optimizer.param_groups:
            lr.append( param_group["lr"] )
        lr = list(set(lr))
        if len(lr)==1:
            return lr[0]
        else:
            return "multi: {}".format(lr[0])

    def zero_grad(self):

        self.optimizer.zero_grad()
