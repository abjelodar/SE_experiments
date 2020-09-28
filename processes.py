# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from data.load_data import DataSet
from model.optim import AdjustableOptim
from model.seblock import SE_BLOCK
from model.resnet import ResNet
from logs.utils import Accuracy, LogStats

class Process:

    def __init__(self, _g_params, dataset_train, dataset_eval=None):

        # initialize all parameters needed for creating the deep model & training ...
        self.params = _g_params

        # Obtain needed information
        self.data_size = dataset_train.data_size
        self.num_classes = dataset_train.num_classes

        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval

        torch.set_printoptions(profile="full")

    def train(self):
        '''
           Creates a deep model (e.g. Resnet or SE based) and trains it using the given dataset (e.g. CIFAR10)
        '''

        # Define the deep model, (e.g. Resnet, SE, etc)
        net = ResNet(self.params)

        # Object instances for computing confusion matrix & mean class accuracy for train & test data
        self.train_accuracy = Accuracy(self.num_classes)
        self.eval_accuracy = Accuracy(self.num_classes)

        # Total number of the defined model
        self.net_total_params = sum(p.numel() for p in net.parameters())
        print ('Total number of model parameters: {}'.format(self.net_total_params))

        # Object for logging training info (e.g. train loss & accuracy and evaluation accuracy)
        log_stats = LogStats(self.params, total_params=self.net_total_params)

        # Find the device to run the model on (if there is a gpu use that)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print (" The device selected is {}".format(device))

        net = net.to(device)

        # Set up the model in train mode
        net.train()

        # Define the multi-gpu training if needed
        if device != 'cpu' and torch.cuda.device_count() > 1:
            net = nn.DataParallel(net, device_ids=self.params.DEVICES)

        # Define the cross entropy loss (combines softmax + negative-log-likelihood loss)
        if torch.cuda.is_available():
            loss_fn = torch.nn.CrossEntropyLoss().cuda()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        # Optimizer defined
        optim = AdjustableOptim(self.params, net, self.data_size)

        start_epoch = 0
        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        dataloader, eval_dataloader = self.create_dataloaders()

        train_time = 0
        eval_time = 0
        # Training script
        for epoch in range(start_epoch, self.params.MAX_EPOCH):

            # Externally shuffle
            if self.params.SHUFFLE_MODE == 'external':
                self.dataset_train.shuffle()

            time_start = time.time()

            print ('')

            # Learning rate decay (the lr update is like the original Resnet paper)
            optim.scheduler_step()

            # Iteration
            for step, (img, label, idx) in enumerate(dataloader):

                optim.zero_grad()

                img = img.to(device)
                label = label.to(device)

                # Feed forward
                pred = net(img)

                # Loss computation & backward
                loss = loss_fn(pred, label)
                # if orthogonality of SE weights is set to True
                if self.params.ORTHOGONAL!="none":
                    loss += self.params.ORTH_WEIGHT * net.orthogonal_loss

                loss.backward()

                # Optimize (updates weights)
                optim.step()

                # loss value added to total loss
                loss_sum += loss.item()

                # Train accuracy calculation
                train_acc = self.train_accuracy.per_class_accuracy_cumulative(pred, label)

                loss_np = loss.item() / self.params.BATCH_SIZE

                if self.params.VERBOSE:
                    print("\r[epoch %2d][step %4d/%4d][%s] loss: %.4f, acc: %.3f, lr: %.2e" % (
                            epoch + 1,
                            step,
                            int(self.data_size / self.params.BATCH_SIZE),
                            'train',
                            loss_np,
                            train_acc,
                            optim.lr()
                    ), end='          ')

            train_time += int(time.time() - time_start)

            eval_acc = 0.0
            # Eval after every epoch
            if self.dataset_eval is not None:
                time_start = time.time()
                eval_acc = self.eval(net, eval_dataloader, epoch)
                eval_time += int(time.time() - time_start)

            # Updates log info for train & test accuracies & losses
            log_stats.update_stats(epoch=epoch, epoch_loss=loss_np, epoch_acc=[train_acc, eval_acc])

            # Reset all computed variables of logs for next epoch
            self.train_accuracy.reset()
            self.eval_accuracy.reset()

            # print('')
            epoch_finish = epoch + 1

            print("\ntrain acc: {},  eval acc: {}".format(train_acc, eval_acc))

            loss_sum = 0

        #self.save_outputs(net, dataloader, device)

        # Keeps a log of total training time & eval time in file "output/stats_logs/all_runs.txt"
        log_stats.log_finalize(train_time, eval_time)

    def save_outputs(self, net, dataloader, device):
        '''
           This function saves the last block output of a resnet model for the entire training data (each training data in a separate file)
        '''

        # Iteration
        for step, (img, label, idx) in enumerate(dataloader):

            img = img.to(device)
            label = label.to(device)

            # Feed forward
            pred = net(img)

            # get the outputs of the last block
            outputs = net.get_last_block_outputs()
            outputs = torch.stack(outputs, 4)
            
            for i in range( outputs.size(0) ):
                file_name = os.path.join( self.params.CIFAR10_MID_PATH, "resnet_outputs_{}.pt".format(idx[i]) )
                torch.save(outputs[i,:], file_name)

    def eval(self, net, eval_dataloader, epoch):
        '''
           Evaluates a deep model on the eval dataset (e.g. CIFAR10)
        '''

        # Find the device to run the model on (if there is a gpu use that)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Iterations
        for step, (img, label, idx) in enumerate(eval_dataloader):

            # Put tesnors on gpu if available
            img = img.to(device)
            label = label.to(device)

            # Feed forward
            pred = net(img)

            # Eval accuracy calculation
            eval_acc = self.eval_accuracy.per_class_accuracy_cumulative(pred, label)

        return eval_acc

    def create_dataloaders(self):
        '''
           Creates data loaders for batching data (i.e. train or test) for training or evaluating a model
        '''
        dataloader = Data.DataLoader(
                self.dataset_train,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False if self.params.SHUFFLE_MODE in ['external'] else True,
                num_workers=self.params.NUM_WORKERS,
                pin_memory=self.params.PIN_MEM,
                drop_last=True
        )

        eval_dataloader = Data.DataLoader(
                self.dataset_eval,
                batch_size=self.params.BATCH_SIZE,
                shuffle=False,
                num_workers=self.params.NUM_WORKERS,
                pin_memory=self.params.PIN_MEM,
                drop_last=True
        )

        return dataloader, eval_dataloader
