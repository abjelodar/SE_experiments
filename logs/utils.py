# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

import os, torch, time, collections
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

# ------------------------------
# ---- Accuracy Computation ----
# ------------------------------

class Accuracy:
    '''
       Class to compute confusion matrix and mean class-wise accuracy for a deep model
    '''

    def __init__(self, num_classes):
        '''
           input: 
                 num_classes: Number of classes in the dataset (e.g. 10 for CIFAR10)
        '''
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        '''
           Set confusion matrix, total mean accuracy, and per class accuracy to all zeros
        '''
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.cumulative_accuracy = 0.0
        self.cumulative_per_class_accuracy = 0.0

    def per_class_accuracy_cumulative(self, preds, labels):
        '''
           Given predictions from a model and ground truth labels, compute confusion matrix & mean class-wise accuracy & mean accuracy
           input: 
                 preds:  predicted batch from the model
                 labels: ground truth batch labels
        '''

        _, preds = torch.max(preds, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            # Cummulative confusion matrix
            self.confusion_matrix[t.long(), p.long()] += 1

        # Class wise accuracy
        self.cumulative_per_class_accuracy = (self.confusion_matrix.diag()/self.confusion_matrix.sum(1)).numpy()

        # Mean class wise accuracy
        self.cumulative_accuracy = np.mean(self.cumulative_per_class_accuracy)

        return self.cumulative_accuracy

    def calc_accuracy(pred, labels):
        '''
           Calculates general model accuracy without considering class-wise accuracies
           input: 
                 pred:  predicted batch from the model
                 labels: ground truth batch labels
        '''

        max_vals, max_indices = torch.max(pred,1)
        n = max_indices.size(0) #index 0 for extracting the # of elements
        acc = (max_indices == Y).sum(dtype=torch.float32)/n
        return acc

# ------------------------------
# ----- Logging of Results -----
# ------------------------------

class LogStats:
    '''
       Creates an api for logging data to log files
    '''

    def __init__(self, _g_params, total_params):
        '''
           Initializes model name, log-file names, and writes initial info of the model to the log file for further logging
           input: 
                 total_params: Total number of parameters in the model (e.g. Resnet, SE)
        '''

        custom_se_block_ids = _g_params.CUSTOM_SE_BLOCKS.split(",")
        custom_se_block_ids = "_".join(custom_se_block_ids)

        if _g_params.ORTHOGONAL!="none":
            orthogonal_name = "_ORTH={}{}".format(_g_params.ORTHOGONAL, _g_params.ORTH_WEIGHT)
        else:
            orthogonal_name = ""

        model_name = "Resnet{}_SE={}_type={}_stride={}_r={}_blockids={}{}".format(
                                                         str(_g_params.RESNET_TYPE), 
                                                         str(_g_params.WITH_SE), 
                                                         _g_params.SE_TYPE,
                                                         str(_g_params.SE_STRIDE),
                                                         str(_g_params.REDUCTION_RATIO),
                                                         custom_se_block_ids,
                                                         orthogonal_name)

        self.model_name = model_name
        self.all_runs_file = os.path.join(_g_params.LOG_STATS_DIR, "all_runs.txt")

        self.stats_log_dir = os.path.join(_g_params.LOG_STATS_DIR, "{}.txt".format(model_name))
        self.logger = open(self.stats_log_dir, 'w')

        self.logger.write(' model properties: \n  depth: Resnet-{}\n  SE: {}\n  SE-type: {}\n  stride: {}\n  reduction-ratio: {}\n'.format(
                                                                                                    str(_g_params.RESNET_TYPE), 
                                                                                                    str(_g_params.WITH_SE), 
                                                                                                    _g_params.SE_TYPE,
                                                                                                    str(_g_params.SE_STRIDE),
                                                                                                    str(_g_params.REDUCTION_RATIO)
                                                                                              ))
        self.logger.write('  total model parameters: {}\n'.format(total_params))
        self.logger.write('\n training stats: \n')
        self.losses = []
        self.train_accs = []
        self.test_accs = []


    def update_stats(self, epoch, epoch_loss, epoch_acc=[0,0]):
        '''
           Update loss and accuracy info (not logged into file yet)
           input: 
                  epoch index, 
                  epoch loss, and 
                  epoch train and test accuracy
        '''
        self.losses.append(round(epoch_loss,6))
        self.train_accs.append(round(epoch_acc[0],4))
        self.test_accs.append(round(epoch_acc[1],4))

    def log_finalize(self, train_time, eval_time):
        '''
           Log all losses and accuracies of all epochs to file.
           Log final train, and test accuracy to a global log file of all experiments
           input: 
                  total train and 
                  evaluation time
        '''
        for i in range(len(self.losses)):
            self.logger.write('  epoch: {}, loss: {:.6f}, train_acc: {:.4f}, test_acc: {:.4f}\n'.format(
                                                                                        i, 
                                                                                        self.losses[i],
                                                                                        self.train_accs[i],
                                                                                        self.test_accs[i]))

        self.logger.write('\n time: {} secs to train.\n'.format(train_time))
        self.logger.write('\n time: {} secs to evaluate.\n'.format(eval_time))

        self.all_logger = open(self.all_runs_file, 'a+')
        self.all_logger.write('{},'.format(self.model_name))
        self.all_logger.write('{},{},{:.6f},{:.4f},{:.4f}\n'.format(train_time,eval_time,self.losses[-1],self.train_accs[-1],self.test_accs[-1]))
        self.all_logger.close()
       
    def save_checkpoint(self, net, optim, params):
        '''
           Save checkpoint
           input:
                 net:   The model to be saved
                 optim: The optimizer used to train the model
        '''

        state = { 'state_dict': net.state_dict(), 'optimizer': optim.optimizer.state_dict(), 'lr_base': optim.lr_base}

        ckpt_filename = os.path.join(params.CKPT_PATH,'ckpt_{}_epoch={}.pkl'.format(self.model_name, epoch_finish))

        torch.save(state, ckpt_filename)


# ------------------------------
# -- Printing Average Results --
# ------------------------------

def all_stats_graph(file_path="output/stats_logs/all_runs.txt", width=5, rounding=2, time_report="multi"):

    train_time_stats = collections.defaultdict(float)
    test__time_stats = collections.defaultdict(float)
    train_acc__stats = collections.defaultdict(float)
    test__acc__stats = collections.defaultdict(float)
    run_countr_stats = collections.defaultdict(int)

    log_file = open(file_path)
    for line in log_file:

        # check to see if the line is just a section break
        if line.startswith("---"):
            continue

        run_name, train_time, test_time, loss, train_acc, test_acc = line.strip().split(",")
            
        train_time_stats[run_name] += float(train_time)
        test__time_stats[run_name] += float(test_time)
        train_acc__stats[run_name] += float(train_acc)
        test__acc__stats[run_name] += float(test_acc)
        run_countr_stats[run_name] += 1

    for run_name in run_countr_stats.keys():

        train_time_stats[run_name] /= run_countr_stats[run_name]
        test__time_stats[run_name] /= run_countr_stats[run_name]
        train_acc__stats[run_name] /= run_countr_stats[run_name]
        test__acc__stats[run_name] /= run_countr_stats[run_name]
        if "SE=False" in run_name and "type=vanilla" in run_name:
            baseline_time__test = test__time_stats[run_name]
            baseline_time_train = train_time_stats[run_name]

    if time_report=="multi":
        for run_name in run_countr_stats.keys():
            train_time_stats[run_name] /= baseline_time_train
            test__time_stats[run_name] /= baseline_time__test
        time_metric = "coef"
    else:
        time_metric = "ms"

    # report test time & acc
    print('{:^80} {:^15}{:^15}{:^10}'.format(" ", "Accuracy(%)", "Time ({})".format(time_metric), "Runs"))

    for run_name in run_countr_stats.keys():
        test_time = round(100.0*test__time_stats[run_name], rounding)
        test__acc = round(100.0*test__acc__stats[run_name], rounding)
        run_count = run_countr_stats[run_name]

        name = " {}:".format(run_name)
        print('{:80} {:^15}{:^15}{:^10}'.format(name, test__acc, test_time, run_count))

