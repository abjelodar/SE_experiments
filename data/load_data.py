# --------------------------------------------------------
# Squeeze & Excitation Project
# Written by Ahmad Babaeian Jelodar
# --------------------------------------------------------

from data.data_utils import create_one_hot, pad4
import os, sys
import numpy as np
import glob, torch, time, random
import torch.utils.data as Data
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms

class DataSet(Data.Dataset):

    def __init__(self, _g_params, split):

        # Loading all image paths
        split_path = _g_params.IMAGE_DATA_PATH[split]

        # identify classes of the dataset by folder names
        classes = []
        classes_list = os.listdir(split_path)
        for cl in classes_list:
            classes.append(cl)

        self.num_classes = len(classes)
        self.classname2id = {}

        # load images of the dataset
        indx = 0
        self.img_data = []
        for cl in classes:
            class_path = os.path.join(split_path,cl)
            images_path_list = glob.glob(class_path + '/*.jpg')

            self.classname2id[cl] = indx
            indx += 1

            # all image paths and their labels are added to the image data list
            self.img_data.extend([(images_path_list[i], cl) for i in range(0,len(images_path_list))])

        # shuffle the data
        self.shuffle()

        self.data_size = len(self.img_data)

        # identify width & height of a sample of the dataset when images of the datset are of the same size
        img_path, lbl = self.img_data[0]
        image = np.asarray( Image.open(img_path).convert('RGB') )
        self.width  = image.shape[0]
        self.height  = image.shape[1]

        print(' Dataset size: {}'.format(self.data_size))
        print('Finished!')


    def shuffle(self):
        # shuffle the data
        np.random.shuffle(self.img_data)

    def __getitem__(self, idx):
        '''
            load image & its label using the idx
            input:
                  idx represents the idx-th instance in the dataset
        '''

        img_path, lbl = self.img_data[idx]

        image = Image.open(img_path).convert('RGB')

        # convert to tensor
        image =  torch.from_numpy(np.asarray(image)).float()

        # move channel from dim 2 to dim 0
        image = image.permute(2, 0, 1)

        return image, self.classname2id[lbl]

    def __len__(self):
        return self.data_size

# ------------------------------
# ---------- CIFAR10 -----------
# ------------------------------

class CIFAR10DataSet(Data.Dataset):

    def __init__(self, data_path, mean=None, train=True, num_classes=10):

        self.train = train

        # Loading all batches of files
        # extract image paths for train and test sets
        batch_files = os.listdir(data_path)
        train_segments = []
        for f in batch_files:
            if f.startswith("data_batch"):
                train_segments.append( os.path.join(data_path, f) )
            elif f=="test_batch":
                test_segment = os.path.join(data_path, f)

        with open(os.path.join(data_path, "batches.meta"), 'rb') as fo:
            meta_data = pickle.load(fo, encoding='bytes')

        # number of classes (10 for CIFAR10)
        self.num_classes = len(meta_data[b'label_names'])

        # load class names for CIFAR10
        self.names = []
        for name in meta_data[b'label_names']:
            name = str(name, 'utf-8')
            self.names.append(name)

        self.width  = 32
        self.height = 32
        self.create_stats()

        indx = 0
        if train==True:
            self.load_images_to_memory(train_segments)
        else:
            self.mean = mean
            self.load_images_to_memory([test_segment])

        # shuffle the data
        self.shuffle()

        self.data_size = len(self.img_data)

        l = 40
        # index map for data augmentation: indice show which crop of the image is taken for training
        self.indx_map = [(0,l-8,0,l-8), 
                        (8,l,0,l-8), 
                        (0,l-8,8,l),
                        (8,l,8,l), 
                        (4,l-4,4,l-4)]

        print(' Dataset size: {}'.format(self.data_size))
        print('Finished!')

    # shuffle the data
    def shuffle(self):
        np.random.shuffle(self.img_data)

    def create_stats(self):
        '''
           creates a constant (fixed) mean image at the begining to deduct all images by it before training
        '''
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.means = []
        self.stds = []
        for i in range(0,len(mean)):
            self.means.append( np.tile([mean[i]], (self.width, self.height)) )
            self.stds.append( np.tile([std[i]], (self.width, self.height)) )

        self.means = np.asarray(self.means)
        self.stds = np.asarray(self.stds)

    def load_images_to_memory(self, segments):
        self.img_data = []

        # load images 
        for seg in segments:
            # open a segment of the CIFAR10 dataset
            cur_dict = self.unpickle(seg)
            for i in range(0,len(cur_dict[b"data"])):
                #img = cur_dict[b"data"][i,:]
                img = cur_dict[b"data"][i,:]/255.0
                img = np.reshape(img, (3, 32, 32))

                # deduct by mean image
                img = (img-self.means)/self.stds

                # pad by 4 zero pixels on each side and create 
                # 	a) padded image
                #	b) padded & flipped image
                img_origin = pad4(img.copy())
                img_flipped = pad4(np.flip(img,2).copy())

                # keep the original image, padded image, and flipped-padded image as a tuple in the dataset (in memory)
                self.img_data.append(([img, img_origin, img_flipped], cur_dict[b"labels"][i]))

    def unpickle(self, file):
        # opens a segment of the CIFAR10 dataset
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def online_augment(self, imgs, l=40):
        '''
           creates a data-augmented version of a given image
           augmentation is either cropped or flipped or both
        '''

        # get crop from flipped image if r is greater or equal to 5 (half of the time)
        r = random.randint(0,9)

        # padded image or flipped & padded image
        img = imgs[2] if r>=5 else imgs[1]

        # r1 and r2 represent what crop of the image should be taked for training
        r1 = random.randint(0,8)
        r2 = random.randint(0,8)

        return img[:,r1:r1+32,r2:r2+32]

    def __getitem__(self, idx):

        # get the next instance of data (image and its label)
        imgs, lbl = self.img_data[idx]

        # augment only when training
        if self.train:
            img = self.online_augment(imgs)
        else:
            img = imgs[0]

        # assertion check to see if image width & height is correct
        if not (img.shape[0]==3 and img.shape[1]==32 and img.shape[2]==32):
            print (img.shape)
            print ("error:-shape-of-image-is-not-3-32-32")
            sys.exit()

        # convert to tensor
        image =  torch.from_numpy(np.asarray(img)).float()

        return image, lbl, idx

    def __len__(self):
        return self.data_size
