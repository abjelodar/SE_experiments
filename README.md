
# Squeeze and Excitation Experiments
In this project the Squeeze and Excitation model from the CVPR 2018 paper and some variations are implemented.
The baseline model implemeneted is the Resnet model and the CIFAR10 dataset is used. 

## Methods
0. Vanilla SE: The exact squeeze and excitation module from the CVPR 2018 paper (SE_BLOCK). 
0. Split SE: A version of squeeze and excitation where the feature map is split into same size sections and the same squeeze module is applied to each split feature map but different excitation modules are applied afterwards. The feature map can be split into 4 or 9 regions. (SPLIT_SE_BLOCK).
0. Parallel SE: A version of squeeze and excitation module where four different squeeze and excitation modules are applied to the four neighbouring pixels. The pixels are first separated and the squeeze and excitation is applied separately and then aggregated (PARALLEL_SE_BLOCK).
0. Recurrent SE: The squeeze and excitation module is modified to use a Bi-LSTM instead of the two FC layers (SR_BLOCK).

## Results
Table 1. Classification accuracy of the various models on CIFAR10 for an average of 5 runs.
| Model | Top-1 | Time Comparison
|:-:|:-:|:-:|
|Resnet| 90.99 | 1X
|Vanilla SE| 91.81 | 1.13X
|Split-4 SE| 91.88 | 1.92X
|Split-4 SE (orthogonal)| 92.12 | 1.92X
|Split-4 SE (only last block)| 92.02 | 1.39X
|Split-9 SE| 91.85 | 3.27X 
|Parallel SE| 91.69 | 1.62X
|Recurrent SE| 91.78 | 1.94X

## Usage
0. Required packages.
        - Pytorch 0.4.1
        - other commonly used Python packages such as glob, pickle, etc.
1. Download the [CIFAR10 dataset](cs.toronto.edu/~kriz/cifar.html) (batches and meta file) and place it in dataset/cifar10/.
2. Use the below commands to run the code.
The command for training and evaluation is:
```
python run.py --WITH_SE --SE_TYPE squeeze_recur --GPU 0
```

