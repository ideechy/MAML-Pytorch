#  MAML-Pytorch
Modified PyTorch implementation of [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400) from [this repo](https://github.com/dragen1860/MAML-Pytorch).

- Re-write meta (outer) learner to be consistent with the Pytorch nn.Module
- Support all three datasets in the original paper
- Minor bug fixes

# Platform
- python: 3.x
- Pytorch: 1.0+

# Usage

## Sinusoid

10-shot sinusoid:
```shell
python main.py --data_source sinusoid --task_num 25 --class_num 5 --train_sample_size_per_class 10 --inner_lr 0.001 --inner_step 1
```

## Omniglot

5-way, 1-shot omniglot:
```shell
python main.py --datasource omniglot --task_num 32 --class_num 5 --train_sample_size_per_class 1 --inner_lr 0.4 --inner_step 1 --img_size [28, 28] --data_folder path/to/omniglot
```
20-way, 1-shot omniglot:
```shell
python main.py --datasource omniglot --task_num 16 --class_num 20 --train_sample_size_per_class 1 --inner_lr 0.1 --inner_step 5 --img_size [28, 28] --data_folder path/to/omniglot
```

# MiniImagenet

First download `MiniImagenet` dataset from [here](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4) and the label files `train/val/test.csv` from [here](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet). Extract it like:
```shell
miniimagenet/
├── images
	├── n0210891500001298.jpg  
	├── n0287152500001298.jpg 
	...
├── test.csv
├── val.csv
└── train.csv
```

5-way 1-shot mini imagenet:
```shell
python main.py --datasource miniimagenet --epoch 60000 --task_num 4 --class_num 5 --update_batch_size 1 --inner_lr 0.01 --inner_step 5 --img_size [84, 84] --data_folder path/to/miniimagenet
```
5-way 5-shot mini imagenet:
```shell
python main.py --datasource miniimagenet --epoch 60000 --task_num 4 --class_num 5 --update_batch_size 5 --inner_lr 0.01 --inner_step 5 --img_size [84, 84] --data_folder path/to/miniimagenet
```