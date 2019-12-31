import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import os
import csv
from collections import defaultdict


class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    data_folder:
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    """

    def __init__(self, args, mode, total_task_num):
        """

        :param root: root path of mini-imagenet
        :param mode: train or test
        :param total_task_num: total task number in a dataset
        """

        self.total_task_num = total_task_num  # total task number in a dataset, not meta batch size
        self.class_num = args.class_num
        self.train_sample_size_per_class = args.train_sample_size_per_class
        self.test_sample_size_per_class = args.test_sample_size_per_class
        self.sample_size_per_class = self.train_sample_size_per_class + self.test_sample_size_per_class
        self.img_size = tuple(args.img_size)

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize(self.img_size),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize(self.img_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(args.data_folder, 'images')  # image path
        self.img_filename_dict = self.load_csv(os.path.join(args.data_folder, mode + '.csv'))  # csv path

    def load_csv(self, csv_filepath):
        """
        return a dict saving the information of csv
        :param csv_path: csv file name
        :return: {label:[file1, file2 ...]}
        """
        img_filename_dict = defaultdict(list)
        with open(csv_filepath) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader, None)  # skip (filename, label)
            for row in reader:
                filename, label = row[0], row[1]
                img_filename_dict[label].append(filename)
        return img_filename_dict

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        train_batch_size = self.train_sample_size_per_class * self.class_num
        test_batch_size = self.test_sample_size_per_class * self.class_num
        input_train = torch.empty((train_batch_size, 3) + self.img_size)
        output_train = torch.zeros(train_batch_size, dtype=torch.long)
        input_test = torch.empty((test_batch_size, 3) + self.img_size)
        output_test = torch.zeros(test_batch_size, dtype=torch.long)
        labels = list(self.img_filename_dict.keys())
        selected_class = np.random.choice(len(labels), self.class_num, False)
        for i, class_i in enumerate(selected_class):
            selected_img = np.random.choice(len(self.img_filename_dict[labels[class_i]]), self.sample_size_per_class, False)
            for j, img_j in enumerate(selected_img[:self.train_sample_size_per_class]):
                img_filename = self.img_filename_dict[labels[class_i]][img_j]
                input_train[i * self.train_sample_size_per_class + j] = self.transform(os.path.join(self.path, img_filename))
                output_train[i * self.train_sample_size_per_class + j] = i
            for j, img_j in enumerate(selected_img[self.train_sample_size_per_class:]):
                img_filename = self.img_filename_dict[labels[class_i]][img_j]
                input_test[i * self.test_sample_size_per_class + j] = self.transform(os.path.join(self.path, img_filename))
                output_test[i * self.test_sample_size_per_class + j] = i
        perm = np.random.permutation(train_batch_size)
        input_train, output_train = input_train[perm], output_train[perm]
        perm = np.random.permutation(test_batch_size)
        input_test, output_test = input_test[perm], output_test[perm]
        return input_train, output_train, input_test, output_test

    def __len__(self):
        return self.total_task_num
    
if __name__ == "__main__":
    import argparse
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--class_num', type=int, default=5, help='number of classes used in classification (e.g. 5-way classification).')
    argparser.add_argument('--train_sample_size_per_class', type=int, default=1, help='number of examples per class used for inner gradient update (K for K-shot learning).')
    argparser.add_argument('--test_sample_size_per_class', type=int, default=1, help='number of examples per class used for outer gradient update.')
    argparser.add_argument('--data_folder', default='C:/Users/haoyu/Pictures/miniimagenet', help='folder for storing the omniglot data.')
    argparser.add_argument('--img_size', type=int, nargs=2, default=[8, 8], help='size of training image.') # remember to input model config if use the non-default img_size
    args = argparser.parse_args()
    
    # the following episode is to view one set of images via tensorboard.
    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = MiniImagenet(args, mode='train', total_task_num=1000)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close()