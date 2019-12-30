import  numpy as np
import os
from omniglot import Omniglot
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid data.
    A "class" is considered a particular sinusoid function.
    """
    def __init__(self, args):
        self.task_num = args.task_num
        self.class_num = args.class_num
        self.train_sample_size_per_class = args.train_sample_size_per_class
        self.test_sample_size_per_class = args.test_sample_size_per_class
        self.sample_size_per_class = self.train_sample_size_per_class + self.test_sample_size_per_class
        if args.data_source == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = args.amp_range
            self.phase_range = args.phase_range
            self.input_range = args.input_range
            self.dim_input = 1
            self.dim_output = 1
        elif args.data_source == 'omniglot':
            assert self.sample_size_per_class <= 20
            self.generate = self.load_omniglot_batch
            self.img_size = tuple(args.img_size)
            #self.dim_input = args.img_size[0] * args.img_size[1]
            self.dim_output = self.class_num
            self.data_filename = 'omniglot_' + str(args.img_size[0]) + 'x' + str(args.img_size[0]) + '.npy'
            #load processed data or download and process the original data
            if not os.path.isfile(os.path.join(args.data_folder, self.data_filename)):
                # if root/data.npy does not exist, just download it
                omniglot = Omniglot(args.data_folder, download=True,
                                    transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                                  lambda x: x.resize(self.img_size),
                                                                  lambda x: np.expand_dims(x, 0),
                                                                  lambda x: x/255.]))
                temp = defaultdict(list)  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
                for (img, label) in omniglot:
                    temp[label].append(img)
                self.omniglot_data = np.array(list(temp.values()), dtype=np.float)  # [[20 imgs],..., 1623 classes in total]
                del temp  # Free memory
                # save all dataset into npy file.
                np.save(os.path.join(args.data_folder, self.data_filename), self.omniglot_data)
                print('\nWrite data into ' + self.data_filename)
            else:
                # if data.npy exists, just load it.
                self.omniglot_data = np.load(os.path.join(args.data_folder, self.data_filename))
                print('\nLoad data from ' + self.data_filename)    
            self.datasets = {"train": self.omniglot_data[:1200], "test": self.omniglot_data[1200:]}
            print('Training data size: {}, test data size: {}'. format(self.datasets["train"].shape, self.datasets["test"].shape))
            # save pointer of current read batch in total cache
            self.indexes = {"train": 0, "test": 0}
            self.datasets_cache = {"train": self.preload_omniglot_data_cache(self.datasets["train"]),  # current epoch data cached
                                   "test": self.preload_omniglot_data_cache(self.datasets["test"])}
        else:
            raise NotImplementedError
        
    def generate_sinusoid_batch(self, mode='train', input_idx=None):
        # Note mode arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], self.task_num)
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], self.task_num)
        outputs = np.zeros((self.task_num, self.sample_size_per_class, self.dim_output))
        inputs = np.zeros((self.task_num, self.sample_size_per_class, self.dim_input))
        for i in range(self.task_num):
            inputs[i] = np.random.uniform(self.input_range[0], self.input_range[1], (self.sample_size_per_class, self.dim_input))
            if input_idx is not None:
                inputs[i,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.sample_size_per_class-input_idx, retstep=False)
            outputs[i] = amp[i] * np.sin(inputs[i]-phase[i])
        return inputs, outputs, amp, phase

    def preload_omniglot_data_cache(self, datasets):
        cache = []
        train_batch_size = self.train_sample_size_per_class * self.class_num
        test_batch_size = self.test_sample_size_per_class * self.class_num
        for _ in range(16):
            inputs_train = np.zeros((self.task_num, train_batch_size) + datasets.shape[2:])
            outputs_train = np.zeros((self.task_num, train_batch_size), dtype=np.int)
            inputs_test = np.zeros((self.task_num, test_batch_size) + datasets.shape[2:])
            outputs_test = np.zeros((self.task_num, test_batch_size), dtype=np.int)
            for i in range(self.task_num):
                input_train, output_train, input_test, output_test = [], [], [], []
                selected_class = np.random.choice(datasets.shape[0], self.class_num, False)
                for j, class_j in enumerate(selected_class):
                    selected_img = np.random.choice(datasets.shape[1], self.sample_size_per_class, False)
                    input_train.append(datasets[class_j][selected_img[:self.train_sample_size_per_class]])
                    output_train.append([j] * self.train_sample_size_per_class)
                    input_test.append(datasets[class_j][selected_img[self.train_sample_size_per_class:]])
                    output_test.append([j] * self.test_sample_size_per_class)
                # shuffle inside a batch
                perm = np.random.permutation(train_batch_size)
                inputs_train[i] = np.array(input_train).reshape((train_batch_size,) + datasets.shape[2:])[perm]
                outputs_train[i] = np.array(output_train).reshape(train_batch_size)[perm]
                perm = np.random.permutation(test_batch_size)
                inputs_test[i] = np.array(input_test).reshape((test_batch_size,) + datasets.shape[2:])[perm]
                outputs_test[i] = np.array(output_test).reshape(test_batch_size)[perm]
            cache.append((inputs_train, outputs_train, inputs_test, outputs_test))
        return cache
    
    def load_omniglot_batch(self, mode='train'):
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.preload_omniglot_data_cache(self.datasets[mode])
        batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return batch