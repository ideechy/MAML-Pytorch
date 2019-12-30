import torch
from torch import nn
from torch.nn import functional as F
from torch.multiprocessing import Pool
import numpy as np
from inner import Inner
from copy import deepcopy

class Outer(nn.Module):
    """
    Meta learner for the outer loop
    """
    def __init__(self, args, config=None):
        """

        :param config: network config file, type:list of (string, list)
        """
        super(Outer, self).__init__()
        self.task_num = args.task_num
        self.inner_step = args.inner_step
        self.inner_lr = args.inner_lr
        self.class_num = args.class_num
        self.train_sample_size_per_class = args.train_sample_size_per_class
        self.test_sample_size_per_class = args.test_sample_size_per_class
        if args.data_source == 'sinusoid':
            self.classification = False
            self.loss_func = F.mse_loss
            if config is None:
                config = [('linear', [40, 1]),
                          ('relu', [True]),
                          ('linear', [40, 40]),
                          ('relu', [True]),
                          ('linear', [1, 40])]
        elif args.data_source == 'omniglot':
            self.classification = True
            self.loss_func = F.cross_entropy
            if config is None:
                config =[('conv2d', [64, 1, 3, 3, 2, 0]),
                         ('relu', [True]),
                         ('bn', [64]),
                         ('conv2d', [64, 64, 3, 3, 2, 0]),
                         ('relu', [True]),
                         ('bn', [64]),
                         ('conv2d', [64, 64, 3, 3, 2, 0]),
                         ('relu', [True]),
                         ('bn', [64]),
                         ('conv2d', [64, 64, 2, 2, 1, 0]),
                         ('relu', [True]),
                         ('bn', [64]),
                         ('flatten', []),
                         ('linear', [self.class_num, 64])]
        elif arg.data_source == 'miniimagenet':
            self.classification = True
            self.loss_func = F.cross_entropy
            if config is None:
                config = [('conv2d', [32, 3, 3, 3, 1, 0]),
                          ('relu', [True]),
                          ('bn', [32]),
                          ('max_pool2d', [2, 2, 0]),
                          ('conv2d', [32, 32, 3, 3, 1, 0]),
                          ('relu', [True]),
                          ('bn', [32]),
                          ('max_pool2d', [2, 2, 0]),
                          ('conv2d', [32, 32, 3, 3, 1, 0]),
                          ('relu', [True]),
                          ('bn', [32]),
                          ('max_pool2d', [2, 2, 0]),
                          ('conv2d', [32, 32, 3, 3, 1, 0]),
                          ('relu', [True]),
                          ('bn', [32]),
                          ('max_pool2d', [2, 1, 0]),
                          ('flatten', []),
                          ('linear', [self.class_num, 32 * 5 * 5])]
        else:
            raise NotImplementedError
        self.model = Inner(config)
    
    def forward(self, x_train, y_train, x_test):
        """

        :param x_train:  [task_num, class_num*train_sample_size_per_class, input_size]
        :param y_train:  [task_num, class_num*train_sample_size_per_class, output_size]
        :param x_test:   [task_num, class_num*test_sample_size_per_class, input_size]
        :return:
        """
        output_size = (self.task_num, self.class_num * self.test_sample_size_per_class)
        if self.classification:
            output_size += (self.class_num,)
            output = y_train.new_empty(output_size, dtype=torch.float)
        else:
            output_size += y_train.shape[2:]
            output = y_train.new_empty(output_size)
              
        for i in range(self.task_num):
            # the first step of the inner loop
            output_inner = self.model(x_train[i], self.model.parameters(), bn_training=True) #logits
            loss_inner = self.loss_func(output_inner, y_train[i]) #loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss_inner, self.model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, self.model.parameters())))
            # the rest of the inner loop
            for _ in range(1, self.inner_step):
                output_inner = self.model(x_train[i], fast_weights, bn_training=True)
                loss_inner = self.loss_func(output_inner, y_train[i])
                grad = torch.autograd.grad(loss_inner, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, fast_weights)))
            # calculate output of the outer loop
            output[i] = self.model(x_test[i], fast_weights, bn_training=True)
            
        return output
    
    def fine_tuning(self, x_train, y_train, x_test, y_test):
        """

        :param x_train:  [task_num, class_num*train_sample_size_per_class, input_size]
        :param y_train:  [task_num, class_num*train_sample_size_per_class, output_size]
        :param x_test:   [task_num, class_num*test_sample_size_per_class, input_size]
        :param y_test:   [task_num, class_num*test_sample_size_per_class, output_size]
        :return:
        """
        # record the first training error and all-step test errors during fine-tuning
        loss_summary = np.zeros(self.inner_step + 2)
        if self.classification: 
            accuracy_summary = np.zeros(self.inner_step + 2)
        
        model = deepcopy(self.model)
              
        for i in range(self.task_num):
            # the first step of the inner loop
            output_inner = model(x_train[i], model.parameters(), bn_training=True) #logits
            loss_inner = self.loss_func(output_inner, y_train[i]) #loss = F.cross_entropy(logits, y_spt[i])
            ## train error before update
            loss_summary[0] += loss_inner.item()                
            grad = torch.autograd.grad(loss_inner, model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, model.parameters())))
            if self.classification:
                with torch.no_grad():
                    y_pred_inner = F.softmax(output_inner, dim=1).argmax(dim=1)
                    accuracy_summary[0] += torch.eq(y_pred_inner, y_train[i]).to(torch.float).mean().item()
            ## test error before update
            with torch.no_grad():
                output_test = model(x_test[i], model.parameters(), bn_training=True)
                loss_test = self.loss_func(output_test, y_test[i])
                loss_summary[1] += loss_test.item()
                if self.classification:
                    y_pred_test = F.softmax(output_test, dim=1).argmax(dim=1)
                    accuracy_summary[1] += torch.eq(y_pred_test, y_test[i]).to(torch.float).mean().item()
            ## test error after the first update
            with torch.no_grad():
                output_test = model(x_test[i], fast_weights, bn_training=True)
                loss_test = self.loss_func(output_test, y_test[i])
                loss_summary[2] += loss_test.item()
                if self.classification:
                    y_pred_test = F.softmax(output_test, dim=1).argmax(dim=1)
                    accuracy_summary[2] += torch.eq(y_pred_test, y_test[i]).to(torch.float).mean().item()
            # the rest of the inner loop
            for j in range(1, self.inner_step):
                output_inner = model(x_train[i], fast_weights, bn_training=True)
                loss_inner = self.loss_func(output_inner, y_train[i])
                grad = torch.autograd.grad(loss_inner, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, fast_weights)))
                with torch.no_grad():
                    output_test = model(x_test[i], fast_weights, bn_training=True)
                    loss_test = self.loss_func(output_test, y_test[i])
                    loss_summary[j + 2] += loss_test.item()
                    if self.classification:
                        y_pred_test = F.softmax(output_test, dim=1).argmax(dim=1)
                        accuracy_summary[j + 2] += torch.eq(y_pred_test, y_test[i]).to(torch.float).mean().item()
            
        del model 
        if self.classification:
            return loss_summary/self.task_num, accuracy_summary/self.task_num
        else:
            return loss_summary/self.task_num