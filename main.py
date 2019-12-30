"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --data_source sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10
    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10
    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/
    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/
    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True
    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True
"""
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from outer import Outer
from data_generator import DataGenerator
import argparse

def train(epoch):
    if args.data_source == 'sinusoid':
        x, y, amp, phase = data_generator.generate()
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        train_batch_size = maml.class_num * maml.train_sample_size_per_class
        x_train = x[:, :train_batch_size].to(device)
        y_train = y[:, :train_batch_size].to(device)
        x_test = x[:, train_batch_size:].to(device)
        y_test = y[:, train_batch_size:].to(device)
    elif args.data_source == 'omniglot':
        x_train, y_train, x_test, y_test = data_generator.generate(mode='test')
        x_train = torch.from_numpy(x_train).to(device=device, dtype=torch.float)
        y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.long)
        x_test = torch.from_numpy(x_test).to(device=device, dtype=torch.float)
        y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.long)
    else:
        raise NotImplementedError
    output = maml(x_train, y_train, x_test)
    loss = sum(list(map(lambda x: maml.loss_func(x[0], x[1]), zip(output, y_test))))/maml.task_num
    outer_optimizer.zero_grad()
    loss.backward()
    outer_optimizer.step()
    if epoch % PRINT_INTERVAL == 0:
        if maml.classification:
            with torch.no_grad():
                y_pred = F.softmax(output, dim=2).argmax(dim=2)
                accuracy = torch.eq(y_pred, y_test).to(torch.float).mean().item()
            print('Epoch {:5} train loss: {:.4f}, accuracy: {:.4f}'.format(epoch, loss.item(), accuracy))
        else:
            print('Epoch {:5} train loss: {:.4f}'.format(epoch, loss.item()))
        
def test(epoch):
    if args.data_source == 'sinusoid':
        x, y, amp, phase = data_generator.generate()
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        train_batch_size = maml.class_num * maml.train_sample_size_per_class
        x_train = x[:, :train_batch_size].to(device)
        y_train = y[:, :train_batch_size].to(device)
        x_test = x[:, train_batch_size:].to(device)
        y_test = y[:, train_batch_size:].to(device)
    elif args.data_source == 'omniglot':
        x_train, y_train, x_test, y_test = data_generator.generate(mode='test')
        x_train = torch.from_numpy(x_train).to(device=device, dtype=torch.float)
        y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.long)
        x_test = torch.from_numpy(x_test).to(device=device, dtype=torch.float)
        y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.long)
    else:
        raise NotImplementedError
    summary = maml.fine_tuning(x_train, y_train, x_test, y_test)
    print('+' + '-'* 50 + '\n| Fine-tuning using meta parameter from epoch {}'.format(epoch))
    if maml.classification:
        loss_summary, accuracy_summary = summary
        print('| Step 0 train loss: {:.4f}, accuracy: {:.4f}'.format(loss_summary[0], accuracy_summary[0]))
        for i in range(1 + maml.inner_step):
            print('| Step {}  test loss: {:.4f}, accuracy: {:.4f}'.format(i, loss_summary[i + 1], accuracy_summary[i + 1]))
    else:
        print('| Step 0 training loss: {:.4f}'.format(summary[0]))
        for i in range(1 + maml.inner_step):
            print('| Step {}  test loss: {:.4f}'.format(i, summary[i + 1]))
    print('+' + '-'* 50)
                
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # global options
    argparser.add_argument('--device', default='cuda', help='device for training the model.')
    argparser.add_argument('--data_source', default='omniglot', help='sinusoid or omniglot or miniimagenet.')
    # task options
    argparser.add_argument('--task_num', type=int, default=32, help='number of tasks sampled per meta-update')
    argparser.add_argument('--class_num', type=int, default=5, help='number of classes used in classification (e.g. 5-way classification).')
    argparser.add_argument('--train_sample_size_per_class', type=int, default=1, help='number of examples per class used for inner gradient update (K for K-shot learning).')
    argparser.add_argument('--test_sample_size_per_class', type=int, default=15, help='number of examples per class used for outer gradient update.')
    # training options
    argparser.add_argument('--epoch', type=int, default=15000, help='number of metatraining iterations.')
    argparser.add_argument('--outer_lr', type=float, default=0.001, help='the base learning rate of the generator')
    argparser.add_argument('--inner_step', type=int, default=5, help='number of inner gradient updates during training.')
    argparser.add_argument('--inner_lr', type=float, default=0.4, help='step size alpha for inner gradient update.')
    # options for sinusoid data
    argparser.add_argument('--amp_range', type=float, nargs=2 default=[0.1, 5.0], help='range of amplitude.')
    argparser.add_argument('--phase_range', type=float, nargs=2, default=[0, np.pi], help='range of phase.')
    argparser.add_argument('--input_range', type=float, nargs=2, default=[-5.0, 5.0], help='range of input.')
    # options for omniglot data
    argparser.add_argument('--data_folder', default='.\data\omniglot', help='folder for storing the omniglot data.')
    argparser.add_argument('--img_size', type=int, nargs=2, default=[28, 28], help='size of training image.') # remember to input model config if use the non-default img_size
    args = argparser.parse_args()
    
    if args.class_num == 1:
        print('{}-shot regression on {} data.'.format(args.train_sample_size_per_class, args.data_source))
    else:
        print('{}-shot {}-way classification on {} data.'.format(args.train_sample_size_per_class, args.class_num, args.data_source))
    print('\nConfigurations:')
    print('Inner loop:\n  Training batch size: {}\n  Test batch size: {}\n  Training step: {}\n  Learning rate: {}'.format(args.train_sample_size_per_class, args.test_sample_size_per_class, args.inner_step, args.inner_lr))
    print('Outer loop:\n  Batch size: {}\n  Training step: {}\n  Learning rate: {}'.format(args.task_num, args.epoch, args.outer_lr))
    print('Data options:')
    if args.data_source == 'sinusoid':
        print('  Amp range: [{:0.2f}, {:0.2f}]\n  Phase range: [{:0.2f}, {:0.2f}]\n  Input Range: [{:0.2f}, {:0.2f}]'.format(args.amp_l, args.amp_u, args.phase_l, args.phase_u, args.input_l, args.input_u))
    elif args.data_source in ['omniglot', 'miniimagenet']:
        print('  Image size: {}'.format(tuple(args.img_size)))
    else:
        print('  None')
    
    device = torch.device(args.device)
    maml = Outer(args).to(device)
    print('\nModel:')
    print(maml.model)
    parameter_num = sum(map(lambda x: np.prod(x.shape) if x.requires_grad else 0, maml.parameters()))
    print('Total trainable parameters: {}'.format(parameter_num))
    outer_optimizer = optim.Adam(maml.parameters(), lr=args.outer_lr)
    data_generator = DataGenerator(args)
    if args.data_source == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    print('\nTraining on {}.\n'.format(args.device))
    for epoch in range(args.epoch + 1):
        train(epoch)
        if epoch % TEST_PRINT_INTERVAL == 0:
            test(epoch)