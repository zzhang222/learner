#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:34:17 2020

@author: zen
"""

from learner.utils import accuracy
import learner as ln
from torch.utils.data import DataLoader
import torchvision

class MNISTData(ln.Data):
    def __init__(self):
        super(MNISTData, self).__init__()
        self.__init_data()
    
    def __init_data(self):
        tr_loader = DataLoader(torchvision.datasets.MNIST('data/mnist', train=True, download=True))
        X_train = tr_loader.dataset.data
        self.X_train = X_train.view([X_train.shape[0],1,X_train.shape[1],X_train.shape[2]])*1.0/256
        self.y_train = tr_loader.dataset.targets
        
        ts_loader = DataLoader(torchvision.datasets.MNIST('data/mnist', train=False, download=True))
        X_test = ts_loader.dataset.data[:64]
        self.X_test = X_test.view([X_test.shape[0],1,X_test.shape[1],X_test.shape[2]])*1.0/256
        self.y_test = ts_loader.dataset.targets[:64]

def main():
    device = 'gpu' # 'cpu' or 'gpu'
    data = MNISTData()
    # FNN
    in_channels = 4
    hidden_channels = 100
    out_channels = 4
    ind = 392
    outd = 392
    layers = 3
    # training
    lr = 0.01
    batch_size = 64
    iterations = 5000
    print_every = 10
    
    net = ln.nn.VPNN(in_channels, hidden_channels, out_channels, ind, outd, layers)
    args = {
        'data': data,
        'net': net,
        'criterion': 'z_loss',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output(data = False)
    
if __name__ == '__main__':
    main()

