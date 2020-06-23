#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:47:57 2020

@author: zen
"""

import torch.nn as nn
from .module import StructureNN
from .fnn import FNN
from .cnn import CNN

class CNN_FNN(StructureNN):
    '''CNN + FNN.
    '''
    def __init__(self, in_channels, channels_hidden, out_channels, ind, outd, 
                 kernel_size = 3, pad = 1, c_layers=3, f_layers = 2, f_width=180, 
                 dropout = 0, softmax=False, batch_norm = False):
        super(CNN_FNN, self).__init__()
        self.in_channels = in_channels
        self.channels_hidden = channels_hidden
        self.out_channels = out_channels
        self.ind = ind
        self.outd = outd
        self.kernel_size = kernel_size
        self.pad = pad
        self.c_layers = c_layers
        self.f_layers = f_layers
        self.f_width = f_width
        self.dropout = dropout
        self.softmax = softmax
        self.batch_norm = batch_norm
        
        self.modus = self.__init_modules()
        
    def forward(self, x):
        x = self.modus['CNN'](x)
        x = x.view([x.shape[0], -1])
        x = self.modus['FNN'](x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=-1)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        modules['CNN'] = CNN(self.in_channels, self.channels_hidden, self.out_channels, 
                 kernel_size = self.kernel_size, pad = self.pad, layers= self.c_layers,
                 batch_norm = self.batch_norm)
        modules['FNN'] = FNN(self.ind, self.outd, layers= self.f_layers, width = self.f_width,
                             dropout = self.dropout)
        return modules