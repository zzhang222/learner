#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 22:34:38 2020

@author: zen
"""
import torch.nn as nn

from .module import StructureNN

class CNN(StructureNN):
    '''Convolutional neural networks.
    '''
    def __init__(self, in_channels, channels_hidden, out_channels, 
                 kernel_size = 3, pad = 1, layers=3, activation='leaky relu',
                 initializer='Glorot uniform', batch_norm = False):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels_hidden = channels_hidden
        self.kernel_size = kernel_size
        self.pad = pad
        self.layers = layers
        self.activation = activation
        self.initializer = initializer
        self.batch_norm = batch_norm
        
        self.modus = self.__init_modules()
        self.__initialize()
        
    def forward(self, x):
        for i in range(1, self.layers):
            ConvM = self.modus['ConvM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            if self.batch_norm:
                NormM = self.modus['NormM{}'.format(i)]
                x = NonM(NormM(ConvM(x)))
            else:
                x = NonM(ConvM(x))
        x = self.modus['ConvMout'](x)
        if self.batch_norm:
            x = self.modus['NormMout'](x)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['ConvM1'] = nn.Conv2d(self.in_channels, self.channels_hidden, 
                                          kernel_size = self.kernel_size, padding = self.pad,
                                          bias = not self.batch_norm)
            modules['NonM1'] = self.Act
            if self.batch_norm:
                modules['NormM1'] = nn.BatchNorm2d(self.channels_hidden)
            for i in range(2, self.layers):
                modules['ConvM{}'.format(i)] = nn.Conv2d(self.channels_hidden, self.channels_hidden, 
                                          kernel_size = self.kernel_size, padding = self.pad,
                                          bias = not self.batch_norm)
                modules['NonM{}'.format(i)] = self.Act
                if self.batch_norm:
                    modules['NormM{}'.format(i)] = nn.BatchNorm2d(self.channels_hidden)
            modules['ConvMout'] = nn.Conv2d(self.channels_hidden, self.out_channels, 
                                          kernel_size = self.kernel_size, padding = self.pad,
                                          bias = not self.batch_norm)
            if self.batch_norm:
                modules['NormMout'] = nn.BatchNorm2d(self.out_channels)
        else:
            modules['ConvMout'] = nn.Conv2d(self.channels_hidden, self.out_channels, 
                                          kernel_size = self.kernel_size, padding = self.pad,
                                          bias = not batch_norm)
            if self.batch_norm:
                modules['NormMout'] = nn.BatchNorm2d(oself.out_channels)
            
        return modules
    
    def __initialize(self):
        for i in range(1, self.layers):
            self.weight_init_(self.modus['ConvM{}'.format(i)].weight)
            if self.batch_norm:
                self.modus['NormM{}'.format(i)].weight.data.fill_(1)
            else:
                self.modus['ConvM{}'.format(i)].bias.data.fill_(0)
        self.weight_init_(self.modus['ConvMout'].weight)
        if self.batch_norm:
            self.modus['NormMout'].weight.data.fill_(1)
        else:
            self.modus['ConvMout'].bias.data.fill_(0)
    