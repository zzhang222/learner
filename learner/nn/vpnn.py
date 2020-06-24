#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:29:24 2020

@author: zen
"""

from .cnn import CNN
from .fnn import FNN
from .module import Module, StructureNN
from ..utils import checkerboard_mask, squeeze_2x2
from enum import IntEnum
import torch.nn as nn
import torch

class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1
    DOUBLE_CHANNEL = 2 # Put two same images together
    
class VPModule_C(Module):
    def __init__(self, in_channels, channels_hidden, out_channels,
                kernel_size = 3, pad = 1, layers= 3,
                batch_norm = True, mask_type = MaskType.CHECKERBOARD, reverse_mask = False):
        super(VPModule_C, self).__init__()
        self.in_channels = in_channels
        self.channels_hidden = channels_hidden
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.layers = layers
        self.batch_norm = batch_norm
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask
        self.modus = self.__init_modules()
        
    def forward(self, x, reverse = False):
        if self.mask_type == MaskType.CHECKERBOARD:
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
            x_b = x * b
            t = self.modus['CNN'](x_b)
            t = t * (1 - b)
            if reverse:
                x = x - t
            else:
                x = x + t
            return x
                
    def __init_modules(self):
        modules = nn.ModuleDict()
        modules['CNN'] = CNN(self.in_channels, self.channels_hidden, self.out_channels, 
                 kernel_size = self.kernel_size, pad = self.pad, layers= self.layers,
                 batch_norm = self.batch_norm)
        return modules
    
class VPModule_F(Module):
    def __init__(self, ind, outd, layers = 2, width = 180):
        super(VPModule_F, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.modus = self.__init_modules()
        
    def forward(self, x, reverse = False):
        x1, x2 = (x.narrow(1, 0, self.ind),
            x.narrow(1, self.ind, self.ind))
        if reverse:
            t2 = self.modus['FNN2'](x2)
            x1 = x1 - t2
            t1 = self.modus['FNN1'](x1)
            x2 = x2 - t1
        else:
            t1 = self.modus['FNN1'](x1)
            x2 = x2 + t1
            t2 = self.modus['FNN2'](x2)
            x1 = x1 + t2
        return torch.cat((x1, x2), 1)
                
    def __init_modules(self):
        modules = nn.ModuleDict()
        modules['FNN1'] = FNN(self.ind, self.outd, layers=self.layers, width=self.width)
        modules['FNN2'] = FNN(self.ind, self.outd, layers=self.layers, width=self.width)
        return modules
    
                
class VPNN(StructureNN):
    #Volumn Preserving Neural Networks (NICE)
    def __init__(self, in_channels, channels_hidden, out_channels, ind, outd, layers_c, layers_f):
        super(VPNN, self).__init__()
        self.in_channels = in_channels
        self.channels_hidden = channels_hidden
        self.out_channels = out_channels
        self.ind = ind
        self.outd = outd
        self.layers_c = layers_c
        self.layers_f = layers_f
        self.modus = self.__init_modules()
            
        
    def forward(self, x, reverse = False):
        shape = x.shape
        if reverse:
            x = x.flatten(start_dim = 1)
            for i in range(self.layers_f, 0, -1):
                x = self.modus['VPF{}'.format(i)](x, reverse = reverse)
            x = x.view([-1,shape[1],shape[2],shape[3]])
            x = squeeze_2x2(x, reverse = False)
            for i in range(self.layers_c, 0, -1):
                x = self.modus['VPC{}'.format(i)](x, reverse = reverse)
            x = squeeze_2x2(x, reverse = True)
        else:
            x = squeeze_2x2(x, reverse = False)
            for i in range(self.layers_c):
                x = self.modus['VPC{}'.format(i+1)](x, reverse = reverse)
            x = squeeze_2x2(x, reverse = True).flatten(start_dim = 1)
            for i in range(self.layers_f):
                x = self.modus['VPF{}'.format(i+1)](x, reverse = reverse)
            x = x.view([-1,shape[1],shape[2],shape[3]])
        return x
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        reverse_mask = True
        for i in range(self.layers_c):
            modules['VPC{}'.format(i+1)] = VPModule_C(self.in_channels, self.channels_hidden, self.out_channels, reverse_mask = reverse_mask)
            reverse_mask = not reverse_mask
        for i in range(self.layers_f):
            modules['VPF{}'.format(i+1)] = VPModule_F(self.ind, self.outd)
        return modules
            
            
        