"""
@author: jpzxshi
"""
from .module import Module
from .module import StructureNN
from .module import LossNN
from .fnn import FNN
from .hnn import HNN
from .cnn import CNN
from .cnn_fnn import CNN_FNN
from .vpnn import VPNN
from .sympnet import LASympNet
from .sympnet import GSympNet

__all__ = [
    'Module',
    'StructureNN',
    'LossNN',
    'FNN',
    'HNN',
    'CNN',
    'CNN_FNN',
    'VPNN'
    'LASympNet',
    'GSympNet',
]


