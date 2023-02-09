import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.append('..')
from .lmcl_resnet_cifar import lmcl_resnet56_cifar, lmcl_resnet110_cifar, lmcl_resnet32_cifar
from .lmcl_wrn_cifar import lmcl_wrn_16_2_cifar, lmcl_wrn_40_2_cifar, lmcl_wrn_28_4_cifar
from .lmcl_shufflenetv2_cifar import lmcl_ShuffleNetV2_1x_cifar

__all__ = ['lmcl_res32_res110_cifar', 'lmcl_res56_res110_cifar', 
           'lmcl_wrn_16_2_wrn_40_2_cifar', 'lmcl_wrn_40_2_wrn_28_4_cifar',
           'lmcl_res56_wrn_40_2_cifar', 'lmcl_res110_wrn_28_4_cifar',
           'lmcl_shufflev2_res110_cifar', 'lmcl_shufflev2_wrn_40_2_cifar']
            

class CrossNets(nn.Module):
    def __init__(self, net1, net2, num_classes=100):
        super(CrossNets, self).__init__()
        self.net1 = net1(num_classes=num_classes, number_net=1)
        self.net2 = net2(num_classes=num_classes, number_net=1)

    def forward(self, x):
        embedding1, logit1 = self.net1(x)
        embedding2, logit2 = self.net2(x)
        return [embedding1[0], embedding2[0]], [logit1[0], logit2[0]]


def lmcl_shufflev2_res110_cifar(num_classes=100, number_net=1):
    return CrossNets(lmcl_ShuffleNetV2_1x_cifar, lmcl_resnet110_cifar, num_classes=num_classes)

def lmcl_shufflev2_wrn_40_2_cifar(num_classes=100, number_net=1):
    return CrossNets(lmcl_ShuffleNetV2_1x_cifar, lmcl_wrn_40_2_cifar, num_classes=num_classes)

def lmcl_res32_res110_cifar(num_classes=100, number_net=1):
    return CrossNets(lmcl_resnet32_cifar, lmcl_resnet110_cifar, num_classes=num_classes)

def lmcl_res56_res110_cifar(num_classes=100, number_net=1):
    return CrossNets(lmcl_resnet56_cifar, lmcl_resnet110_cifar, num_classes=num_classes)


def lmcl_wrn_16_2_wrn_40_2_cifar(num_classes=100, number_net=1):
    return CrossNets(lmcl_wrn_16_2_cifar, lmcl_wrn_40_2_cifar, num_classes=num_classes)

def lmcl_wrn_40_2_wrn_28_4_cifar(num_classes=100, number_net=1):
    return CrossNets(lmcl_wrn_40_2_cifar, lmcl_wrn_28_4_cifar, num_classes=num_classes)

def lmcl_res56_wrn_40_2_cifar(num_classes=100, number_net=1):
    return CrossNets(lmcl_resnet56_cifar, lmcl_wrn_40_2_cifar, num_classes=num_classes)

def lmcl_res110_wrn_28_4_cifar(num_classes=100, number_net=1):
    return CrossNets(lmcl_resnet110_cifar, lmcl_wrn_28_4_cifar, num_classes=num_classes)


if __name__ == '__main__':
    import torch
    x = torch.randn(2, 3, 32, 32)
    net = lmcl_res110_wrn_28_4_cifar(num_classes=100)
    logit, ss_logits = net(x)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))



