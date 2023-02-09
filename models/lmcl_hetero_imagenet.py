import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.append('..')
from .lmcl_resnet_imagenet import lmcl_resnet18_imagenet, lmcl_resnet50_imagenet
from .lmcl_shufflenetv2_imagenet import lmcl_ShuffleNetV2_1x_imagenet


__all__ = ['lmcl_res18_res50_imagenet', 'lmcl_res18_shufflenetv2_1x_imagenet',
           'lmcl_res50_shufflenetv2_1x_imagenet']
            

class CrossNets(nn.Module):
    def __init__(self, net1, net2, num_classes=100):
        super(CrossNets, self).__init__()
        self.net1 = net1(num_classes=num_classes, number_net=1)
        self.net2 = net2(num_classes=num_classes, number_net=1)

    def forward(self, x):
        embedding1, logit1 = self.net1(x)
        embedding2, logit2 = self.net2(x)
        return [embedding1[0], embedding2[0]], [logit1[0], logit2[0]]


def lmcl_res18_res50_imagenet(num_classes=100, number_net=1):
    return CrossNets(lmcl_resnet18_imagenet, lmcl_resnet50_imagenet, num_classes=num_classes)

def lmcl_res18_shufflenetv2_1x_imagenet(num_classes=100, number_net=1):
    return CrossNets(lmcl_resnet18_imagenet, lmcl_ShuffleNetV2_1x_imagenet, num_classes=num_classes)

def lmcl_res50_shufflenetv2_1x_imagenet(num_classes=100, number_net=1):
    return CrossNets(lmcl_ShuffleNetV2_1x_imagenet, lmcl_resnet50_imagenet, num_classes=num_classes)



if __name__ == '__main__':
    net = res32_res56()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    


