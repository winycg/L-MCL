
# Online Knowledge Distillation via Mutual
Contrastive Learning for Visual Recognition

This project provides part source code for our Online Knowledge Distillation via Mutual Contrastive Learning for Visual Recognition.


## Installation

### Requirements

Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.7.0

NCCL for CUDA 11.1


## CIFAR-100 Results
### Dataset preparation
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

### Train baseline networks
```
python main_cifar_baseline.py --arch [network_name]
```
- --arch, your network name, such as `resnet32_cifar`, `wrn_40_2_cifar`, `ShuffleNetV2_1x_cifar`, `hcgnet_A1_cifar`


### Train two networks by L-MCL
```
python main_layer_mcl_cifar_meta.py --arch [network_pair_name] \
```
- --arch, your network pair name, such as `lmcl_resnet32_cifar`, `lmcl_wrn_40_2_cifar`, `lmcl_ShuffleNetV2_1x_cifar`, `lmcl_hcgnet_A1_cifar`, `lmcl_shufflev2_res110_cifar`, `lmcl_shufflev2_wrn_40_2_cifar`

###  Results on the homogeneous network pair
| Network | Baseline | L-MCL |
|:---------------:|:-----------------:|:-----------------:|
| ResNet-32 | 70.91 | 75.82 |
| ResNet-56 | 73.15 | 77.51 |
| ResNet-110 | 75.29 | 79.48 |
| WRN-16-2 | 72.55 | 77.31 |
| WRN-40-2 | 76.89 | 80.96 |
| ShuffleNetV2 0.5× |67.39 | 72.61 |
| ShuffleNetV2 1× | 70.93 | 77.25 |
| HCGNet-A1 | 77.42 | 81.66 |
| HCGNet-A2 | 79.00 | 83.14 |

###  Results on the heterogeneous network pair
| Net1 & Net2 | Baseline | L-MCL |
|:---------------:|:-----------------:|:-----------------:|
| ResNet-32&ResNet-110 | 70.91&75.29 | 75.82&79.71 |
| ResNet-56&ResNet-110 | 73.15&75.29 | 77.04&79.56 |
| WRN-16-2&WRN-40-2 | 72.55&76.89 | 76.88&80.24 |
| WRN-40-2&WRN-28-4 | 76.89&79.17 | 80.42&82.61 |
| ResNet-56&WRN-40-2 | 73.15&76.89 | 77.54&80.72 |
| ResNet-110&WRN-28-4 | 75.29&79.17 | 80.02&82.84 |
| ShuffleNetV2&ResNet-110 |70.93&75.29 | 76.85&79.17 |
| ShuffleNetV2&WRN-40-2 | 70.93&76.89 | 77.06&80.29 |



