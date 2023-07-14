
# Online Knowledge Distillation via Mutual Contrastive Learning for Visual Recognition

This project provides source code for our Online Knowledge Distillation via Mutual Contrastive Learning for Visual Recognition published at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI-2023). 

Official paper link: https://ieeexplore.ieee.org/abstract/document/10073628

ArXiv paper link: https://arxiv.org/pdf/2207.11518.pdf


## Installation

### Requirements

Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.7.0

Install Python packages：
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy==1.10.1
```


## CIFAR-100 Results

### Train baseline networks
```
python main_cifar_baseline.py --arch [network_name] \
--data [your dataset path] \
--checkpoint-dir [your checkpoint path] 
```
- --arch, your network name, such as `resnet32_cifar`, `resnet56_cifar`, `resnet110_cifar`, `wrn_16_2_cifar`, `wrn_40_2_cifar`, `ShuffleNetV2_05x_cifar`, `ShuffleNetV2_1x_cifar`, `hcgnet_A1_cifar`,`hcgnet_A2_cifar`,


### Train two networks by L-MCL
```
python main_layer_mcl_cifar_meta.py --arch [network_pair_name] \
--data [your dataset path] \
--checkpoint-dir [your checkpoint path] 
```
- --arch, your network pair name, such as homogeneous network pair: `lmcl_resnet32_cifar`, `lmcl_resnet56_cifar`, `lmcl_resnet110_cifar`,`lmcl_wrn_16_2_cifar`,`lmcl_wrn_40_2_cifar`, `lmcl_ShuffleNetV2_05x_cifar`,`lmcl_ShuffleNetV2_1x_cifar`, `lmcl_hcgnet_A1_cifar`, `lmcl_hcgnet_A2_cifar`,
and heterogeneous network pair: 
`lmcl_res32_res110_cifar`, 
`lmcl_res56_res110_cifar`, 
`lmcl_wrn_16_2_wrn_40_2_cifar`, 
`lmcl_wrn_40_2_wrn_28_4_cifar`, 
`lmcl_res56_wrn_40_2_cifar`, 
`lmcl_res110_wrn_28_4_cifar`, 
`lmcl_res56_res110_cifar`, 
`lmcl_shufflev2_res110_cifar`, `lmcl_shufflev2_wrn_40_2_cifar`

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

## ImageNet Results

### Train baseline networks
The training script is referred from https://github.com/pytorch/examples/tree/main/imagenet , which uses multi-processing distributed training to launch N processes per node, which has N GPUs.
```
python main_imagenet_baseline.py \
    --data [your dataset path] \
    --checkpoint-dir [your checkpoint path] \
    --arch [network_pair_name] \
    --dist-url 'tcp://127.0.0.1:1242' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0
```
- --arch, your network name, such as `resnet18_imagenet`, `resnet34_imagenet`, `resnet50_imagenet`, `ShuffleNetV2_1x_imagenet`


### Train two networks by L-MCL
```
python -m torch.distributed.launch --nproc_per_node=2 \
    main_layer_mcl_imagenet_meta.py \
    --data [your dataset path] \
    --checkpoint-dir [your checkpoint path]  \
    --arch [network_pair_name] \
    --master_port 11234 \
    --gpu-id 0,1,2,3,4,5,6,7 
```
- --arch, your network pair name, such as homogeneous network pair: `lmcl_resnet18_imagenet`, `lmcl_resnet34_imagenet`, `lmcl_resnet50_imagenet`,`lmcl_ShuffleNetV2_1x_imagenet`
and heterogeneous network pair: 
`lmcl_res18_shufflenetv2_1x_imagenet`, 
`lmcl_res50_shufflenetv2_1x_imagenet`, 
`lmcl_res18_res50_imagenet`

###  Results on the homogeneous network pair
| Network | Baseline | L-MCL |
|:---------------:|:-----------------:|:-----------------:|
| ResNet-34 | 73.68 | 75.12 |
| ResNet-50 | 76.28 | 78.35 |
| ShuffleNetV2 | 64.25 | 65.91 |


###  Results on the heterogeneous network pair
| Net1 & Net2 | Baseline | L-MCL |
|:---------------:|:-----------------:|:-----------------:|
| ShuffleNetV2&ResNet-18 | 64.25&69.95 | 66.06&71.16 |
| ShuffleNetV2&ResNet-50 | 64.25&76.28 | 66.44&76.57 |
| ResNet-18&ResNet-50 | 69.95&76.28 | 71.69&77.34 |

## Citation
If our work helps you, we would appreciate it if you could give this repo a star or cite our paper!
```
@article{yang2023online,
  title={Online knowledge distillation via mutual contrastive learning for visual recognition},
  author={Yang, Chuanguang and An, Zhulin and Zhou, Helong and Zhuang, Fuzhen and Xu, Yongjun and Zhang, Qian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```



