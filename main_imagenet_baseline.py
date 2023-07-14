import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import os
import shutil
import argparse 
import numpy as np


import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds


from bisect import bisect_right
import time
import math
from utils import correct_num, AverageMeter, set_logger, adjust_lr


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet34_imagenet', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[30,60,90], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume-checkpoint',  type=str, default='./checkpoint/XXX.pth.tar')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory for storing checkpoint files')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')                    


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if args.rank == 0:
        logger = set_logger(args.log_txt)

    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)

    num_classes = 1000
    model = getattr(models, args.arch)
    net = model(num_classes=num_classes).cuda(args.gpu)

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net)
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=args.weight_decay)


    if args.resume:
        print('load pre-trained weights from: {}'.format(args.resume_checkpoint))     
        checkpoint = torch.load(args.resume_checkpoint,
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        print('load successfully!')


    train_set = torchvision.datasets.ImageFolder(
    args.traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    test_set = torchvision.datasets.ImageFolder(
        args.valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ]))

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    def train(epoch, criterion_list, optimizer):
        train_loss = AverageMeter('train_loss', ':.4e')
        train_loss_cls = AverageMeter('train_loss_cls', ':.4e')

        top1_num = 0
        top5_num = 0
        total = 0

        lr = adjust_lr(optimizer, epoch, args)

        start_time = time.time()
        criterion_cls = criterion_list[0]

        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_start_time = time.time()
            inputs = inputs.float().cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            logits  = net(inputs)

            loss_cls = torch.tensor(0.).cuda()
            loss_cls += criterion_cls(logits, targets)
            
            loss = loss_cls
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))
            train_loss_cls.update(loss_cls.item(), inputs.size(0))

            top1, top5 = correct_num(logits, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)

            if args.rank == 0:
                print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Train Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (top1_num/(total)).item()))

        class_acc1 = round((top1_num/(total)).item(), 4)
        class_acc5 = round((top5_num/(total)).item(), 4)

        if args.rank == 0:
            logger.info('Epoch:{}\t Lr:{:.5f}\t Duration:{:.3f}'
                        '\n Train_loss:{:.5f}\t Train_loss_cls:{:.5f}'
                        '\nTrain Top-1 accuracy: {}\nTrain Top-5 accuracy: {}'
                        .format(epoch, lr, time.time() - start_time,
                                train_loss.avg, train_loss_cls.avg,
                                str(class_acc1), str(class_acc5)))
            

    def test(epoch, criterion_cls, net):
        test_loss_cls = AverageMeter('Loss', ':.4e')
        top1_num = 0
        top5_num = 0
        total = 0
        
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                batch_start_time = time.time()
                inputs, targets = inputs.cuda(), targets.cuda()
                logits = net(inputs)
                loss_cls = torch.tensor(0.).cuda()
                loss_cls = criterion_cls(logits, targets)

                test_loss_cls.update(loss_cls.item(), inputs.size(0))

                top1, top5 = correct_num(logits, targets, topk=(1, 5))
                top1_num += top1
                top5_num += top5
                total += targets.size(0)
                if args.rank == 0:
                    print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Test Top-1 Acc:{:.4f}'.format(
                        epoch, batch_idx, len(testloader), time.time()-batch_start_time, (top1_num/(total)).item()))
            class_acc1 = round((top1_num/total).item(), 4)
            class_acc5 = round((top5_num/total).item(), 4)

            if args.rank == 0:
                logger.info('Test epoch:{}\t Test_loss_cls:{:.5f}\nTest top-1 accuracy: {}\nTest top-5 accuracy: {}'
                            .format(epoch, test_loss_cls.avg, str(class_acc1), str(class_acc5)))
        return class_acc1

    if args.evaluate: 
        if args.rank == 0:
            args.logger.info('load pre-trained weights from: {}'.format(args.resume_checkpoint))     
        checkpoint = torch.load(args.resume_checkpoint,
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        test(start_epoch, criterion_cls, net)
    else:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls, net)

            if args.rank == 0:
                state = {
                    'net': net.module.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

                is_best = False
                if best_acc < acc:
                    best_acc = acc
                    is_best = True

                if is_best:
                    shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        if args.rank == 0:
            logger.info('Evaluate the best model:')
            logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
            args.evaluate = True
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                    map_location='cuda:{}'.format(args.gpu))
            net.module.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            top1_acc = test(start_epoch, criterion_cls, net)
            logger.info('best_accuracy: {}'.format(top1_acc))


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.log_txt =  str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch' + '_' +  args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed) +'.txt'


    args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch'+ '_' + args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed)


    args.traindir = os.path.join(args.data, 'train')
    args.valdir = os.path.join(args.data, 'val')

    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.set_printoptions(precision=4)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
    args.log_txt = os.path.join(args.checkpoint_dir, args.log_txt)
    if args.rank == 0:
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        logger = set_logger(args.log_txt)
        logger.info("==========\nArgs:{}\n==========".format(args))
    
    print('==> Building model..')
    num_classes = 1000

    net = getattr(models, args.arch)(num_classes=num_classes)
    net.eval()
    if args.rank == 0:
        logger.info('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
            % (args.arch, cal_param_size(net)/1e6, cal_multi_adds(net, (1, 3, 224, 224))/1e9))
    del(net)
    
    args.distributed = args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.rank == 0:
        logger.info('ngpus_per_node:{}'.format(ngpus_per_node))
        logger.info('multiprocessing_distributed')
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))