import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np


import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num, set_logger


from bisect import bisect_right
import time
import math

from losses.cifar_sup_layer_mcl_meta_loss import Sup_MCL_Loss_Meta
from losses.meta_optimizers import MetaSGD
from dataset.class_sampler import MPerClassSampler


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet32_n', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight deacy')
parser.add_argument('--lr-type', default='cosine', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 225], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--times', type=int, default=2, help='times of meta optimization')
parser.add_argument('--meta-freq', type=int, default=50, help='frequency of meta optimization')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet32.pth.tar', type=str, help='resume checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--evaluate-checkpoint', default='./checkpoint/resnet32_best.pth.tar', type=str, help='evaluate checkpoint')
parser.add_argument('--number-net', type=int, default=2, help='number of networks')
parser.add_argument('--logit-distill', action='store_true', help='combine with logit distillation')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')


parser.add_argument('--kd_T', type=float, default=3, help='temperature of KL-divergence')
parser.add_argument('--tau', default=0.1, type=float, help='temperature for contrastive distribution')
parser.add_argument('--alpha', type=float, default=0.1, help='weight balance for VCL')
parser.add_argument('--gamma', type=float, default=1., help='weight balance for Soft VCL')
parser.add_argument('--beta', type=float, default=0.1, help='weight balance for ICL')
parser.add_argument('--lam', type=float, default=1., help='weight balance for Soft ICL')
parser.add_argument('--feat-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--number-stage', default=3, type=int, help='feature dimension')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch' + '_' +  args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed)

args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


log_txt =  os.path.join(args.checkpoint_dir, args.log_dir +'.txt')

logger = set_logger(log_txt)
logger.info("==========\nArgs:{}\n==========".format(args))


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)


num_classes = 100
trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                  [0.2675, 0.2565, 0.2761])
                                         ]))

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                 [0.2675, 0.2565, 0.2761]),
                                        ]))


trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                          sampler=MPerClassSampler(labels=trainset.targets, m=2, batch_size=args.batch_size, length_before_new_iter=len(trainset.targets)),
                                          pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                     pin_memory=(torch.cuda.is_available()))


logger.info('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes, number_net=args.number_net)
net.eval()
resolution = (1, 3, 32, 32)
logger.info('Arch: %s, Params: %.2fM, FLOPs: %.2fG'
        % (args.arch, cal_param_size(net)/1e6, cal_multi_adds(net, resolution)/1e9))
del(net)


net = model(num_classes=num_classes, number_net=args.number_net).cuda()
net = torch.nn.DataParallel(net)

cudnn.benchmark = True


def inner_objective(data):
    inputs, targets = data[0], data[1]
    
    embeddings, logits = net(inputs)

    weights = LossWeightNetwork(embeddings)

    loss_vcl, loss_soft_vcl, loss_icl, loss_soft_icl = criterion_mcl(embeddings, targets, weights)
    loss_mcl = args.alpha * loss_vcl + args.gamma * loss_soft_vcl \
            + args.beta * loss_icl + args.lam * loss_soft_icl
               
    return loss_mcl
    
def outer_objective(data):
    inputs, targets = data[0], data[1]
    embeddings, logits = net(inputs)
    
    loss_cls = torch.tensor(0.).cuda()
    for i in range(len(logits)):
        for j in range(args.number_stage):
            loss_cls = loss_cls + criterion_ce(logits[i][j], targets)
    return loss_cls
    
    
def train(epoch, criterion_list, aggregator, meta_optimizer, weight_optimizer):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_logit_kd = AverageMeter('train_loss_logit_kd', ':.4e')
    train_loss_vcl = AverageMeter('train_loss_vcl', ':.4e')
    train_loss_icl = AverageMeter('train_loss_icl', ':.4e')
    train_loss_soft_vcl = AverageMeter('train_loss_soft_vcl', ':.4e')
    train_loss_soft_icl = AverageMeter('train_loss_soft_icl', ':.4e')

    top1_num = [[0 for _ in range(args.number_stage)] for _ in range(args.number_net)]
    top5_num = [[0 for _ in range(args.number_stage)] for _ in range(args.number_net)]
    ens_top1_num = [0 for _ in range(args.number_net)]
    ens_top5_num = [0 for _ in range(args.number_net)]
    total = 0

    lr = adjust_lr(meta_optimizer, epoch, args)
    lr = adjust_lr(weight_optimizer, epoch, args)
    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_mcl = criterion_list[2]
    aggregator = criterion_list[3]

    net.train()
    criterion_mcl.train()
    aggregator.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_start_time = time.time()
        inputs = inputs.float().cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        embeddings, logits = net(inputs)
        
        with torch.no_grad():
            weights = LossWeightNetwork(embeddings)

        loss_cls = torch.tensor(0.).cuda()
        loss_logit_kd = torch.tensor(0.).cuda()

        for i in range(len(logits)):
            for j in range(args.number_stage):
                loss_cls = loss_cls + criterion_ce(logits[i][j], targets)

        aggregated_logits = aggregator(embeddings, logits)
        for i in range(len(logits)):
            loss_cls = loss_cls + criterion_ce(aggregated_logits[i], targets)

        for i  in range(len(logits)):
            for j in range(len(logits)):
                loss_logit_kd += criterion_div(logits[i][-1], aggregated_logits[j].detach())


        loss_vcl, loss_soft_vcl, loss_icl, loss_soft_icl = criterion_mcl(embeddings, targets, weights)
        loss_mcl = args.alpha * loss_vcl + args.gamma * loss_soft_vcl \
               + args.beta * loss_icl + args.lam * loss_soft_icl
        
        loss = loss_cls + loss_logit_kd + loss_mcl

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_logit_kd.update(loss_logit_kd.item(), inputs.size(0))
        train_loss_vcl.update(args.alpha * loss_vcl.item(), inputs.size(0))
        train_loss_soft_vcl.update(args.gamma * loss_soft_vcl.item(), inputs.size(0))
        train_loss_icl.update(args.beta * loss_icl.item(), inputs.size(0))
        train_loss_soft_icl.update(args.lam * loss_soft_icl.item(), inputs.size(0))


        for i in range(len(logits)):
            for j in range(args.number_stage):
                top1, top5 = correct_num(logits[i][j], targets, topk=(1, 5))
                top1_num[i][j] += top1
                top5_num[i][j] += top5
            top1, top5 = correct_num(aggregated_logits[i], targets, topk=(1, 5))
            ens_top1_num[i] += top1
            ens_top5_num[i] += top5
        total += targets.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (top1_num[0][-1]/total).item()))

        if batch_idx % args.meta_freq == 0:
            data = (inputs, targets)
            for _ in range(args.times):
                meta_optimizer.zero_grad()
                meta_optimizer.step(inner_objective, data)

            meta_optimizer.zero_grad()
            meta_optimizer.step(outer_objective, data)

            meta_optimizer.zero_grad()
            weight_optimizer.zero_grad()
            outer_objective(data).backward()
            meta_optimizer.meta_backward()
            weight_optimizer.step()

    acc1 = []
    acc5 = []
    for i in range(args.number_net):
        sub_acc1 = []
        for j in range(args.number_stage):
            sub_acc1.append(round((top1_num[i][j]/total).item(), 4))
        acc1.append(sub_acc1)

    for i in range(args.number_net):
        sub_acc5 = []
        for j in range(args.number_stage):
            sub_acc5.append(round((top5_num[i][j]/total).item(), 4))
        acc5.append(sub_acc5)
    
    ens_acc1 = []
    for i in range(args.number_net):
        ens_acc1.append(round((ens_top1_num[i]/total).item(), 4))

    logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                '\n Train_loss:{:.5f}'
                '\t Train_loss_cls:{:.5f}'
                '\t Train_loss_logit_kd:{:.5f}'
                '\t Train_loss_vcl:{:.5f}'
                '\t Train_loss_soft_vcl:{:.5f}'
                '\t Train_loss_icl:{:.5f}'
                '\t Train_loss_soft_icl:{:.5f}'
                '\n Train top-1 accuracy: {} \n'
                'Ensemble top-1 accuracy: {}'
                .format(epoch, lr, time.time() - start_time,
                        train_loss.avg,
                        train_loss_cls.avg,
                        train_loss_logit_kd.avg,
                        train_loss_vcl.avg,
                        train_loss_soft_vcl.avg,
                        train_loss_icl.avg,
                        train_loss_soft_icl.avg,
                        str(acc1),
                        str(ens_acc1)))
    
    

def test(epoch, criterion_ce):
    global best_accs
    test_loss_cls = AverageMeter('test_loss_cls', ':.4e')

    top1_num = [[0 for _ in range(args.number_stage)] for _ in range(args.number_net)]
    top5_num = [[0 for _ in range(args.number_stage)] for _ in range(args.number_net)]
    total = 0
    
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_start_time = time.time()
            inputs, targets = inputs.cuda(), targets.cuda()
            _, logits = net(inputs)

            loss_cls = 0.
            for i in range(len(logits)):
                for j in range(args.number_stage):
                    loss_cls = loss_cls + criterion_ce(logits[i][j], targets)

            test_loss_cls.update(loss_cls, inputs.size(0))

            for i in range(len(logits)):
                for j in range(args.number_stage):
                    top1, top5 = correct_num(logits[i][j], targets, topk=(1, 5))
                    top1_num[i][j] += top1
                    top5_num[i][j] += top5
            total += targets.size(0)

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}'.format(epoch, batch_idx, len(testloader), time.time()-batch_start_time))
        
        acc1 = []
        acc5 = []
        for i in range(args.number_net):
            sub_acc1 = []
            for j in range(args.number_stage):
                sub_acc1.append(round((top1_num[i][j]/total).item(), 4))
            acc1.append(sub_acc1)

        for i in range(args.number_net):
            sub_acc5 = []
            for j in range(args.number_stage):
                sub_acc5.append(round((top5_num[i][j]/total).item(), 4))
            acc5.append(sub_acc5)

        logger.info('Test epoch:{}\t Test_loss_cls:{:.5f}\t Test top-1 accuracy:{}'
                    .format(epoch, test_loss_cls.avg, str(acc1)))


    max_acc1 = 0.
    for i in range(args.number_net):
        max_acc1 = max(max_acc1, max(acc1[i]))
        best_accs[i] = max(best_accs[i], max(acc1[i]))
    return max_acc1
    

if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    best_accs = [0. for i in range(args.number_net)]
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_ce = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)

    if args.evaluate:      
        logger.info('Evaluate pre-trained weights from: {}'.format(args.evaluate_checkpoint))
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_ce)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)

        data = torch.randn(1, 3, 32, 32).cuda()
        net.eval()
        embedding, logits = net(data)
        args.number_stage = len(logits[0])
        
        args.rep_dim = []
        for i in range(args.number_net):
            args.rep_dim.append(embedding[i][0].size(1))

        args.feat_dims = []
        for i in range(args.number_net):
            sub_dims = []
            for j in range(args.number_stage):
                sub_dims.append(embedding[i][j].size(1))
            args.feat_dims.append(sub_dims)
                
        LossWeightNetwork = getattr(models, 'LossWeightNetwork')(args.feat_dims).cuda()
        
        criterion_mcl = Sup_MCL_Loss_Meta(args).cuda()
        trainable_list.append(criterion_mcl)
        aggregator = getattr(models, 'Aggregator')(dim_in=args.rep_dim, number_stage=args.number_stage, number_net=args.number_net).cuda()
        trainable_list.append(aggregator)
        trainable_list.cuda()
        
        target_params = list(net.parameters()) + list(criterion_mcl.parameters())
        meta_optimizer = MetaSGD(target_params,
                                   [net, criterion_mcl],
                                   lr=args.init_lr,
                                   momentum=0.9,
                                   weight_decay=args.weight_decay, 
                                   rollback=True, cpu=args.times>2)
        
        weight_optimizer = optim.SGD(LossWeightNetwork.parameters(),
                                    lr=args.init_lr,
                                    momentum=0.9, 
                                    weight_decay=args.weight_decay, nesterov=True)
        
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)


        criterion_mcl = torch.nn.DataParallel(criterion_mcl)
        aggregator = torch.nn.DataParallel(aggregator)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_ce)
        criterion_list.append(criterion_div)  
        criterion_list.append(criterion_mcl) 
        criterion_list.append(aggregator)
        criterion_list.cuda()

        if args.resume:
            logger.info('Resume pre-trained weights from: {}'.format(args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
            weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, aggregator, meta_optimizer, weight_optimizer)
            acc = test(epoch, criterion_ce)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'weight_optimizer': weight_optimizer.state_dict(),
                'meta_optimizer': meta_optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        logger.info('Evaluate the best model:')
        logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_ce)

        logger.info('Test top-1 best_accuracy: {}'.format(top1_acc))
        logger.info('Test top-1 best_accuracies: {}'.format(str(best_accs)))




