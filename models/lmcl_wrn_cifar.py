import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['lmcl_wrn_16_2_cifar', 'lmcl_wrn_40_2_cifar', 'lmcl_wrn_28_4_cifar']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, is_feat=True):
        feats = []
        out = self.conv1(x)
        out = self.block1(out)
        feats.append(out)
        out = self.block2(out)
        feats.append(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        embedding = out
        feats.append(embedding)
        out = self.fc(out)
        return feats, out


class Auxiliary_Classifier(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(Auxiliary_Classifier, self).__init__()
        self.nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        block = BasicBlock  
        n = (depth - 4) // 6
        self.block_extractor1 = nn.Sequential(*[NetworkBlock(n, self.nChannels[1], self.nChannels[2], block, 2),
                                                NetworkBlock(n, self.nChannels[2], self.nChannels[3], block, 2),])
        self.block_extractor2 = nn.Sequential(*[NetworkBlock(n, self.nChannels[2], self.nChannels[3], block, 2)])
        
        self.bn1 = nn.BatchNorm2d(self.nChannels[3])
        self.bn2 = nn.BatchNorm2d(self.nChannels[3])

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.nChannels[3], num_classes)
        self.fc2 = nn.Linear(self.nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                               
    def forward(self, x):
        ss_logits = []
        ss_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor'+str(idx))(x[i])
            out = self.relu(getattr(self, 'bn'+str(idx))(out))
            out = self.avg_pool(out)
            out = out.view(-1, self.nChannels[3])
            ss_feats.append(out)
            out = getattr(self, 'fc'+str(idx))(out)
            ss_logits.append(out)
        return ss_feats, ss_logits



class WideResNet_Auxiliary(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_Auxiliary, self).__init__()
        self.backbone = WideResNet(depth, num_classes, widen_factor=widen_factor)
        self.auxiliary_classifier = Auxiliary_Classifier(depth=depth, num_classes=num_classes, widen_factor=widen_factor)
        
    def forward(self, x):
        feats, logit = self.backbone(x, is_feat=True)
        ss_feats, ss_logits  = self.auxiliary_classifier(feats[:-1])
        ss_feats.append(feats[-1])
        ss_logits.append(logit)
        return ss_feats, ss_logits


class WideResNet_n(nn.Module):
    def __init__(self, depth=16, widen_factor=2, num_classes=100, number_net=2):
        super(WideResNet_n, self).__init__()
        self.number_net = number_net

        self.module_list = nn.ModuleList([])
        for i in range(number_net):
            self.module_list.append(WideResNet_Auxiliary(num_classes=num_classes,
                                           depth=depth, widen_factor=widen_factor))

    def forward(self, x):
        logits = []
        embeddings = []
        for i in range(self.number_net):
            embedding, logit = self.module_list[i](x)
            logits.append(logit)
            embeddings.append(embedding)
        return embeddings, logits


def lmcl_wrn_16_2_cifar(num_classes, number_net):
    return WideResNet_n(depth=16, widen_factor=2, num_classes=num_classes, number_net=number_net)


def lmcl_wrn_40_2_cifar(num_classes, number_net):
    return WideResNet_n(depth=40, widen_factor=2, num_classes=num_classes, number_net=number_net)


def lmcl_wrn_28_4_cifar(num_classes, number_net):
    return WideResNet_n(depth=28, widen_factor=4, num_classes=num_classes, number_net=number_net)


if __name__ == '__main__':
    import torch
    x = torch.randn(2, 3, 32, 32)
    #net = wrn_16_2(num_classes=100)
    net = wrn_16_2_aux(num_classes=100,  number_net=2)
    logit, ss_logits = net(x)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))