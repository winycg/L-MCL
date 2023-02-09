import torch
import torch.nn as nn
import torch.nn.functional as F

class LossWeightNetwork(nn.Module):
    def __init__(self, number_features):
        super(LossWeightNetwork, self).__init__()
        self.proj = nn.ModuleList([])
        self.number_features = number_features
        self.number_net = len(number_features)
        for i in range(self.number_net):
            sub_proj = nn.ModuleList([])
            for j in  range(len(number_features[i])):
                sub_proj.append(nn.Linear(number_features[i][j], 128))
            self.proj.append(sub_proj)

    def similarity(self, f_a, f_b):
        f_a = F.normalize(f_a, dim=1)
        f_b = F.normalize(f_b, dim=1)
        sim = (f_a * f_b).sum(dim=1)
        sim = torch.sigmoid(sim)
        return sim
        
        
    def forward(self, features):
        projected_feats = []
        for i in range(self.number_net):
            sub_proj = []
            for j in range(len(features[i])):
                sub_proj.append(self.proj[i][j](features[i][j]))
            projected_feats.append(sub_proj)
        
        weights = []
        for i in range(self.number_net):
            for j in range(self.number_net):
                if i == j:
                    continue
                else:
                    for k in range(len(projected_feats[i])):
                        for z in range(len(projected_feats[j])):
                            weights.append(self.similarity(projected_feats[i][k], projected_feats[j][z]))

        return weights