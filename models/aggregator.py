import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class Aggregator(nn.Module):
    def __init__(self, dim_in, number_stage, number_net):
        super(Aggregator, self).__init__()
        self.number_stage = number_stage
        self.number_net = number_net
        for i in range(self.number_net):
            setattr(self, 'proj_head_' + str(i), nn.Sequential(
            nn.Linear(number_stage * dim_in[i], dim_in[i]),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in[i], number_stage)
            ))

    def forward(self, embeddings, logits):
        aggregated_logits = []
        for i in range(self.number_net):
            feature = torch.cat(embeddings[i],dim=1)
            logit = getattr(self, 'proj_head_' + str(i))(feature)
            weights = F.softmax(logit, dim=1)
            weighted_logit = 0.
            for j in range(self.number_stage):
                weighted_logit += logits[i][j] * weights[:, j].unsqueeze(-1)
            aggregated_logits.append(weighted_logit)
        return aggregated_logits