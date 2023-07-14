import torch
from torch import nn
import math
import torch.nn.functional as F

class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """

    def __init__(self, args):
        super(ContrastMemory, self).__init__()
        self.number_net = args.number_net
        self.feat_dim = args.feat_dim
        self.n_data = args.n_data
        self.args = args
        self.momentum = args.nce_m
        self.kl = KLDiv(args.kd_T)

        stdv = 1. / math.sqrt(self.feat_dim / 3)
        for i in range(args.number_net):
            self.register_buffer('memory_' + str(i), torch.rand(args.n_data, args.feat_dim).mul_(2 * stdv).add_(-stdv))

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def forward(self, embeddings_a, embeddings_b, a, b, pos_idx, neg_idx, weight):
        batchSize = embeddings_a.size(0)
        idx = torch.cat([pos_idx.cuda(), neg_idx.cuda()], dim=1)
        K = self.args.pos_k + self.args.neg_k + 1

        inter_logits = []
        soft_icl_loss = 0.

        neg_rep = torch.index_select(getattr(self, 'memory_' + str(a)), 0, idx.view(-1)).detach()
        neg_rep = neg_rep.view(batchSize, K, self.feat_dim)
        cos_simi_ij = torch.div(
            torch.bmm(neg_rep, embeddings_b.view(batchSize, self.feat_dim, 1)).squeeze(-1),
            self.args.tau)
        inter_logits.append(cos_simi_ij)

        neg_rep = torch.index_select(getattr(self, 'memory_' + str(b)), 0, idx.view(-1)).detach()
        neg_rep = neg_rep.view(batchSize, K, self.feat_dim)
        cos_simi_ji = torch.div(
            torch.bmm(neg_rep, embeddings_a.view(batchSize, self.feat_dim, 1)).squeeze(-1),
            self.args.tau)
        inter_logits.append(cos_simi_ji)

        soft_icl_loss += self.kl(cos_simi_ij, cos_simi_ji.detach(), weight)
        soft_icl_loss += self.kl(cos_simi_ji, cos_simi_ij.detach(), weight)


        mask = torch.zeros(batchSize, 1 + self.args.pos_k + self.args.neg_k).cuda()
        mask[:, :self.args.pos_k+1] = 1.
        icl_loss = 0.
        for logit in inter_logits:
            log_prob = logit - torch.log(torch.exp(logit).sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob * weight).sum(1) / mask.sum(1)
            icl_loss += - mean_log_prob_pos.mean()


        intra_logits = []
        idx = idx[:, 1:].contiguous()
        K = self.args.pos_k + self.args.neg_k


        neg_rep_a = torch.index_select(getattr(self, 'memory_' + str(a)), 0, idx.view(-1)).detach()
        neg_rep_a = neg_rep_a.view(batchSize, K, self.feat_dim)
        cos_simi = torch.div(
            torch.bmm(neg_rep_a, embeddings_a.view(batchSize, self.feat_dim, 1)).squeeze(-1),
            self.args.tau)
        intra_logits.append(cos_simi)

        neg_rep_b = torch.index_select(getattr(self, 'memory_' + str(a)), 0, idx.view(-1)).detach()
        neg_rep_b = neg_rep_b.view(batchSize, K, self.feat_dim)
        cos_simi = torch.div(
            torch.bmm(neg_rep_b, embeddings_b.view(batchSize, self.feat_dim, 1)).squeeze(-1),
            self.args.tau)
        intra_logits.append(cos_simi)

        soft_vcl_loss = 0.
        soft_vcl_loss += self.kl(intra_logits[0], intra_logits[1].detach(), weight)
        soft_vcl_loss += self.kl(intra_logits[1], intra_logits[0].detach(), weight)

        vcl_loss = 0.
        mask = torch.zeros(batchSize, self.args.pos_k + self.args.neg_k).cuda()
        mask[:, :self.args.pos_k] = 1.
        for logit in intra_logits:
            log_prob = logit - torch.log(torch.exp(logit).sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob * weight).sum(1) / mask.sum(1)
            vcl_loss += - mean_log_prob_pos.mean()

        return vcl_loss, soft_vcl_loss, icl_loss, soft_icl_loss


    def update_memory(self, embeddings, pos_idx):
        pos_idx = pos_idx[:, 0].contiguous()
        pos_idx = self.concat_all_gather(pos_idx)

        with torch.no_grad():
            for i in range(len(embeddings)):
                pos = torch.index_select(getattr(self, 'memory_' + str(i)), 0, pos_idx.view(-1))
                pos.mul_(self.momentum)
                embeddings[i] = self.concat_all_gather(embeddings[i])
                pos.add_(torch.mul(embeddings[i], 1 - self.momentum))
                l_norm = pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_v = pos.div(l_norm)
                getattr(self, 'memory_' + str(i)).index_copy_(0, pos_idx, updated_v)


class Sup_MCL_Loss_Meta(nn.Module):
    def __init__(self, args):
        super(Sup_MCL_Loss_Meta, self).__init__()
        self.embed_list = nn.ModuleList([])
        self.args = args
        for i in range(args.number_net):
            sub_embed_list = nn.ModuleList([])
            for j in range(args.number_stage):
                sub_embed_list.append(Embed(args.rep_dim[i], args.feat_dim))
            self.embed_list.append(sub_embed_list)
        self.contrast = ContrastMemory(args)

    def forward(self, embeddings, pos_idx, neg_idx, weights):
        for i in range(self.args.number_net):
            for j in range(self.args.number_stage):
                embeddings[i][j] = self.embed_list[i][j](embeddings[i][j])
        last_embeddings = [embeddings[i][-1] for i in range(self.args.number_net)]

        vcl_loss_all = 0.
        soft_vcl_loss_all = 0.
        icl_loss_all = 0.
        soft_icl_loss_all = 0.
        
        count = 0
        for i in range(self.args.number_net):
            for j in range(self.args.number_net):
                if i == j:
                    continue
                else:
                    for k in range(len(embeddings[i])):
                        for z in range(len(embeddings[j])):
                            weight = weights[count].unsqueeze(-1)
                            vcl_loss, soft_vcl_loss, icl_loss, soft_icl_loss = \
                                self.contrast(embeddings[i][k], embeddings[j][z], i, j, pos_idx, neg_idx, weight)
                            
                            count += 1
                            vcl_loss_all += vcl_loss
                            soft_vcl_loss_all += soft_vcl_loss
                            icl_loss_all +=  icl_loss
                            soft_icl_loss_all +=  soft_icl_loss

        self.contrast.update_memory(last_embeddings, pos_idx)
        return vcl_loss_all, soft_vcl_loss_all, icl_loss_all, soft_icl_loss_all
    


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.proj_head(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



class KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KLDiv, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, weights=None):
        log_p_s = F.log_softmax(y_s/self.T, dim=1)
        log_p_t = F.log_softmax(y_t/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        input_size = p_t.size(0)
        if weights is not None:
            loss = (self.T**2) * (weights * p_t * (log_p_t - log_p_s)).sum() / input_size
        else:
            loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.T**2)

        return loss
