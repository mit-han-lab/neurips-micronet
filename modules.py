import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from u import from_torch

class AdaptiveEmbedding(nn.Module):
    def __init__(self, c):
        super(AdaptiveEmbedding, self).__init__()
        self.c = c

        if c.get('n_embeds'):
            n_embeds = c.n_embeds
        else:
            c.n_embeds = n_embeds = [c.n_embed // (c.adaptive_ratio ** i) for i in range(len(c.cutoffs) + 1)]
        assert n_embeds[0] == c.n_embed

        self.layers = nn.ModuleList(
            nn.Embedding(end - start, n_embed_i) for n_embed_i, start, end in zip(
                n_embeds, [0] + c.cutoffs, c.cutoffs + [c.n_vocab]
            )
        )
        self.projections = nn.ModuleList(
            nn.Linear(n_embed_i, c.n_embed, bias=False) for n_embed_i in n_embeds[1:]
        )
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0, 0.02)
        
    def forward(self, x):
        c = self.c
        x_flat = x.reshape(-1)

        emb_flat = torch.zeros([x_flat.size(0), c.n_embed], dtype=torch.float if c.opt_level == 'O0' else torch.half, device=x.device)
        for i, (layer, start, end) in enumerate(zip(self.layers, [0] + c.cutoffs, c.cutoffs + [c.n_vocab])):
            mask_i = (x_flat >= start) & (x_flat < end)
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue

            inp_i = x_flat.index_select(0, indices_i) - start
            emb_i = layer(inp_i)
            if i == 0:
                if c.opt_level != 'O0':
                    emb_i = emb_i.half()
            else:
                emb_i = self.projections[i - 1](emb_i)
            emb_flat.index_copy_(0, indices_i, emb_i)

        emb = emb_flat.view(*x.size(), c.n_embed)
        emb.mul_(np.sqrt(c.n_embed))
        return emb

class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, c):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.c = c.setdefault(use_cache=False)
        n_layers = len(c.cutoffs) + 1

        if c.get('n_embeds'):
            n_embeds = c.n_embeds
        else:
            n_embeds = [c.n_embed // (c.adaptive_ratio ** i) for i in range(n_layers)]
        assert n_embeds[0] == c.n_embed
        assert n_layers == len(n_embeds)

        # the first layer gets (n_layers - 1) more classes to determine if token belongs in those layers
        self.clusters = nn.Linear(c.n_embed, n_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n_embed_i, end - start) for n_embed_i, start, end in zip(
                n_embeds, [0] + c.cutoffs, c.cutoffs + [c.n_vocab]
            )
        )
        self.projections = nn.ModuleList(
            nn.Linear(c.n_embed, n_embed_i, bias=False) for n_embed_i in n_embeds[1:]
        )
        self.cache_keys = self.cache_values = None

        if c.use_cache:
            if c.get('cache_theta_init'):
                theta = torch.tensor(c.cache_theta_init)
                self.cache_theta_inv_softplus = nn.Parameter((theta.exp() - 1).log())
            
            if c.get('cache_lambda_init'):
                lam = torch.tensor(c.cache_lambda_init)
                self.cache_lambda_inv_sigmoid = nn.Parameter((lam / (1 - lam)).log())

    def query_cache(self, hidden, target):
        # assume n_batch = 1
        c = self.c

        n_seq = hidden.size(0)
        cache_keys, cache_values = self.cache_keys, self.cache_values
        if cache_keys is None:
            n_cache = 0
            cache_keys = hidden
            cache_values = target
        else:
            n_cache = cache_keys.size(0)
            cache_keys = torch.cat((cache_keys.detach(), hidden))
            cache_values = torch.cat((cache_values, target))
        cache_keys = cache_keys[-(c.n_cache + n_seq):]
        cache_values = cache_values[-(c.n_cache + n_seq):]
        
        if c.get('cache_theta_init'):
            theta = F.softplus(self.cache_theta_inv_softplus)
        else:
            theta = c.cache_theta
        self.last_theta = from_torch(theta)
        attn = theta * F.pad(hidden.mm(cache_keys.t()), (c.n_cache - n_cache, 0), value=-(1e8 if c.opt_level == 'O0' else np.inf))# (n_s, n_c + n_s)
        # (n_s, n_cache)
        logprobs = attn.reshape(-1).unfold(0, c.n_cache, attn.size(1) + 1).log_softmax(dim=1)
        indices = F.pad(cache_values, (c.n_cache - n_cache, 0), value=-1).unfold(0, c.n_cache, 1)[:n_seq]

        mask = indices != target[:, None]
        logprobs = logprobs - mask.to(logprobs.dtype) * 10e8
        logprobs[torch.isnan(logprobs)] = -np.inf
        # logprobs.masked_fill_(mask, -np.inf)

        pos_logprob = logprobs.logsumexp(dim=1)
        self.cache_keys = cache_keys[-c.n_cache:]
        self.cache_values = cache_values[-c.n_cache:]
        return pos_logprob
        
    def forward(self, hidden, target, keep_order=False, soft_labels=None, soft_probs=None, is_distilling=False, current_step=0.):
        # hidden: (n_seq * n_batch, n_embed)
        # target: (n_seq * n_batch)
        c = self.c
        assert hidden.size(0) == target.size(0), 'Input and target should have the same size in the batch dimension'
        
        if c.use_cache:
            cache_logprob = self.query_cache(hidden, target)

        head_logit = torch.cat((self.layers[0](hidden), self.clusters(hidden)), dim=1)

        if c.get('gen_soft'):
            head_prob = head_logit.softmax(dim=1)
            all_prob = head_prob[:, :-len(c.cutoffs)]
            for i in range(len(c.cutoffs)):
                proj_i = self.projections[i](hidden)
                tail_logit_i = self.layers[i + 1](proj_i)
                tail_prob_i = tail_logit_i.softmax(dim=1)
                tail_prob_i = tail_prob_i * head_prob[:, -(i + 1)].unsqueeze(1)
                all_prob = torch.cat((all_prob, tail_prob_i), dim=1)
            return all_prob, None
        # print("target size", target.size())

        if c.get('distillation_teacher') == 'file' and is_distilling:
            head_logprob = head_logit.log_softmax(dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
            distill_loss = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)
            topk = soft_labels.size(1)
            hiddens = {}
            offset = 0
            for i, (start, end) in enumerate(zip([0] + c.cutoffs, c.cutoffs + [c.n_vocab])):
                mask_i = (target >= start) & (target < end)

                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                if c.use_cache:
                    cache_logprob_i = cache_logprob.index_select(0, indices_i)

                target_i = (target.index_select(0, indices_i) - start)[:, None]
                head_logprob_i = head_logprob.index_select(0, indices_i)

                soft_labels_i = (soft_labels.index_select(0, indices_i))
                soft_probs_i = (soft_probs.index_select(0, indices_i))
                soft_labels_i = soft_labels_i.reshape(-1)
                soft_probs_i = soft_probs_i.reshape(-1)

                mask_soft_labels_i = (soft_labels_i >= start) & (soft_labels_i < end)
                mask_soft_labels_i_first_bin = (soft_labels_i < c.cutoffs[0])
                # print("mask_soft_labels_i_first_bin size", i, mask_soft_labels_i_first_bin.size())


                # those in the bin need to be substracted
                masked_soft_labels_i = mask_soft_labels_i.type(soft_labels_i.type()) * (soft_labels_i - start)
                masked_soft_probs_i = mask_soft_labels_i.type(soft_probs_i.type()) * soft_probs_i

                if c.get('distill_first_bin') == 1:
                    masked_soft_labels_i_first_bin = mask_soft_labels_i_first_bin.type(soft_labels_i.type()) * soft_labels_i
                    masked_soft_probs_i_first_bin = mask_soft_labels_i_first_bin.type(soft_probs_i.type()) * soft_probs_i
                    # print("masked_soft_labels_i_first_bin size", i ,masked_soft_labels_i_first_bin.size())
                    # print("masked_soft_probs_i_first_bin size", i ,masked_soft_probs_i_first_bin.size())


                masked_soft_labels_i = masked_soft_labels_i.reshape(-1, topk)
                masked_soft_probs_i = masked_soft_probs_i.reshape(-1, topk)
                # print("masked_soft_labels_i size", i, masked_soft_labels_i.size())
                # print("masked_soft_probs_i size", i, masked_soft_probs_i.size())
                # print("masked_soft_labels_i[0]", i, masked_soft_labels_i[0])
                # print("masked_soft_probs_i[0]", i, masked_soft_probs_i[0])


                if c.get('distill_first_bin') == 1:
                    masked_soft_labels_i_first_bin = masked_soft_labels_i_first_bin.reshape(-1, topk)
                    masked_soft_probs_i_first_bin = masked_soft_probs_i_first_bin.reshape(-1, topk)
                    # print("masked_soft_labels_i_first_bin size", i ,masked_soft_labels_i_first_bin.size())
                    # print("masked_soft_probs_i_first_bin size", i ,masked_soft_probs_i_first_bin.size())
                    # print("masked_soft_labels_i_first_bin[0]", i ,masked_soft_labels_i_first_bin[0])
                    # print("masked_soft_probs_i_first_bin[0]", i ,masked_soft_probs_i_first_bin[0])

                masked_soft_probs_i = masked_soft_probs_i + 1e-8

                if c.get('distill_first_bin') == 1:
                    masked_soft_probs_i_first_bin = masked_soft_probs_i_first_bin + 1e-8

                # normalize soft_probs_i to sum=1 as the flag indicated
                # if c.get('no_normalize') != 1:
                #     masked_soft_probs_i = masked_soft_probs_i / masked_soft_probs_i.sum(axis=1).unsqueeze(1)

                if c.get('distill_first_bin') == 1:
                    # masked_soft_labels_i_all = torch.cat((masked_soft_labels_i_first_bin, masked_soft_labels_i), 1)
                    masked_soft_probs_i_all = torch.cat((masked_soft_probs_i_first_bin, masked_soft_probs_i), 1)
                    # print("masked_soft_probs_i_all size", i ,masked_soft_probs_i_all.size())
                    # print("masked_soft_probs_i_all[0]", i ,masked_soft_probs_i_all[0])

                    if c.get('no_normalize') != 1:
                        masked_soft_probs_i_all = masked_soft_probs_i_all / masked_soft_probs_i_all.sum(axis=1).unsqueeze(1)
                else:
                    if c.get('no_normalize') != 1:
                        masked_soft_probs_i = masked_soft_probs_i / masked_soft_probs_i.sum(axis=1).unsqueeze(1)

                if i == 0:
                    hiddens[i] = (hidden.detach().index_select(0, indices_i), target_i)
                    logprob_i = head_logprob_i.gather(1, target_i).squeeze(1)

                    logprob_distill_i = torch.gather(head_logprob_i, 1, masked_soft_labels_i)

                    distill_loss_i = torch.bmm(
                        logprob_distill_i.view(logprob_distill_i.size(0), 1, logprob_distill_i.size(1)),
                        masked_soft_probs_i.view(masked_soft_probs_i.size(0), masked_soft_probs_i.size(1), 1))
                    distill_loss_i = distill_loss_i.squeeze()

                else:
                    if c.get('distill_first_bin') == 1:
                        logprob_distill_i_first_bin = torch.gather(head_logprob_i, 1, masked_soft_labels_i_first_bin)
                        # print("logprob_distill_i_first_bin size", i ,logprob_distill_i_first_bin.size())
                        # print("logprob_distill_i_first_bin[0]", i ,logprob_distill_i_first_bin[0])

                    hidden_i = hidden.index_select(0, indices_i)
                    proj_i = self.projections[i - 1](hidden_i)
                    tail_logit_i = self.layers[i](proj_i)
                    tail_logprob_i = tail_logit_i.log_softmax(dim=1)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i).squeeze(1)

                    logprob_distill_i = head_logprob_i[:, -i].unsqueeze(1) + torch.gather(tail_logprob_i, 1,
                                                                                          masked_soft_labels_i)

                    if c.get('distill_first_bin') == 1:
                        logprob_distill_i = torch.cat((logprob_distill_i_first_bin, logprob_distill_i), 1)
                        # print("logprob_distill_i size", i, logprob_distill_i.size())
                        # print("logprob_distill_i[0]", i, logprob_distill_i[0])
                        masked_soft_probs_i = masked_soft_probs_i_all
                        # print("masked_soft_probs_i size", i, masked_soft_probs_i.size())
                        # print("masked_soft_probs_i[0]", i, masked_soft_probs_i[0])
                    hiddens[i] = (proj_i.detach(), target_i)
                    # logprob_distill_i = torch.gather(logprob_distill_i_all, 1, masked_soft_labels_i)

                    distill_loss_i = torch.bmm(
                        logprob_distill_i.view(logprob_distill_i.size(0), 1, logprob_distill_i.size(1)),
                        masked_soft_probs_i.view(masked_soft_probs_i.size(0), masked_soft_probs_i.size(1), 1))
                    distill_loss_i = distill_loss_i.squeeze()
                if c.use_cache:
                    if c.get('cache_lambda_init'):
                        cache_lambda = self.cache_lambda_inv_sigmoid.sigmoid()
                    else:
                        cache_lambda = torch.tensor(c.cache_lambda, device=cache_logprob_i.device, dtype=cache_logprob_i.dtype)
                    self.last_lambda = from_torch(cache_lambda)
                    logprob_i = torch.stack([cache_lambda.log() + cache_logprob_i, (1 - cache_lambda).log() + logprob_i]).logsumexp(dim=0)

                if keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                    distill_loss.index_copy_(0, indices_i, -distill_loss_i)
 
                else:
                    nll[offset: offset + logprob_i.size(0)].copy_(-logprob_i)
                    distill_loss[offset: offset + distill_loss_i.size(0)].copy_(-distill_loss_i)
                    offset += logprob_i.size(0)

            if c.get('teacher_annealing'):
                hard_ratio = c.get('annealing_hard_min') + min(1, current_step / c.get('total_step')) * (c.get('annealing_hard_max') - c.get('annealing_hard_min'))
                # print(hard_ratio)
                return hard_ratio * nll + (1 - hard_ratio) * distill_loss, hiddens

            return c.get('hard_lambda') * nll + c.get('soft_lambda') * distill_loss, hiddens


        head_logprob = head_logit.log_softmax(dim=1)

        nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

        hiddens = {}
        offset = 0
        for i, (start, end) in enumerate(zip([0] + c.cutoffs, c.cutoffs + [c.n_vocab])):
            mask_i = (target >= start) & (target < end)
            
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue
            
            if c.use_cache:
                cache_logprob_i = cache_logprob.index_select(0, indices_i)
            
            target_i = (target.index_select(0, indices_i) - start)[:, None]
            head_logprob_i = head_logprob.index_select(0, indices_i)

            if i == 0:
                hiddens[i] = (hidden.detach().index_select(0, indices_i), target_i)
                logprob_i = head_logprob_i.gather(1, target_i).squeeze(1)
            else:
                hidden_i = hidden.index_select(0, indices_i)
                proj_i = self.projections[i - 1](hidden_i)
                tail_logit_i = self.layers[i](proj_i)
                tail_logprob_i = tail_logit_i.log_softmax(dim=1)
                
                logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i).squeeze(1)
                hiddens[i] = (proj_i.detach(), target_i)
            
            if c.use_cache:
                if c.get('cache_lambda_init'):
                    cache_lambda = self.cache_lambda_inv_sigmoid.sigmoid()
                else:
                    cache_lambda = torch.tensor(c.cache_lambda, device=cache_logprob_i.device, dtype=cache_logprob_i.dtype)
                self.last_lambda = cache_lambda
                logprob_i = torch.stack([cache_lambda.log() + cache_logprob_i, (1 - cache_lambda).log() + logprob_i]).logsumexp(dim=0)
            if keep_order:
                nll.index_copy_(0, indices_i, -logprob_i)
            else:
                nll[offset: offset + logprob_i.size(0)].copy_(-logprob_i)
                offset += logprob_i.size(0)

        return nll, hiddens

def hebbian_weight_update(c, net, hiddens, counters, temp_counters):
    with torch.no_grad():
        for i, (hidden_i, target_i) in hiddens.items():
            target_i = target_i.reshape(-1)
            c_i = counters[i]
            n_i = temp_counters[i]
            n_cls = c_i.size(0)
            lam = (1 / c_i.float()).clamp(min=c.hebbian_gamma) * (c_i < c.hebbian_T).float()

            n_i.index_add_(0, target_i, torch.ones_like(target_i))

            weight = (net.module if c.distributed else net).loss.layers[i].weight.data
            
            h_sums = torch.zeros_like(weight)
            hidden_i = hidden_i * weight[target_i].norm(dim=1, keepdim=True).to(hidden_i.dtype) / hidden_i.norm(dim=1, keepdim=True)
            h_sums.index_add_(0, target_i, hidden_i.to(h_sums.dtype))

            if c.distributed:
                all_h_sums = [torch.zeros_like(h_sums) for _ in range(c.world_size)]
                torch.distributed.all_gather(all_h_sums, h_sums)
                
                all_n_is = [torch.zeros_like(n_i) for _ in range(c.world_size)]
                torch.distributed.all_gather(all_n_is, n_i)
                h_sums = sum(all_h_sums)
                n_i = sum(all_n_is)
            
            c_i += n_i # update total count

            mask = n_i > 0
            h_sums = h_sums[mask]
            n_i = n_i[mask]

            h_means = lam[mask][:, None] * h_sums / n_i[:, None].to(h_sums.dtype) # divide by mean then scale by lambda
            
            weight[mask].mul_(1 - lam[mask][:, None]) # scale by 1 - lambda
            weight[mask].add_(h_means)
            
            n_i[:] = 0

