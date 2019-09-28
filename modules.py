import os
import sys
import numpy as np
import torch
import torch.nn as nn

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
            cache_keys = torch.cat((cache_keys, hidden))
            cache_values = torch.cat((cache_values, target))
        cache_keys = cache_keys[-(c.n_cache + n_seq):]
        cache_values = cache_values[-(c.n_cache + n_seq):]
        
        attn = c.cache_theta * F.pad(hidden.mm(cache_keys.t()), (c.n_cache - n_cache, 0), value=-np.inf) # (n_s, n_c + n_s)
        # (n_s, n_cache)
        probs = attn.reshape(-1).unfold(0, c.n_cache, attn.size(1) + 1).softmax(dim=1)
        indices = F.pad(cache_values, (c.n_cache - n_cache, 0)).unfold(0, c.n_cache, 1)[:n_seq]

        mask = indices != target[:, None]
        probs.masked_fill_(mask, 0)

        pos_prob = probs.sum(dim=1)
        self.cache_keys = cache_keys[-c.n_cache:]
        self.cache_values = cache_values[-c.n_cache:]
        return pos_prob
        
    def forward(self, hidden, target, keep_order=False):
        # hidden: (n_seq * n_batch, n_embed)
        # target: (n_seq * n_batch)
        c = self.c
        assert hidden.size(0) == target.size(0), 'Input and target should have the same size in the batch dimension'
        
        if c.use_cache:
            cache_prob = self.query_cache(hidden, target)

        head_logit = torch.cat((self.layers[0](hidden), self.clusters(hidden)), dim=1)
        head_logprob = head_logit.log_softmax(dim=1)

        nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

        hiddens = {}
        offset = 0
        for i, (start, end) in enumerate(zip([0] + c.cutoffs, c.cutoffs + [c.n_vocab])):
            mask_i = (target >= start) & (target < end)
            if c.use_cache:
                cache_prob_i = cache_prob[mask_i]
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue
            
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
                logprob_i = (c.cache_lambda * cache_prob_i + (1 - c.cache_lambda) * logprob_i.exp()).log()

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

