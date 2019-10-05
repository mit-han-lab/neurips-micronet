import os
import sys
from apex import amp
import torch
import torch.nn as nn    
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from IPython import embed

from ut import Namespace
from modules import AdaptiveEmbedding
from modules import ProjectedAdaptiveLogSoftmax

from tensor_train import TTLayer

get_net = lambda c: eval(c.model_class)(c)

class Decoder(nn.Module):
    def __init__(self, c, layer_i):
        super(Decoder, self).__init__()
        self.layer_i = layer_i
        c_global = c
        c = Namespace(**c.layers[layer_i]).setdefault(
            n_embed=c.n_embed, n_inner=c.n_inner, n_head=c.n_head, n_k=c.n_k, n_v=c.n_v, n_seq=c.n_seq, dropout=c.dropout, pos_emb=c.pos_emb,
            mask_pad=c.mask_pad, fix_softmax=c.fix_softmax, light_conv=c.light_conv, tensor_train=c.tensor_train
        )
        
        n_embed = c.n_embed
        if c.light_conv:
            n_embed = c.n_embed // 2
            c.n_head = c.n_head // 2
            
            import fairseq
            c.setdefault(lc_kernel_size=c_global.get('lc_kernel_size'))
            
            self.light_conv = fairseq.modules.LightweightConv(
                n_embed, c.lc_kernel_size, padding_l=c.lc_kernel_size - 1, 
                weight_softmax=False, num_heads=c.n_head, weight_dropout=0
            )
       
        self.ln1 = nn.LayerNorm(n_embed)
        
        self.qkv = nn.Linear(n_embed, c.n_head * (2 * c.n_k + c.n_v))
        if c.pos_emb == 'trained':
            self.pos_emb = nn.Parameter(torch.Tensor(c.n_k, c.n_seq + 1))
            nn.init.normal_(self.pos_emb, 0, 0.02)

        self.out = nn.Linear(c.n_head * c.n_v, n_embed, bias=False)
        self.dropout = nn.Dropout(c.dropout)

        self.ln2 = nn.LayerNorm(c.n_embed)
        self.fc = nn.Sequential(
            TTLayer(c_global.modes_embed, c_global.modes_inner, c_global.ranks_e2i) if c.tensor_train else nn.Linear(c.n_embed, c.n_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(c.dropout),
            TTLayer(c_global.modes_inner, c_global.modes_embed, c_global.ranks_i2e) if c.tensor_train else nn.Linear(c.n_inner, c.n_embed),
            nn.Dropout(c.dropout),
        )
        self.c = c
    
    def forward(self, x, prev=None):
        # x: (n_group * n_seq, n_batch, n_embed)
        # pos_emb: (n_k, n_seq + 1)
        # mask: (2 * n_seq, 2 * n_seq) parallelogram
        
        c = self.c
        n_s = min(c.n_seq, x.size(0))
        n_g = x.size(0) // n_s
        n_b = x.size(1)
        n_h = c.n_head
        n_k = c.n_k
        n_v = c.n_v
        
        if c.light_conv:
            if prev is None:
                incremental_state = {} # create new lightconv state
            else:
                prev, incremental_state = prev
            x, x2 = x.chunk(2, dim=2)
            out2 = self.light_conv(x2.transpose(0, 1).contiguous(), incremental_state=incremental_state)

        qkv = self.qkv(self.ln1(x)).reshape(n_g * n_s, n_b * n_h, 2 * n_k + n_v)
        q, kv = qkv.split([n_k, n_k + n_v], dim=-1)
        
        q = q.reshape(n_g, n_s, n_b * n_h, n_k)

        padding = prev or torch.zeros((n_s, n_b * n_h, n_k + n_v), dtype=kv.dtype, device=kv.device)
        kv = torch.cat((padding, kv))
        k, v = kv.unfold(0, 2 * n_s, n_s).split([n_k, n_v], dim=2) # (n_g, n_bh, n_kv, 2 * n_s)

        qk = torch.einsum('gsbk,gbkt->gbst', q, k) # (n_g, n_bh, n_s, 2 * n_s)
        qk = qk.reshape(n_g, n_b * n_h, -1).unfold(2, n_s + 1, 2 * n_s + 1) # (n_g, n_bh, n_s, n_s + 1)

        pos_emb = self.pos_emb
        qe = torch.einsum('gsbk,kt->gbst', q, pos_emb.to(q.dtype))

        attn = qk + qe
        attn.mul_(n_k ** -0.5)
        
        if prev is None and c.mask_pad:
            mask = torch.triu(torch.ones(attn.shape[2:], dtype=torch.uint8, device=attn.device), 1).flip([1])
            attn[0].masked_fill_(mask, -np.inf)
        if c.fix_softmax:
            attn = attn.softmax(dim=-1)
        else:
            attn.softmax(dim=-1)

        attn = F.pad(attn, (0, n_s))
        attn = attn.reshape(n_g, n_b * n_h, -1).unfold(2, 2 * n_s, 2 * n_s) # (n_g, n_bh, n_s, 2 * n_s)

        attnv = torch.einsum('gbst,gbvt->gsbv', attn, v) # (n_g, n_s, n_bh, n_v)
        attn_out = self.out(attnv.reshape(n_g * n_s, n_b, n_h * n_v)) # (n_g * n_s, n_b, n_embed)
        attn_out = self.dropout(attn_out)

        out = x + attn_out
        
        next = kv[-n_s:].detach()
        if c.light_conv:
            out = torch.cat((out, out2.transpose(0, 1)), dim=2)
            next = next, incremental_state

        out = out + self.fc(self.ln2(out))

        return out, next

class Transformer(nn.Module):
    def __init__(self, c):
        super(Transformer, self).__init__()
        self.c = c.setdefault(layers=[{} for _ in range(c.n_layers)], light_conv=False, mask_pad=False, fix_softmax=False, tie_layers=False, tensor_train=False)
        self.embed = AdaptiveEmbedding(c)

        self.dropout = nn.Dropout(c.dropout)

        if c.tie_layers:
            if c.tie_layers is True:
                self.layer = Decoder(c, 0)
                self.layers = [self.layer] * c.n_layers
            else: # tie_layers is a list of how many repeats
                assert sum(c.tie_layers) == c.n_layers
                self.layers_base = nn.ModuleList([Decoder(c, i) for i in range(len(c.tie_layers))])
                self.layers = [layer for i, layer in enumerate(self.layers_base) for j in range(c.tie_layers[i])]
        else:
            self.layers = nn.ModuleList(Decoder(c, i) for i in range(c.n_layers))

        self.loss = ProjectedAdaptiveLogSoftmax(c)

        # tie output embedding weights to input embedding weights
        for layer_embed, layer_loss in zip(self.embed.layers, self.loss.layers):
            layer_loss.weight = layer_embed.weight

    def forward(self, inputs, labels, prevs=None, soft_labels=None, soft_probs=None, is_distilling=False, current_step=0.):
        # inputs: (n_group * n_seq, n_batch)
        # labels: (n_group * n_seq, n_batch)
        c = self.c

        n_gs = inputs.size(0)
        n_s = c.n_seq
        if n_gs % n_s != 0:
            padding = torch.zeros((n_s - n_gs % n_s, inputs.size(1)), dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat((inputs, padding))
            labels = torch.cat((labels, padding))

        x = self.embed(inputs)
        x = self.dropout(x)

        prevs = prevs or [None] * c.n_layers
        nexts = []
        for layer, prev in zip(self.layers, prevs):
            x, prev = layer(x, prev=prev)
            nexts.append(prev)
        
        x = self.dropout(x)
        if c.get('distillation_teacher') == 'file' and is_distilling:
            soft_labels_reshape = soft_labels.reshape(-1, soft_labels.size(2))
            soft_probs_reshape = soft_probs.reshape(-1, soft_probs.size(2))
            loss, hiddens = self.loss(hidden=x.reshape(-1, x.size(2)), target=labels.reshape(-1),
                                      soft_labels=soft_labels_reshape, soft_probs=soft_probs_reshape,
                                      is_distilling=is_distilling, current_step=current_step)
            loss = loss.reshape(labels.shape)[:n_gs]
            return dict(loss=loss.mean(), state=nexts, hiddens=hiddens)

        loss, hiddens = self.loss(x.reshape(-1, x.size(2)), labels.reshape(-1))
        if c.get('gen_soft'):
            # loss = loss.reshape(labels.shape)[:n_gs]
            return loss, hiddens

        loss = loss.reshape(labels.shape)[:n_gs]
        return dict(loss=loss.mean(), state=nexts, hiddens=hiddens)


