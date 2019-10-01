from __future__ import absolute_import, print_function

import subprocess, sys, os, re, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json, yaml, gc
from datetime import datetime
from time import time
from fnmatch import fnmatch
from glob import glob
from tqdm import tqdm
from collections import OrderedDict, defaultdict, Counter
import q
qq = q
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from apex import amp

from u import *
from model import get_net, Transformer
from modules import hebbian_weight_update
from optim import scheduler, get_opt
from ut import count_params, to_torch, from_torch
from data import SampleIterator, SequentialIterator
from ut import Cache, Proj, Wiki, Data


class SequentialIteratorGenSoft:
    def __init__(self, c, batch_size, split):
        self.c = c
        worker = c.get('worker')
        if worker != None:
            # self.tokens = (Cache / 'wikitext-103' / split + '.npy').load()
            # print(self.tokens.shape)
            # print(Cache / 'wikitext-103' / split + '.npy')
            print(worker)
            if worker == 7:
                self.tokens = self.tokens = (Cache / 'wikitext-103' / ('sorted_' if c.vocab_sorted else '') + split + '.npy').load()[
                              6720 * 1920 * (worker):]
            else:
                self.tokens = self.tokens = (Cache / 'wikitext-103' / ('sorted_' if c.vocab_sorted else '') + split + '.npy').load()[
                              6720 * 1920 * (worker): 6720 * 1920 * (worker + 1)]
            # self.tokens = self.tokens = (Cache / 'wikitext-103' / ('sorted_' if c.vocab_sorted else '') + split + '.npy').load()[6720*1920*(worker): 6720*1920*(worker)+10000]
        else:
            self.tokens = self.tokens = (Cache / 'wikitext-103' / ('sorted_' if c.vocab_sorted else '') + split + '.npy').load()

        self.batch_size = batch_size

    def __iter__(self):
        c = self.c
        bs = self.batch_size
        tokens = self.tokens

        n = len(tokens) - 1

        start, end = 0, n
        if c.distributed:
            start = n * c.local_rank // c.world_size
            end = n * (c.local_rank + 1) // c.world_size

        span_i = (end - start) // bs
        span = span_i * bs
        end = start + span

        starts = np.arange(start, end, span_i)
        for i in range(0, span_i, c.eval_chunk):
            starts_i = starts + i
            batch_i = np.array([tokens[s_i: s_i + c.eval_chunk + 1] for s_i in starts_i])
            yield batch_i.astype(np.int64)

    def __len__(self):
        n = len(self.tokens) - 1
        bs = self.batch_size
        start, end = 0, n
        span_i = (end - start) // bs

        return span_i // self.c.eval_chunk


def gen_soft_labels(c):
    c.setdefault(hebbian=False, distributed=False)
    net = get_net(c)
    opt = get_opt(c, net)
    net, opt, step = c.init_model(net, opt=opt, step='max', train=True)

    print('generating soft labels...')
    data_gen_tr = SequentialIteratorGenSoft(c, c.get('gen_soft_batch'), split='train')
    # data_gen_tr = iter(data_gen_tr)
    clear_gpu_memory()
    net.eval()
    with torch.no_grad():
        i = 0
        for batch in tqdm(data_gen_tr):
            x = to_torch(batch, c.device).t()
            # print(x.size())
            # print(x[0:20])
            inputs, labels = x[:-1], x[1:]
            probs, _ = net(inputs, labels)

            # loss_hard = -torch.log(probs.gather(1, labels).squeeze(1)).mean()

            values, indices = torch.topk(probs, c.get('topk'), dim=1)

            indices_ = indices.cpu().numpy()
            values_ = values.cpu().numpy()
            labels_ = labels.cpu().numpy()
            # print(indices_[0:5])
            # print(labels_[0:5])
            # exit(0)


            if probs.size(0) != inputs.size(0):
                indices_ = indices_[-inputs.size(0):, :]
                values_ = values_[-inputs.size(0):, :]
                # labels_ = labels_[-inputs.size(0):, :]

            if i == 0:
                all_soft_indices = indices_
                all_soft_values = values_
            else:
                all_soft_indices = np.concatenate((all_soft_indices, indices_), axis=0)
                all_soft_values = np.concatenate((all_soft_values, values_), axis=0)

            # print(all_soft_indices.shape)
            # print(all_soft_values.shape)

            i += 1
            # if i > 100:
            #     break
        all_soft_indices = np.concatenate((all_soft_indices[0:1, :], all_soft_indices), axis=0)
        all_soft_values = np.concatenate((all_soft_values[0:1, :], all_soft_values), axis=0)
        np.save(c.get('file_out_path') + 'all_soft_indices' + str(c.get('worker')) + '.npy', all_soft_indices)
        np.save(c.get('file_out_path') + 'all_soft_values' + str(c.get('worker')) + '.npy', all_soft_values)

        in_indices = np.load(c.get('file_out_path') + 'all_soft_indices' + str(c.get('worker')) + '.npy')

        cnt = 0.
        # print(in_indices.shape)
        # print(len(data.tokens))
        for k in range(len(data_gen_tr.tokens)):
            # print(data.tokens[k])
            # print(in_indices[k])
            if data_gen_tr.tokens[k] in in_indices[k]:
                cnt += 1
        print(cnt / len(data_gen_tr.tokens))





if __name__ == '__main__':
    config = Config.from_args()
    if config.get('gen_soft'):
        gen_soft_labels(config)

