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
from data import SampleIterator, SequentialIterator, DistillationSampleIterator

def train(c):
    c.setdefault(hebbian=False)
    net = get_net(c)

    emb_params = count_params(net.embed) + count_params(net.loss.projections) + count_params(net.loss.clusters)
    opt = get_opt(c, net)
    net, opt, step = c.init_model(net, opt=opt, step='max', train=True)
    if c.get('distillation_teacher') == 'file':
        data_tr_distill = DistillationSampleIterator(c, c.train_batch, split='train')
        iter_tr_distill = iter(data_tr_distill)

    step_lr = scheduler(c, opt, step)
    data_tr = SampleIterator(c, c.train_batch, split='valid' if c.debug else 'train')
    iter_tr = iter(data_tr)
    data_val = SequentialIterator(c, c.eval_batch, split='valid')

    s = Namespace(net=net, opt=opt, step=step)
    c.on_train_start(s)

    c.log('Embedding has %s parameters' % emb_params)

    if c.hebbian:
        counters = [torch.ones(end - start, dtype=torch.long, device=c.device) for start, end in zip([0] + c.cutoffs, c.cutoffs + [c.n_vocab])]
        temp_counters = [torch.zeros_like(x) for x in counters]

    best_val_loss = np.inf
    if s.results is not None and 'val_loss' in s.results.columns:
        best_val_loss = s.results['val_loss'].dropna().max()
    try:
        while step < s.step_max:
            step_lr(step)

            if c.get('distillation_teacher') == 'file':
                x_hard_labels, x_soft_labels, x_soft_probs = next(iter_tr_distill)

                x_hard_labels = to_torch(x_hard_labels, c.device).t()

                x_soft_labels = to_torch(x_soft_labels, c.device)
                x_soft_labels = x_soft_labels.permute(1, 0, 2)

                x_soft_probs = to_torch(x_soft_probs, c.device)
                x_soft_probs = x_soft_probs.permute(1, 0, 2)

                inputs, hard_labels = x_hard_labels[:-1], x_hard_labels[1:]
                soft_labels = x_soft_labels[1:]
                soft_probs = x_soft_probs[1:]

                t_s = time()

                preds = net(inputs=inputs, labels=hard_labels, soft_labels=soft_labels, soft_probs=soft_probs,
                            is_distilling=True, current_step=step)
                loss = preds['loss']
                total_loss = loss
                extras = {}

            else:
                x = to_torch(next(iter_tr), c.device).t()

                t_s = time()
                inputs, labels = x[:-1], x[1:]
                preds = net(inputs, labels)
                loss = preds['loss']
                if c.model_class == 'UniversalTransformer':
                    act_loss = preds['act_loss']
                    total_loss = act_loss + loss
                    extras = dict(act_loss=from_torch(act_loss), n_updates=from_torch(preds['n_updates'].mean()))
                else:
                    total_loss = loss
                    extras = {}

            opt.zero_grad()
            if torch.isnan(total_loss):
                raise RuntimeError('Encountered nan loss during training')
            with amp.scale_loss(total_loss, opt) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), c.get('clip_grad', 0.5))
            opt.step()

            if c.hebbian:
                hebbian_weight_update(c, net, preds['hiddens'], counters, temp_counters)

            time_model = np.round(time() - t_s, 5)

            loss = from_torch(loss)
            perplexity = np.nan if loss > 5 else np.e ** loss
            step_result = pd.Series(Dict(
                loss=loss,
                perplexity=perplexity,
                time=time_model,
                **extras
            )).add_prefix('train_')
            step_result['lr'] = next(iter(opt.param_groups))['lr']

            s.step = step = step + 1
            if step % c.step_eval == 0:
                step_result = step_result.append(
                    pd.Series(evaluate(c, data_val, net)).add_prefix('val_')
                )
                s.record_step = step_result['val_loss'] < best_val_loss
                clear_gpu_memory()
            s.step_result = step_result
            c.on_step_end(s)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        if c.main:
            c.log(err)
        else:
            print(err)
    finally:
        c.on_train_end(s)

def evaluate(c, data, net):
    clear_gpu_memory()
    was_training = net.training
    net.eval()
    
    t_s = time()
    with torch.no_grad():
        weights = []
        losses = []
        prevs = None

        for batch in data:
            x = to_torch(batch, c.device).t()
            inputs, labels = x[:-1], x[1:]

            preds = net.forward(inputs, labels, prevs=prevs)
            losses.append(preds['loss'])
            weights.append(labels.size(0))
        weights = np.array(weights)
        weights = weights / weights.sum()
        loss = sum(x * w for x, w in zip(losses, weights))

    if c.distributed:
        gathered_losses = [torch.zeros_like(loss) for _ in range(c.world_size)]
        torch.distributed.all_gather(gathered_losses, loss)
        loss = sum(gathered_losses) / len(gathered_losses)
    if was_training:
        net.train()
    loss = from_torch(loss)
    perplexity = np.nan if loss > 5 else np.e ** loss
    return dict(loss=loss, perplexity=perplexity, time=np.round(time() - t_s))


def evaluate_cache_search(config):
    net, step = config.var(device="cuda:0").load_model("max")
    print(step)

    # search best cache hyperparamters on validation
    data_val = SequentialIterator(config,config.eval_batch, split="valid")
    thetas = [1.2e-2, 1.1e-2, 1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3]
    lambdas = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    best_theta = -1
    best_lambda = -1
    best_ppl = 1000000
    perplexity = {}
    for theta in thetas:
        for lam in lambdas:
            if (theta, lam) in perplexity:
                continue
            net.loss.cache_keys = net.loss.cache_values = None
            perplexity[theta, lam] = evaluate(config.var(use_cache=True, n_cache=500, cache_theta=theta, cache_lambda=lam), data_val, net)['perplexity']
            print(perplexity[theta, lam])

            if perplexity[theta, lam] < best_ppl:
                best_theta = theta
                best_lambda = lam
                best_ppl = perplexity[theta, lam]

    # evaluate on test
    data_test = SequentialIterator(config, config.eval_batch, split="test")
    eval_output = evaluate(config.var(use_cache=True, n_cache=500, cache_theta=best_theta, cache_lambda=best_lambda), data_test, net)
    config.log("TEST RESULT: %s" % eval_output)
    return eval_output


if __name__ == '__main__':
    config = Config.from_args()
    if config.get('eval_cache_search'):
        evaluate_cache_search(config)
    else:
        train(config)
