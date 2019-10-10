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

try:
    from apex import amp
except ImportError:
    pass

from u import *
from model import get_net, Transformer
from modules import hebbian_weight_update
from optim import scheduler, get_opt 
from ut import count_params, to_torch, from_torch
from data import SampleIterator, SequentialIterator, DistillationSampleIterator

module_path = "/home/demi/projects/distiller"
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
import distiller.apputils as apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger
msglogger = apputils.config_pylogger('logging.conf', None)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)




def train(c, net, compression_scheduler=None):
    c.setdefault(hebbian=False)
    assert not c.distributed and not c.parallel

    emb_params = count_params(net.embed) + count_params(net.loss.projections) + count_params(net.loss.clusters)
    opt = get_opt(c, net)
    net, opt, step = c.init_model(net, opt=opt, step='max', train=True)
    step_lr = scheduler(c, opt, step)
    data_tr = SampleIterator(c, c.train_batch, split='valid' if c.debug else 'train')
    iter_tr = iter(data_tr)
    data_val = SequentialIterator(c, c.eval_batch, split='valid')

    s = Namespace(net=net, opt=opt, step=step)
    c.on_train_start(s)

    c.log('Embedding has %s parameters' % emb_params)

    if c.get("steps_per_epoch"):
        steps_per_epoch = c.steps_per_epoch
    else:
        steps_per_epoch = len(data_tr.tokens) // data_tr.bs // c.train_chunk
    print("#### steps per epoch %d ####" % steps_per_epoch)
    

    if c.hebbian:
        counters = [torch.ones(end - start, dtype=torch.long, device=c.device) for start, end in zip([0] + c.cutoffs, c.cutoffs + [c.n_vocab])]
        temp_counters = [torch.zeros_like(x) for x in counters]

    best_val_loss = np.inf
    if s.results is not None and 'val_loss' in s.results.columns:
        best_val_loss = s.results['val_loss'].dropna().max()
    try:
        while step < s.step_max:
            batch = step % steps_per_epoch
            epoch = step // steps_per_epoch 
            if step % steps_per_epoch == 0:
                c.log("====> batch=%d, epoch=%d, step=%d" % (batch, epoch, step))
                if compression_scheduler:
                    compression_scheduler.on_epoch_begin(epoch)

            if compression_scheduler:
                compression_scheduler.on_minibatch_begin(epoch, minibatch_id=batch, minibatches_per_epoch=steps_per_epoch)


            step_lr(step)

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

            if compression_scheduler:
                _  = compression_scheduler.before_backward_pass(epoch, minibatch_id=batch,
                                                           minibatches_per_epoch=steps_per_epoch,
                                                           loss=total_loss, return_loss_components=False)

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
            step_result['theta'] = preds['theta']
            step_result['lambda'] = preds['lambda'].item()

            if compression_scheduler:
                compression_scheduler.on_minibatch_end(epoch, minibatch_id=batch, minibatches_per_epoch=steps_per_epoch)

            if step % steps_per_epoch == 0:
                if compression_scheduler:
                    compression_scheduler.on_epoch_end(epoch)


            if step % c.step_eval == 0:
                distiller.log_weights_sparsity(net, epoch, loggers=[tflogger, pylogger])
                t, total = distiller.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
                c.log("total sparsity: %.3lf" % total)

                step_result = step_result.append(
                    pd.Series(evaluate(c, data_val, net)).add_prefix('val_')
                )
                s.record_step = step_result['val_loss'] < best_val_loss
                clear_gpu_memory()
            s.step_result = step_result

            """
            # sanity check
            t, total = distiller.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
            val_result = evaluate(c, data_val, net)
            c.log("#### before on step end: sparsity %.3lf | val result: %s" % (total, val_result))
            """


            c.on_step_end(s)

            """
            # sanity check
            t, total = distiller.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
            val_result = evaluate(c, data_val, net)
            c.log("#### after on step end: sparsity %.3lf | val result: %s" % (total, val_result))
            """


            c.log("@@@@ step %d end @@@@" % step)


            s.step = step = step + 1
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

    if was_training:
        net.train()
    loss = from_torch(loss)
    perplexity = np.nan if loss > 5 else np.e ** loss
    return dict(loss=loss, perplexity=perplexity, time=np.round(time() - t_s))

def evaluate_cache_search(config, net):
    opt = get_opt(config, net)
    net, opt, step = config.init_model(net, opt=opt, step='max', train=True)
    distiller.model_summary(net, "sparsity", 'wikitext-103')
    perplexity = {}

    # search best cache hyperparamters on validation
    data_val = SequentialIterator(config,config.eval_batch, split="valid")
    nocache_ppl = evaluate(config, data_val, net)
    config.log("nocahce val ppl: %s" % nocache_ppl)
    thetas = [2e-2, 1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3]
    lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    thetas = thetas[:5]
    lambdas = lambdas[3:9]
    best_theta = -1
    best_lambda = -1
    best_ppl = 1000000
    data_test = SequentialIterator(config, config.eval_batch, split="test")
    for theta in thetas:
        for lam in lambdas:
            if (theta, lam) in perplexity:
                continue
            try:
                net.loss.cache_keys = net.loss.cache_values = None
            except:
                net.module.loss.cache_keys = net.module.loss.cache_values = None
            perplexity[theta, lam] = evaluate(config.var(use_cache=True, n_cache=2000, cache_theta=theta, cache_lambda=lam), data_val, net)['perplexity']
            print("ppl theta=", theta," lam=", lam, "perpelxity=", perplexity[theta, lam])
            eval_output = evaluate(config.var(use_cache=True, n_cache=2000, cache_thetaa=best_theta, cache_lambda=best_lambda), data_test, net)
            config.log("TEST RESULT: %s" % eval_output)
            if perplexity[theta, lam] < best_ppl:
                best_theta = theta
                best_lambda = lam
                best_ppl = perplexity[theta, lam]

    # evaluate on test
    data_test = SequentialIterator(config, config.eval_batch, split="test")
    print("Final Evaluation")
    distiller.model_summary(net, "sparsity", 'wikitext-103')
    eval_output = evaluate(config.var(use_cache=True, n_cache=2000, cache_thetaa=best_theta, cache_lambda=best_lambda), data_test, net)
    config.log("VAL RESULT: ppl(%.3lf) theta(%.3lf) lambda(%.3lf)" % (best_ppl, best_theta, best_lambda))
    config.log("TEST RESULT: %s" % eval_output)
    return eval_output




if __name__ == '__main__':
    config = Config.from_args()
    print("config=", config)
    net = get_net(config)

    if config.get("summary"):
        opt = get_opt(config, net)
        net, opt, step = config.init_model(net, opt=opt, step='max', train=True)
        config.log("===> summary of model @ step %d" % step)
        distiller.model_summary(net, config.summary, 'wikitext-103')
        exit(0)

    if config.get("compress"):
        config.log("===> compress from: %s" % config.compress)
        compression_scheduler = distiller.config.file_config(net, None, config.compress)
        

    if config.get('eval_cache_search'):
        evaluate_cache_search(config, net)
    else:
        train(config, net, compression_scheduler)
