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

from ut import *


if __name__ == "__main__":
    decoder = (Cache / 'vocab.npy').load()
    encoder = get_encoder(decoder)
    n_vocab = len(decoder)

# base hyperparameters for transformer
    transformer = dict(
        model=Proj / 'main.py', model_class='Transformer', n_vocab=n_vocab, step_save=5000,
        train_batch=17, train_chunk=1088,
        step_eval=500, eval_batch=1, eval_chunk=4096,
        cutoffs=[5000, 25000, 50000], adaptive_ratio=4, pos_emb='trained',
        n_seq=64, n_layers=16, n_embed=256, n_head=8, n_k=32, n_v=32, n_inner=1024, dropout=0.1,
        lr=0.0005, step_warmup=100, scheduler='cosine'
    )

    sorted_hebbian = transformer.copy()
    sorted_hebbian.update(dict(
        hebbian=True, hebbian_gamma=0.01, hebbian_T=500,
        vocab_sorted=True, cutoffs=[3500, 25000], n_embeds=[256, 64, 4]
    ))

    sorted_hebbian_mask = sorted_hebbian.copy()
    sorted_hebbian_mask.update(dict(mask_pad=True, fix_softmax=True, train_batch=16))

    tie_layers_8x2 = sorted_hebbian_mask.copy()
    tie_layers_8x2.update(dict(
        tie_layers=[2] * 8,
        train_batch=16
    ))
    c = Config(Wiki / 'tie_layers,8x2', tie_layers_8x2).save(True)
    print(c.train(env_gpu=1, steps=200000, opt='O1'))


