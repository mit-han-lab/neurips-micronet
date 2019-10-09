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

from torchprofile import profile_macs

def profile(c):
    c.setdefault(hebbian=False, distributed=False)
    net, step = config.var(device="cpu").load_model("max")
    print('profiling')
    net.eval()

    data_val = SequentialIterator(config, config.eval_batch, split="test")
    with torch.no_grad():
        for batch in data_val:
            x = to_torch(batch, c.device).t()

            inputs, labels = x[:-1], x[1:]

            macs = profile_macs(net, (inputs, labels))
            print("==> FLOPS: ", macs / config.eval_chunk * 2)

            print("==> Models size: ", count_params(net))

            exit(0)



if __name__ == '__main__':
    config = Config.from_args()
    if config.get('profile'):
        profile(config)