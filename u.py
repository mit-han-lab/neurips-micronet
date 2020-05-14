### Common global imports ###
from __future__ import absolute_import, print_function

import subprocess, sys, os, re, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json, yaml, gc
from datetime import datetime
from time import time
from fnmatch import fnmatch
from glob import glob
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
import warnings
warnings.filterwarnings('ignore')

from io import StringIO

### Util methods ###

def get_encoder(decoder):
    return dict((x, i) for i, x in enumerate(decoder))

def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)

def save_json(path, dict_):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)

def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)

def format_yaml(dict_):
    dict_ = recurse(dict_, lambda x: x._ if type(x) is Path else dict(x) if type(x) is dict else x)
    return yaml.dump(dict_)

def load_text(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

def save_text(path, string):
    with open(path, 'w') as f:
        f.write(string)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def wget(link, output_dir):
    cmd = 'wget %s -P %s' % (path, output_dir)
    shell(cmd)
    output_path = Path(output_dir) / os.path.basename(link)
    if not output_path.exists(): raise RuntimeError('Failed to run %s' % cmd)
    return output_path

def extract(input_path, output_path=None):
    if input_path[-3:] == '.gz':
        if not output_path:
            output_path = input_path[:-3]
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
    else:
        raise RuntimeError('Don\'t know file extension for ' + input_path)

def shell(cmd, wait=True, ignore_error=2):
    if type(cmd) != str:
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not wait:
        return process
    out, err = process.communicate()
    return out.decode(), err.decode() if err else None

def attributes(obj):
    import inspect, pprint
    pprint.pprint(inspect.getmembers(obj, lambda a: not inspect.isroutine(a)))

def import_module(module_name, module_path):
    import imp
    module = imp.load_source(module_name, module_path)
    return module

_log_path = None
def logger(directory=None):
    global _log_path
    if directory and not _log_path:
        from datetime import datetime
        _log_path = Path(directory) / datetime.now().isoformat().replace(':', '_').rsplit('.')[0] + '.log'
    return log

def log(text):
    print(text)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(text)
            f.write('\n')

class Path(str):
    def __init__(self, path):
        pass

    def __add__(self, subpath):
        return Path(str(self) + str(subpath))

    def __truediv__(self, subpath):
        return Path(os.path.join(str(self), str(subpath)))

    def __floordiv__(self, subpath):
        return (self / subpath)._

    def ls(self, show_hidden=True, dir_only=False, file_only=False):
        subpaths = [Path(self / subpath) for subpath in os.listdir(self) if show_hidden or not subpath.startswith('.')]
        isdirs = [os.path.isdir(subpath) for subpath in subpaths]
        subdirs = [subpath for subpath, isdir in zip(subpaths, isdirs) if isdir]
        files = [subpath for subpath, isdir in zip(subpaths, isdirs) if not isdir]
        if dir_only:
            return subdirs
        if file_only:
            return files
        return subdirs, files

    def recurse(self, dir_fn=None, file_fn=None):
        if dir_fn is not None:
            dir_fn(self)
        dirs, files = self.ls()
        if file_fn is not None:
            list(map(file_fn, files))
        for dir in dirs:
            dir.recurse(dir_fn=dir_fn, file_fn=file_fn)

    def mk(self):
        os.makedirs(self, exist_ok=True)
        return self

    def rm(self):
        if self.isfile() or self.islink():
            os.remove(self)
        elif self.isdir():
            shutil.rmtree(self)
        return self

    def mv(self, dest):
        shutil.move(self, dest)

    def mv_from(self, src):
        shutil.move(src, self)

    def cp(self, dest):
        shutil.copy(self, dest)

    def cp_from(self, src):
        shutil.copy(src, self)

    def link(self, target, force=False):
        if self.exists():
            if not force:
                return
            else:
                self.rm()
        os.symlink(target, self)

    def exists(self):
        return os.path.exists(self)

    def isfile(self):
        return os.path.isfile(self)

    def isdir(self):
        return os.path.isdir(self)

    def islink(self):
        return os.path.islink(self)

    def rel(self, start=None):
        return Path(os.path.relpath(self, start=start))

    def clone(self):
        name = self._name
        match = re.search('__([0-9]+)$', name)
        if match is None:
            base = self + '__'
            i = 1
        else:
            initial = match.group(1)
            base = self[:-len(initial)]
            i = int(initial) + 1
        while True:
            path = Path(base + str(i))
            if not path.exists():
                return path
            i += 1


    @property
    def _(self):
        return str(self)

    @property
    def _real(self):
        return Path(os.path.realpath(self))

    @property
    def _up(self):
        path = os.path.dirname(self)
        if path is '':
            path = os.path.dirname(self._real)
        return Path(path)

    @property
    def _name(self):
        return os.path.basename(self)

    @property
    def _ext(self):
        frags = self._name.rsplit('.', 1)
        if len(frags) == 1:
            return ''
        return frags[1]

    extract = extract
    load_json = load_json
    save_json = save_json
    load_txt = load_text
    save_txt = save_text
    load_p = load_pickle
    save_p = save_pickle

    def load_csv(self, index_col=0, **kwargs):
        return pd.read_csv(self, index_col=index_col, **kwargs)

    def save_csv(self, df, float_format='%.5g', **kwargs):
        df.to_csv(self, float_format=float_format, **kwargs)

    def load_npy(self):
        return np.load(self, allow_pickle=True)

    def save_npy(self, obj):
        np.save(self, obj)

    def load_yaml(self):
        with open(self, 'r') as f:
            return yaml.safe_load(f)

    def save_yaml(self, obj):
        obj = recurse(obj, lambda x: x._ if type(x) is Path else dict(x) if type(x) is dict else x)
        with open(self, 'w') as f:
            yaml.dump(obj, f, default_flow_style=False, allow_unicode=True)

    def load(self):
        return eval('self.load_%s' % self._ext)()

    def save(self, obj):
        return eval('self.save_%s' % self._ext)(obj)

    def wget(self, link):
        if self.isdir():
            return Path(wget(link, self))
        raise ValueError('Path %s needs to be a directory' % self)


class Namespace(object):
    def __init__(self, *args, **kwargs):
        self.var(*args, **kwargs)

    def var(self, *args, **kwargs):
        kvs = dict()
        for a in args:
            if type(a) is str:
                kvs[a] = True
            else: # a is a dictionary
                kvs.update(a)
        kvs.update(kwargs)
        self.__dict__.update(kvs)
        return self

    def unvar(self, *args):
        for a in args:
            self.__dict__.pop(a)
        return self

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def setdefault(self, *args, **kwargs):
        args = [a for a in args if a not in self.__dict__]
        kwargs = {k: v for k, v in kwargs.items() if k not in self.__dict__}
        return self.var(*args, **kwargs)


##### Functions for compute

using_ipython = True
try:
    _ = get_ipython().__class__.__name__
except NameError:
    using_ipython = False

try:
    import numpy as np
    import pandas as pd

    import scipy.stats
    import scipy as sp
    from scipy.stats import pearsonr as pearson, spearmanr as spearman, kendalltau

    if not using_ipython:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt


    def _sel(self, col, value):
        if type(value) == list:
            return self[self[col].isin(value)]
        return self[self[col] == value]
    pd.DataFrame.sel = _sel
except ImportError:
    pass
try:
    from sklearn.metrics import roc_auc_score as auroc, average_precision_score as auprc, roc_curve as roc, precision_recall_curve as prc, accuracy_score as accuracy
except ImportError:
    pass

def recurse(x, fn):
    T = type(x)
    if T in [dict, OrderedDict]:
        return T((k, recurse(v, fn)) for k, v in x.items())
    elif T in [list, tuple]:
        return T(recurse(v, fn) for v in x)
    return fn(x)

def from_numpy(x):
    def helper(x):
        if type(x).__module__ == np.__name__:
            if type(x) == np.ndarray:
                return recurse(list(x), helper)
            return np.asscalar(x)
        return x
    return recurse(x, helper)

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_gpu_info(ssh_fn=lambda x: x):
    nvidia_str, _ = shell(ssh_fn('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,nounits'))
    nvidia_str = nvidia_str.replace('[Not Supported]', '100').replace(', ', ',')
    nvidia_str_io = StringIO(nvidia_str)

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    devices_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices_str:
        devices = list(map(int, devices_str.split(',')))
        gpu_df = gpu_df.loc[devices]
        gpu_df.index = gpu_df.index.map({k: i for i, k in enumerate(devices)})

    out_df = pd.DataFrame(index=gpu_df.index)
    out_df['memory_total'] = gpu_df['memory.total [MiB]']
    out_df['memory_used'] = gpu_df['memory.used [MiB]']
    out_df['memory_free'] = out_df['memory_total'] - out_df['memory_used']
    out_df['utilization'] = gpu_df['utilization.gpu [%]'] / 100
    out_df['utilization_free'] = 1 - out_df['utilization']
    return out_df

def get_process_gpu_info(pid=None, ssh_fn=lambda x: x):
    nvidia_str, _ = shell(ssh_fn('nvidia-smi --query-compute-apps=pid,gpu_name,used_gpu_memory --format=csv,nounits'))
    nvidia_str_io = StringIO(nvidia_str.replace(', ', ','))

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    if pid is None:
        return gpu_df
    if pid == -1:
        pid = os.getpid()
    return gpu_df.loc[pid]


##### torch functions

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    def to_torch(x, device='cuda'):
        def helper(x):
            if x is None:
                return None
            elif type(x) == torch.Tensor:
                return x.to(device)
            elif type(x) in [str, bool, int, float]:
                return x
            return torch.from_numpy(x).to(device)
        return recurse(x, helper)

    def from_torch(t):
        def helper(t):
            if type(t) != torch.Tensor:
                return t
            x = t.detach().cpu().numpy()
            if x.size == 1 or np.isscalar(x):
                return np.asscalar(x)
            return x
        return recurse(t, helper)

    def count_params(network, requires_grad=False):
        return sum(p.numel() for p in network.parameters() if not requires_grad or p.requires_grad)

    def report_memory(device=None, max=False):
        if device:
            device = torch.device(device)
            if max:
                alloc = torch.cuda.max_memory_allocated(device=device)
            else:
                alloc = torch.cuda.memory_allocated(device=device)
            alloc /=  1024 ** 2
            print('%.3f MBs' % alloc)
            return alloc

        numels = Counter()
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(type(obj), obj.size())
                numels[obj.device] += obj.numel()
        print()
        for device, numel in sorted(numels.items()):
            print('%s: %s elements, %.3f MBs' % (str(device), numel, numel * 4 / 1024 ** 2))

    def clear_gpu_memory():
        gc.collect()
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                obj.cpu()
        gc.collect()
        torch.cuda.empty_cache()

except ImportError:
    pass

try:
    from apex import amp
except ImportError:
    pass

def main_only(method):
    def wrapper(self, *args, **kwargs):
        if self.main:
            return method(self, *args, **kwargs)
    return wrapper

class Config(Namespace):
    def __init__(self, res, *args, **kwargs):
        self.res = Path(res)._real
        super(Config, self).__init__(*args, **kwargs)
        self.setdefault(
            name=self.res._real._name,
            main=True,
            logger=True,
            device='cuda',
            debug=False,
            opt_level='O0'
        )

    def __repr__(self):
        return format_yaml(vars(self))

    def __hash__(self):
        return hash(repr(self))

    @property
    def path(self):
        return self.res / (type(self).__name__.lower() + '.yaml')

    def load(self):
        if self.path.exists():
            for k, v in self.path.load().items():
                setattr(self, k, v)
        return self

    never_save = {'res', 'name', 'main', 'logger', 'distributed', 'parallel', 'device', 'debug'}
    @property
    def attrs_save(self):
        return {k: v for k, v in vars(self).items() if k not in self.never_save}

    def save(self, force=False):
        if force or not self.path.exists():
            self.res.mk()
            self.path.save(from_numpy(self.attrs_save))
        return self

    def clone(self):
        return self._clone().save()

    def clone_(self):
        return self.cp_(self.res._real.clone())

    def cp(self, path, *args, **kwargs):
        return self.cp_(path, *args, **kwargs).save()

    def cp_(self, path, *args, **kwargs):
        '''
        path: should be absolute or relative to self.res._up
        '''
        attrs = self.attrs_save
        for a in args:
            kwargs[a] = True
        kwargs = {k: v for k, v in kwargs.items() if v != attrs.get(k)}

        merged = attrs.copy()
        merged.update(kwargs)

        if os.path.isabs(path):
            new_res = path
        else:
            new_res = self.res._up / path
        return Config(new_res).var(**merged)

    @classmethod
    def from_args(cls):
        import argparse
        parser = argparse.ArgumentParser(description='Model arguments')
        parser.add_argument('res', type=Path, help='Result directory')
        parser.add_argument('kwargs', nargs='*', help='Extra arguments that goes into the config')

        args = parser.parse_args()

        kwargs = {}
        for kv in args.kwargs:
            splits = kv.split('=')
            if len(splits) == 1:
                v = True
            else:
                v = splits[1]
                try:
                    v = eval(v)
                except (SyntaxError, NameError):
                    pass
            kwargs[splits[0]] = v

        return cls(args.res).load().var(**kwargs).save()

    @classmethod
    def clean(cls, *directories):
        configs = cls.load_all(*directories)
        for config in configs:
            if not (config.train_results.exists() or len(config.models.ls()[1]) > 0):
                config.res.rm()
                self.log('Removed %s' % config.res)

    @main_only
    def log(self, text):
        logger(self.res if self.logger else None)(text)

    def on_train_start(self, s):
        step = s.step
        s.step_max = self.steps

        self.setdefault(
            step_save=np.inf,
            time_save=np.inf,
            patience=np.inf,
            step_print=1,
        )
        s.var(
            step_max=self.steps,
            last_save_time=time(),
            record_step=False,
            last_record_step=step,
            last_record_state=None,
            results=self.load_train_results()
        )

        if self.main and self.training.exists():
            self.log('Quitting because another training is found')
            exit()
        self.set_training(True)
        import signal
        def handler(signum, frame):
            self.on_train_end(s)
            exit()
        s.prev_handler = signal.signal(signal.SIGINT, handler)

        s.writer = None
        if self.main and self.get('use_tb', True):
            from torch.utils.tensorboard import SummaryWriter
            s.writer = SummaryWriter(log_dir=self.res, flush_secs=10)

        if self.stopped_early.exists():
            self.log('Quitting at step %s because already stopped early before' % step)
            s.step_max = step
            return

        self.log(str(self))
        self.log('Network has %s parameters' % count_params(s.net))

        s.progress = None
        if self.main:
            self.log('Training %s from step %s to step %s' % (self.name, step, s.step_max))
            s.progress = iter(RangeProgress(step, s.step_max, desc=self.name))

    def on_step_end(self, s):
        step = s.step
        results = s.results
        step_result = s.step_result

        if results is None:
            s.results = results = pd.DataFrame(columns=step_result.index, index=pd.Series(name='step'))

        prev_time = 0
        if len(results):
            last_step = results.index[-1]
            prev_time = (step - 1) / last_step * results.loc[last_step, 'total_train_time']
        tot_time = step_result['total_train_time'] = prev_time + step_result['train_time']

        if step_result.index.isin(results.columns).all():
            results.loc[step] = step_result
        else:
            step_result.name = step
            s.results = results = results.append(step_result)

        if s.record_step:
            s.last_record_step = step
            s.last_record_state = self.get_state(s.net, s.opt, step)
            self.log('Recorded state at step %s' % step)
            s.record_step = False
        if step - s.last_record_step > self.patience:
            self.set_stopped_early()
            self.log('Stopped early after %s / %s steps' % (step, s.step_max))
            s.step_max = step
            return

        if s.writer:
            for k, v in step_result.items():
                if 'time' in k:
                    v /= 60.0 # convert seconds to minutes
                s.writer.add_scalar(k, v, global_step=step, walltime=tot_time)

        if step % self.step_save == 0 or time() - s.last_save_time >= self.time_save:
            self.save_train_results(results)
            self.save_state(step, self.get_state(s.net, s.opt, step), link_best=False)
            s.last_save_time = time()

        if step % self.step_print == 0:
            self.log(' | '.join([
                'step {:3d}'.format(step),
                '{:4.2f} mins'.format(step_result['total_train_time'] / 60),
                *('{} {:10.5g}'.format(k, v) for k, v in zip(step_result.index, step_result)
                    if k != 'total_train_time')
            ]))
            if s.progress: next(s.progress)
        sys.stdout.flush()

    def on_train_end(self, s):
        step = s.step
        if s.results is not None:
            self.save_train_results(s.results)
            s.results = None

        if s.last_record_state:
            if not self.model_save(s.last_record_step).exists():
                save_path = self.save_state(s.last_record_step, s.last_record_state, link_best=True)
            s.last_record_state = None

        # Save latest model
        if step > 0 and not self.model_save(step).exists():
            save_path = self.save_state(step, self.get_state(s.net, s.opt, step))

        if s.progress: s.progress.close()
        if s.writer: s.writer.close()

        self.set_training(False)

        import signal
        signal.signal(signal.SIGINT, s.prev_handler)

    def train(self, steps=1000000, cd=True, gpu=True, env_gpu=True, opt='O0', log=True):
        cd = ('cd %s\n' % self.res) if cd else ''

        cmd = []
        if env_gpu is False or env_gpu is None:
            cmd.append('CUDA_VISIBLE_DEVICES=')
            n_gpu = 0
        elif type(env_gpu) is int:
            cmd.append('CUDA_VISIBLE_DEVICES=%s' % env_gpu)
            n_gpu = 1
        elif type(env_gpu) in [list, tuple]:
            cmd.append('CUDA_VISIBLE_DEVICES=%s' % ','.join(map(str, env_gpu)))
            n_gpu = len(env_gpu)
        else:
            n_gpu = 4
        cmd.append('python3')

        if n_gpu > 1:
            cmd.append(
                '-m torch.distributed.launch --nproc_per_node=%s --use_env' % n_gpu
            )
        cmd.extend([
            Path(self.model).rel(self.res),
            '.',
            'steps=%s' % steps,
            'opt_level=%s' % opt
        ])

        if gpu is False or gpu is None:
            cmd.append('device=cpu')
        elif type(gpu) is int:
            cmd.append('device=cuda:%s' % gpu)

        return cd + ' '.join(cmd)

    def init_model(self, net, opt=None, step='max', train=True):
        if train:
            assert not self.training.exists(), 'Training already exists'
        # configure parallel training
        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        self.n_gpus = 0 if self.device == 'cpu' else 1 if self.device.startswith('cuda:') else len(devices.split(','))
        can_parallel = self.n_gpus > 1
        self.setdefault(distributed=can_parallel) # use distributeddataparallel
        self.setdefault(parallel=can_parallel and not self.distributed) # use dataparallel
        self.local_rank = 0
        self.world_size = 1 # number of processes
        if self.distributed:
            self.local_rank = int(os.environ['LOCAL_RANK']) # rank of the current process
            self.world_size = int(os.environ['WORLD_SIZE'])
            assert self.world_size == self.n_gpus
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.main = self.local_rank == 0

        net.to(self.device)
        if train and self.opt_level != 'O0':
            # configure mixed precision
            net, opt = amp.initialize(net, opt, opt_level=self.opt_level, loss_scale=self.get('loss_scale'))
        step = self.set_state(net, opt=opt, step=step)

        if self.distributed:
            import apex
            net = apex.parallel.DistributedDataParallel(net)
        elif self.parallel:
            net = nn.DataParallel(net)

        if train:
            net.train()
            return net, opt, step
        else:
            net.eval()
            return net, step

    @property
    def train_results(self):
        return self.res / 'train_results.csv'

    def load_train_results(self):
        if self.train_results.exists():
            return pd.read_csv(self.train_results, index_col=0)
        return None

    @main_only
    def save_train_results(self, results):
        results.to_csv(self.train_results, float_format='%.6g')


    @property
    def stopped_early(self):
        return self.res / 'stopped_early'

    @main_only
    def set_stopped_early(self):
        self.stopped_early.save_txt('')


    @property
    def training(self):
        return self.res / 'is_training'

    @main_only
    def set_training(self, is_training):
        if is_training:
            self.training.save_txt('')
        else:
            self.training.rm()


    @property
    def models(self):
        return (self.res / 'models').mk()

    def model_save(self, step):
        return self.models / ('model-%s.pth' % step)

    def model_step(self, path):
        m = re.match('.+/model-(\d+)\.pth', path)
        if m:
            return int(m.groups()[0])

    @property
    def model_best(self):
        return self.models / 'best_model.pth'

    @main_only
    def link_model_best(self, model_save):
        self.model_best.rm().link(Path(model_save).rel(self.models))

    def get_saved_model_steps(self):
        _, save_paths = self.models.ls()
        if len(save_paths) == 0:
            return []
        return sorted([x for x in map(self.model_step, save_paths) if x is not None])

    @main_only
    def clean_models(self, keep=5):
        model_steps = self.get_saved_model_steps()
        delete = len(model_steps) - keep
        keep_paths = [self.model_best._real, self.model_save(model_steps[-1])._real]
        for e in model_steps:
            if delete <= 0:
                break
            path = self.model_save(e)._real
            if path in keep_paths:
                continue
            path.rm()
            delete -= 1
            self.log('Removed model %s' % path.rel(self.res))

    def set_state(self, net, opt=None, step='max', path=None):
        state = self.load_state(step=step, path=path)
        if state is None:
            return 0
        if self.get('append_module_before_load'):
            state['net'] = dict(('module.' + k, v) for k, v in state['net'].items())
        net.load_state_dict(state['net'])
        if opt and 'opt' in state:
            opt.load_state_dict(state['opt'])
        if 'amp' in state and self.opt_level != 'O0':
            amp.load_state_dict(state['amp'])
        return state['step']

    @main_only
    def get_state(self, net, opt, step):
        try:
            net_dict = net.module.state_dict()
        except AttributeError:
            net_dict = net.state_dict()
        state = dict(step=step, net=net_dict, opt=opt.state_dict())
        try:
            state['amp'] = amp.state_dict()
        except:
            pass
        return to_torch(state, device='cpu')

    def load_state(self, step='max', path=None):
        '''
        step: best, max, integer, None if path is specified
        path: None if step is specified
        '''
        if path is None:
            if step == 'best':
                path = self.model_best
            else:
                if step == 'max':
                    steps = self.get_saved_model_steps()
                    if len(steps) == 0:
                        return None
                    step = max(steps)
                path = self.model_save(step)
        save_path = Path(path)
        if save_path.exists():
            return to_torch(torch.load(save_path), device=self.device)
        return None

    @main_only
    def save_state(self, step, state, clean=True, link_best=False):
        save_path = self.model_save(step)
        torch.save(state, save_path)
        self.log('Saved model %s at step %s' % (save_path, step))
        if clean and self.get('max_save'):
            self.clean_models(keep=self.max_save)
        if link_best:
            self.link_model_best(save_path)
            self.log('Linked %s to new saved model %s' % (self.model_best, save_path))
        return save_path

import enlighten

progress_manager = enlighten.get_manager()
active_counters = []

class Progress(object):

    def __init__(self, total, desc='', leave=False):
        self.counter = progress_manager.counter(total=total, desc=desc, leave=leave)
        active_counters.append(self.counter)

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()

    def close(self):
        self.counter.close()
        if self.counter in active_counters:
            active_counters.remove(self.counter)
        if len(active_counters) == 0:
            progress_manager.stop()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

class RangeProgress(Progress):
    def __init__(self, start, end, step=1, desc='', leave=False):
        self.i = start
        self.start = start
        self.end = end
        self.step = step
        super(RangeProgress, self).__init__((end - start) // step, desc=desc, leave=leave)

    def __next__(self):
        if self.i != self.start:
            self.counter.update()
        if self.i == self.end:
            self.close()
            raise StopIteration()
        i = self.i
        self.i += self.step
        return i

### Paths ###
Proj = Path(__file__)._up
Cache = Proj / 'cache'
Distiller = Proj / 'distiller'
Data = Proj / 'data'

Res = (Proj / 'results').mk()