from u import *
import torch
import torch.nn as nn    
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

Scratch = Path("/home/demi/projects")

Proj = Scratch / 'micronet'
Cache = Proj / 'cache'
Data = Proj / 'data'

Wiki = (Proj / 'wikitext-103').mk()

class Namespace(object):
    def __init__(self, *args, **kwargs):
        self.var(*args, **kwargs)
    
    def var(self, *args, **kwargs):
        kvs = Dict()
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


def get_encoder(decoder):
    return OrderedDict((x, i) for i, x in enumerate(decoder))
