from apex import amp

from u import *

'''
python model.py path

root
    model.py
    data.py
    util/util.py
    u/
        __init__.py
    *.ipynb
    exp_group1/
        exp1/
            config.yaml
            src/
                model.py (copy)
                data.py (copy)
                util/ (copy)
            models/
            tb_events

    exp_group2/
'''
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
    
    never_save = {'res', 'name', 'main', 'logger', 'distributed', 'parallel', 'device', 'steps', 'debug'}
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
    def load_all(cls, *directories, df=False, kwargs={}):
        configs = []
        def dir_fn(d):
            c = Config(d, **kwargs)
            if not c.path.exists():
                return
            configs.append(c.load())
        for d in map(Path, directories):
            d.recurse(dir_fn)
        if not df:
            return configs
        config_map = {c: c.attrs_save for c in configs}
        return pd.DataFrame(config_map).T.fillna('')
    
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
            from u.progress_bar import RangeProgress
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
        self.n_gpus = 0 if self.device == 'cpu' else 1 if self.device.startswith('cuda:') else len(get_gpu_info()) if devices is None else len(devices.split(','))
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
        if train:
            # configure mixed precision
            net, opt = amp.initialize(net, opt, opt_level=self.opt_level, loss_scale=self.get('loss_scale'), verbosity=int(self.opt_level != 'O0'))
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
    
    def load_model(self, step='best', train=False):
        '''
        step can be 'best', 'max', an integer, or None
        '''
        model = import_module('model', str(self.model))
        net = model.get_net(self)
        opt = model.get_opt(self, net) if train else None
        return self.init_model(net, opt=opt, step=step, train=train)

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
            state['net'] = OrderedDict(('module.' + k, v) for k, v in state['net'].items())
        net.load_state_dict(state['net'])
        if opt:
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
        return to_torch(dict(step=step, net=net_dict, opt=opt.state_dict(), amp=amp.state_dict()), device='cpu')

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
