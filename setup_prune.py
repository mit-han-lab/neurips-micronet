from u import *
from copy import deepcopy

c_base = Config.from_args().setdefault(device='cuda', logger=False, step='max')
out_res = Path(c_base.get('out_res', c_base.res._up / '_'.join(['prune', c_base.res._name])))
c = deepcopy(c_base).var(res=out_res).unvar('step')
for k in 'annealing_hard_max', 'annealing_hard_min', 'distill', 'distill_first_bin', 'out_res':
    if c.get(k):
        c.unvar(k)
c = c.var(
    compress='distiller_prune.yaml',
    lr=0.0001,
    opt_level='O0',
    steps_per_epoch=1000,
    step_warmup=-1,
    step_eval=1000,
    step_save=1000,
    steps=175000
).save(True)

state = c_base.load_state(c_base.step)
print('step', state['step'])
state.pop('opt', None)
state['step'] = 0
c.save_state(state['step'], state)
distiller_config = (Proj / 'distiller_prune.yaml').load()
(c.res / 'distiller_prune.yaml').save(distiller_config)

print('Created new config directory %s, YOU MAY NEED TO CHANGE THE BATCH SIZE IN config.yaml\nRun the following command to train the pruner\n' % c.res)
print('cd %s && CUDA_VISIBLE_DEVICES=%s python3 ../../prune.py .' % (c.res._real, os.environ.get('CUDA_VISIBLE_DEVICES', '')))
