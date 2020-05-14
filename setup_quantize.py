from u import *
from copy import deepcopy

c_base = Config.from_args().setdefault(device='cuda', logger=False, step='max', n_cache=0)
state = c_base.load_state(c_base.step)
c_base.step = state['step']

out_res = c_base.res._up / '_'.join([
    'quantize',
    c_base.res._name,
    'step' + str(c_base.step),
    'cache' + str(c_base.n_cache),
    'bits' + str(c_base.bits)
])

c = deepcopy(c_base).var(res=out_res)
for k in 'hebbian', 'hebbian_T', 'hebbian_gamma', 'step', 'out_res':
    if c.get(k):
        c.unvar(k)
c = c.var(
    model='quantize.Transformer',
    compress='distiller_quantize.yaml',
    train_batch=1,
    steps_per_epoch=1000,
    step_warmup=0,
    lr=0.0001,
    step_eval=1,
    bits=c_base.bits,
    steps=1
)

cache_search_path = c_base.res / ('cache_step%s_n%s.yaml' % (c_base.step, c_base.n_cache))
if cache_search_path.exists():
    for k in 'cache_theta_init', 'cache_lambda_init':
        c.unvar(k)
    del state['net']['loss.cache_lambda_inv_sigmoid']
    del state['net']['loss.cache_theta_inv_softplus']
    params = cache_search_path.load()
    c.var(**params)
    print('Loaded cache search parameters')
    print(params)

print('step', state['step'])
del state['opt']
state['step'] = 0

c.save(True)
c.save_state(state['step'], state)

distiller_config = (Proj / 'distiller_quantize.yaml').load()
for attr in 'bits_activations', 'bits_weights', 'bits_bias':
    distiller_config['quantizers']['linear_quantizer'][attr] = c.bits
(c.res / 'distiller_quantize.yaml').save(distiller_config)

print('Created new config directory %s\nRun the following command to train the quantization\n' % c.res)
print('cd %s && CUDA_VISIBLE_DEVICES=%s python3 ../../quantize.py .' % (c.res._real, os.environ.get('CUDA_VISIBLE_DEVICES', '')))