from u import *
import model
import quantize
from copy import deepcopy
from data import SequentialIterator, evaluate

c = Config.from_args()
print('Cache search on %s with cache size %s' % (c.res, c.n_cache))
net = eval(c.get('model', 'model.Transformer'))(c)
net, step = c.init_model(net, step=c.get('step', 'max'), train=False)
print('Loaded model from step', step)

data = SequentialIterator(c, c.eval_batch, split='valid')
for k in 'cache_theta_init', 'cache_lambda_init':
    if c.get(k):
        c.unvar(k)
state = net.state_dict()
net.loss.cache_keys = net.loss.cache_values = None
if 'loss.cache_lambda_inv_sigmoid' in state:
    lambda_init = from_torch(state['loss.cache_lambda_inv_sigmoid'].sigmoid())
    theta_init = from_torch(F.softplus(state['loss.cache_theta_inv_softplus']))
    print('trained lambda', lambda_init)
    print('trained theta', theta_init)
else: # initial cache parameters if we didn't train cache parameters
    lambda_init = 0.1
    theta_init = 0.016
    print('initial lambda', lambda_init)
    print('initial theta', theta_init)

c.var(use_cache=True, n_cache=c.n_cache, cache_lambda=lambda_init, cache_theta=theta_init)

ppl = evaluate(c, data, net)['perplexity']
lam, theta = lambda_init, theta_init
lam_delta, theta_delta = lam / 20, theta / 20
print('Initial val PPL=%s    lambda=%.3g (%.3g)    theta=%.3g (%.3g)' % (ppl, lam, lam_delta, theta, theta_delta))
while True:
    ppl_plus = evaluate(c.var(cache_lambda=lam + lam_delta, cache_theta=theta), data, net)['perplexity']
    ppl_minus = evaluate(c.var(cache_lambda=lam - lam_delta, cache_theta=theta), data, net)['perplexity']
    new_ppl, new_lam = min((ppl_plus, lam + lam_delta), (ppl_minus, (lam - lam_delta)))
    if new_ppl < ppl:
        ppl, lam = new_ppl, new_lam
        lam_delta *= 1.2
    else:
        lam_delta *= 0.5
    print('PPL=%s    lambda=%.3g (%.3g)    theta=%.3g (%.3g)' % (ppl, lam, lam_delta, theta, theta_delta))

    ppl_plus = evaluate(c.var(cache_lambda=lam, cache_theta=theta + theta_delta), data, net)['perplexity']
    ppl_minus = evaluate(c.var(cache_lambda=lam, cache_theta=theta - theta_delta), data, net)['perplexity']
    new_ppl, new_theta = min((ppl_plus, theta + theta_delta), (ppl_minus, (theta - theta_delta)))
    if new_ppl < ppl:
        ppl, theta = new_ppl, new_theta
        theta_delta *= 1.2
    else:
        theta_delta *= 0.5
    print('PPL=%s    lambda=%.3g (%.3g)    theta=%.3g (%.3g)' % (ppl, lam, lam_delta, theta, theta_delta))
    if lam_delta / lam < 0.005 and theta_delta / theta < 0.005:
        break
(c.res / ('cache_step%s_n%s.yaml' % (step, c.n_cache))).save(dict(use_cache=True, n_cache=c.n_cache, cache_lambda=from_torch(lam), cache_theta=from_torch(theta)))