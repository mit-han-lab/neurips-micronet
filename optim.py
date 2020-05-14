from u import *

get_opt = lambda c, net: optim.Adam(net.parameters(), lr=c.lr)

def scheduler(c, opt, step):
    c.setdefault(
        scheduler=None,
        step_warmup=-1,
    )
    get_lr = lambda step: c.lr * min(1, max(step, 1) / max(1, c.step_warmup))
    def step_lr(step):
        lr = get_lr(step)
        for g in opt.param_groups:
            g['lr'] = float(lr)
    if c.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, c.steps, last_epoch=step or step - 1)
        warmup_step_lr = step_lr
        def step_lr(step):
            if step <= c.step_warmup:
                warmup_step_lr(step)
            else:
                scheduler.step(step)
    elif c.scheduler == 'rsqrt':
        warmup = get_lr
        get_lr = lambda step: warmup(step) / np.sqrt(max(1, c.step_warmup, step))
    return step_lr
