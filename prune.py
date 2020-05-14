from u import *
from model import Transformer
from modules import hebbian_weight_update
from optim import scheduler, get_opt
from data import SampleIterator, SequentialIterator, DistillationSampleIterator, evaluate

sys.path.append(Distiller)
import distiller

def train(c, net, compression_scheduler=None):
    import distiller.apputils as apputils
    from distiller.data_loggers import TensorBoardLogger, PythonLogger
    msglogger = apputils.config_pylogger('logging.conf', None)
    tflogger = TensorBoardLogger(msglogger.logdir)
    tflogger.log_gradients = True
    pylogger = PythonLogger(msglogger)
    c.setdefault(hebbian=False)

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

            if compression_scheduler:
                _  = compression_scheduler.before_backward_pass(epoch, minibatch_id=batch,
                                                           minibatches_per_epoch=steps_per_epoch,
                                                           loss=loss, return_loss_components=False)

            opt.zero_grad()
            if torch.isnan(loss):
                raise RuntimeError('Encountered nan loss during training')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), c.get('clip_grad', 0.5))
            opt.step()

            if c.hebbian:
                hebbian_weight_update(c, net, preds['hiddens'], counters, temp_counters)

            time_model = np.round(time() - t_s, 5)

            loss = from_torch(loss)
            perplexity = np.nan if loss > 5 else np.e ** loss
            step_result = pd.Series(dict(
                loss=loss,
                perplexity=perplexity,
                time=time_model
            )).add_prefix('train_')
            step_result['lr'] = next(iter(opt.param_groups))['lr']
            step_result['theta'] = preds['theta']
            step_result['lambda'] = preds['lambda'].item()

            if compression_scheduler:
                compression_scheduler.on_minibatch_end(epoch, minibatch_id=batch, minibatches_per_epoch=steps_per_epoch)

            if step % steps_per_epoch == 0:
                if compression_scheduler:
                    compression_scheduler.on_epoch_end(epoch)

            s.step = step = step + 1
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
            c.on_step_end(s)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        if c.main:
            c.log(err)
        else:
            print(err)
    finally:
        c.on_train_end(s)

if __name__ == '__main__':
    c = Config.from_args()
    print(c)
    net = Transformer(c)

    if c.get("summary"):
        net, step = c.init_model(net, step=c.get('step', 'max'), train=False)
        c.log("===> summary of model @ step %d" % step)
        distiller.model_summary(net, c.summary, 'wikitext-103')
        exit(0)

    if c.get("compress"):
        c.log("===> compress from: %s" % c.compress)
        compression_scheduler = distiller.config.file_config(net, None, c.compress)

    train(c, net, compression_scheduler)
