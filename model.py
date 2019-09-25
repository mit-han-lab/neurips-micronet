from u import *
from ut import *
from data import *

from apex import amp

class AdaptiveEmbedding(nn.Module):
    def __init__(self, c):
        super(AdaptiveEmbedding, self).__init__()
        self.c = c

        if c.get('n_embeds'):
            n_embeds = c.n_embeds
        else:
            c.n_embeds = n_embeds = [c.n_embed // (c.adaptive_ratio ** i) for i in range(len(c.cutoffs) + 1)]
        assert n_embeds[0] == c.n_embed

        self.layers = nn.ModuleList(
            nn.Embedding(end - start, n_embed_i) for n_embed_i, start, end in zip(
                n_embeds, [0] + c.cutoffs, c.cutoffs + [c.n_vocab]
            )
        )
        self.projections = nn.ModuleList(
            nn.Linear(n_embed_i, c.n_embed, bias=False) for n_embed_i in n_embeds[1:]
        )
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0, 0.02)
        
    def forward(self, x):
        c = self.c
        x_flat = x.reshape(-1)

        emb_flat = torch.zeros([x_flat.size(0), c.n_embed], dtype=torch.float if c.opt_level == 'O0' else torch.half, device=x.device)
        for i, (layer, start, end) in enumerate(zip(self.layers, [0] + c.cutoffs, c.cutoffs + [c.n_vocab])):
            mask_i = (x_flat >= start) & (x_flat < end)
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue

            inp_i = x_flat.index_select(0, indices_i) - start
            emb_i = layer(inp_i)
            if i == 0:
                if c.opt_level != 'O0':
                    emb_i = emb_i.half()
            else:
                emb_i = self.projections[i - 1](emb_i)
            emb_flat.index_copy_(0, indices_i, emb_i)

        emb = emb_flat.view(*x.size(), c.n_embed)
        emb.mul_(np.sqrt(c.n_embed))
        return emb

class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, c):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.c = c.setdefault(use_cache=False)
        n_layers = len(c.cutoffs) + 1

        if c.get('n_embeds'):
            n_embeds = c.n_embeds
        else:
            n_embeds = [c.n_embed // (c.adaptive_ratio ** i) for i in range(n_layers)]
        assert n_embeds[0] == c.n_embed
        assert n_layers == len(n_embeds)

        # the first layer gets (n_layers - 1) more classes to determine if token belongs in those layers
        self.clusters = nn.Linear(c.n_embed, n_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n_embed_i, end - start) for n_embed_i, start, end in zip(
                n_embeds, [0] + c.cutoffs, c.cutoffs + [c.n_vocab]
            )
        )
        self.projections = nn.ModuleList(
            nn.Linear(c.n_embed, n_embed_i, bias=False) for n_embed_i in n_embeds[1:]
        )
        self.cache_keys = self.cache_values = None

    def query_cache(self, hidden, target):
        # assume n_batch = 1
        c = self.c

        n_seq = hidden.size(0)
        cache_keys, cache_values = self.cache_keys, self.cache_values
        if cache_keys is None:
            n_cache = 0
            cache_keys = hidden
            cache_values = target
        else:
            n_cache = cache_keys.size(0)
            cache_keys = torch.cat((cache_keys, hidden))
            cache_values = torch.cat((cache_values, target))
        cache_keys = cache_keys[-(c.n_cache + n_seq):]
        cache_values = cache_values[-(c.n_cache + n_seq):]
        
        attn = c.cache_theta * F.pad(hidden.mm(cache_keys.t()), (c.n_cache - n_cache, 0), value=-np.inf) # (n_s, n_c + n_s)
        # (n_s, n_cache)
        probs = attn.reshape(-1).unfold(0, c.n_cache, attn.size(1) + 1).softmax(dim=1)
        indices = F.pad(cache_values, (c.n_cache - n_cache, 0)).unfold(0, c.n_cache, 1)[:n_seq]

        mask = indices != target[:, None]
        probs.masked_fill_(mask, 0)

        pos_prob = probs.sum(dim=1)
        self.cache_keys = cache_keys[-c.n_cache:]
        self.cache_values = cache_values[-c.n_cache:]
        return pos_prob
        
    def forward(self, hidden, target, keep_order=False):
        # hidden: (n_seq * n_batch, n_embed)
        # target: (n_seq * n_batch)
        c = self.c
        assert hidden.size(0) == target.size(0), 'Input and target should have the same size in the batch dimension'
        
        if c.use_cache:
            cache_prob = self.query_cache(hidden, target)

        head_logit = torch.cat((self.layers[0](hidden), self.clusters(hidden)), dim=1)
        head_logprob = head_logit.log_softmax(dim=1)

        nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

        hiddens = {}
        offset = 0
        for i, (start, end) in enumerate(zip([0] + c.cutoffs, c.cutoffs + [c.n_vocab])):
            mask_i = (target >= start) & (target < end)
            if c.use_cache:
                cache_prob_i = cache_prob[mask_i]
            indices_i = mask_i.nonzero().squeeze()

            if indices_i.numel() == 0:
                continue
            
            target_i = (target.index_select(0, indices_i) - start)[:, None]
            head_logprob_i = head_logprob.index_select(0, indices_i)

            if i == 0:
                hiddens[i] = (hidden.detach().index_select(0, indices_i), target_i)
                logprob_i = head_logprob_i.gather(1, target_i).squeeze(1)
            else:
                hidden_i = hidden.index_select(0, indices_i)
                proj_i = self.projections[i - 1](hidden_i)
                tail_logit_i = self.layers[i](proj_i)
                tail_logprob_i = tail_logit_i.log_softmax(dim=1)
                
                logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i).squeeze(1)
                hiddens[i] = (proj_i.detach(), target_i)
            
            if c.use_cache:
                logprob_i = (c.cache_lambda * cache_prob_i + (1 - c.cache_lambda) * logprob_i.exp()).log()

            if keep_order:
                nll.index_copy_(0, indices_i, -logprob_i)
            else:
                nll[offset: offset + logprob_i.size(0)].copy_(-logprob_i)
                offset += logprob_i.size(0)

        return nll, hiddens

class Decoder(nn.Module):
    def __init__(self, c, layer_i):
        super(Decoder, self).__init__()
        self.layer_i = layer_i
        c_global = c
        c = Namespace(**c.layers[layer_i]).setdefault(
            n_embed=c.n_embed, n_inner=c.n_inner, n_head=c.n_head, n_k=c.n_k, n_v=c.n_v, n_seq=c.n_seq, dropout=c.dropout, pos_emb=c.pos_emb,
            light_conv=c.light_conv
        )
        
        n_embed = c.n_embed
        if c.light_conv:
            n_embed = c.n_embed // 2
            c.n_head = c.n_head // 2
            
            import fairseq
            c.setdefault(lc_kernel_size=c_global.get('lc_kernel_size'))
            
            self.light_conv = fairseq.modules.LightweightConv(
                n_embed, c.lc_kernel_size, padding_l=c.lc_kernel_size - 1, 
                weight_softmax=False, num_heads=c.n_head, weight_dropout=0
            )
        
        self.ln1 = nn.LayerNorm(n_embed)
        
        self.qkv = nn.Linear(n_embed, c.n_head * (2 * c.n_k + c.n_v))
        if c.pos_emb == 'trained':
            self.pos_emb = nn.Parameter(torch.Tensor(c.n_k, c.n_seq + 1))
            nn.init.normal_(self.pos_emb, 0, 0.02)

        self.out = nn.Linear(c.n_head * c.n_v, n_embed, bias=False)
        self.dropout = nn.Dropout(c.dropout)

        self.ln2 = nn.LayerNorm(c.n_embed)
        self.fc = nn.Sequential(
            nn.Linear(c.n_embed, c.n_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(c.dropout),
            nn.Linear(c.n_inner, c.n_embed),
            nn.Dropout(c.dropout),
        )
        self.c = c
    
    def forward(self, x, prev=None):
        # x: (n_group * n_seq, n_batch, n_embed)
        # pos_emb: (n_k, n_seq + 1)
        # mask: (2 * n_seq, 2 * n_seq) parallelogram
        
        c = self.c
        n_s = min(c.n_seq, x.size(0))
        n_g = x.size(0) // n_s
        n_b = x.size(1)
        n_h = c.n_head
        n_k = c.n_k
        n_v = c.n_v

        if c.light_conv:
            if prev is None:
                incremental_state = {} # create new lightconv state
            else:
                prev, incremental_state = prev
            x, x2 = x.chunk(2, dim=2)
            out2 = self.light_conv(x2.transpose(0, 1).contiguous(), incremental_state=incremental_state)

        qkv = self.qkv(self.ln1(x)).reshape(n_g * n_s, n_b * n_h, 2 * n_k + n_v)
        q, kv = qkv.split([n_k, n_k + n_v], dim=-1)
        
        q = q.reshape(n_g, n_s, n_b * n_h, n_k)

        padding = prev or torch.zeros((n_s, n_b * n_h, n_k + n_v), dtype=kv.dtype, device=kv.device)
        kv = torch.cat((padding, kv))
        k, v = kv.unfold(0, 2 * n_s, n_s).split([n_k, n_v], dim=2) # (n_g, n_bh, n_kv, 2 * n_s)

        qk = torch.einsum('gsbk,gbkt->gbst', q, k) # (n_g, n_bh, n_s, 2 * n_s)
        qk = qk.reshape(n_g, n_b * n_h, -1).unfold(2, n_s + 1, 2 * n_s + 1) # (n_g, n_bh, n_s, n_s + 1)

        pos_emb = self.pos_emb
        qe = torch.einsum('gsbk,kt->gbst', q, pos_emb.to(q.dtype))

        attn = qk + qe
        attn.mul_(n_k ** -0.5)
        
        if prev is None:
            mask = torch.triu(torch.ones(qk.shape[2:], dtype=torch.uint8, device=qk.device), 1).flip([1])
            qk[0].masked_fill_(mask, -np.inf)
        attn.softmax(dim=-1)

        attn = F.pad(attn, (0, n_s))
        attn = attn.reshape(n_g, n_b * n_h, -1).unfold(2, 2 * n_s, 2 * n_s) # (n_g, n_bh, n_s, 2 * n_s)

        attnv = torch.einsum('gbst,gbvt->gsbv', attn, v) # (n_g, n_s, n_bh, n_v)
        attn_out = self.out(attnv.reshape(n_g * n_s, n_b, n_h * n_v)) # (n_g * n_s, n_b, n_embed)
        attn_out = self.dropout(attn_out)

        out = x + attn_out
        
        next = kv[-n_s:].detach()
        if c.light_conv:
            out = torch.cat((out, out2.transpose(0, 1)), dim=2)
            next = next, incremental_state
        out = out + self.fc(self.ln2(out))

        return out, next

class Transformer(nn.Module):
    def __init__(self, c):
        super(Transformer, self).__init__()
        self.c = c.setdefault(layers=[{} for _ in range(c.n_layers)], light_conv=False)
        self.embed = AdaptiveEmbedding(c)

        self.dropout = nn.Dropout(c.dropout)

        self.layers = nn.ModuleList(Decoder(c, i) for i in range(c.n_layers))

        self.loss = ProjectedAdaptiveLogSoftmax(c)

        # tie output embedding weights to input embedding weights
        for layer_embed, layer_loss in zip(self.embed.layers, self.loss.layers):
            layer_loss.weight = layer_embed.weight

    def forward(self, inputs, labels, prevs=None):
        # inputs: (n_group * n_seq, n_batch)
        # labels: (n_group * n_seq, n_batch)
        c = self.c

        n_gs = inputs.size(0)
        n_s = c.n_seq
        if n_gs % n_s != 0:
            padding = torch.zeros((n_s - n_gs % n_s, inputs.size(1)), dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat((inputs, padding))
            labels = torch.cat((labels, padding))

        x = self.embed(inputs)
        x = self.dropout(x)

        prevs = prevs or [None] * len(self.layers)
        new_prevs = []
        for layer, prev in zip(self.layers, prevs):
            x, prev = layer(x, prev=prev)
            new_prevs.append(prev)
        
        x = self.dropout(x)

        loss, hiddens = self.loss(x.reshape(-1, x.size(2)), labels.reshape(-1))
        loss = loss.reshape(labels.shape)[:n_gs]
        return dict(loss=loss.mean(), state=new_prevs, hiddens=hiddens)

class RNN(nn.Module):
    def __init__(self, c):
        super(RNN, self).__init__()
        self.c = c
        self.embed = AdaptiveEmbedding(c)

        self.dropout = nn.Dropout(c.dropout)

        LSTM = nn.LSTM
        GRU = nn.GRU
        self.rnn = eval(c.net)(c.n_embed, c.n_hidden, num_layers=c.num_layers, dropout=c.dropout)
        self.fc = nn.Sequential(
            nn.Linear(c.n_hidden, c.n_embed),
            nn.ReLU(inplace=True),
            nn.LayerNorm(c.n_embed)
        )            

        self.loss = ProjectedAdaptiveLogSoftmax(c)

        # tie output embedding weights to input embedding weights
        for layer_embed, layer_loss in zip(self.embed.layers, self.loss.layers):
            layer_loss.weight = layer_embed.weight

    def forward(self, inputs, labels, prevs=None):
        x = self.embed(inputs)
        x, state = self.rnn(x, prevs)
        
        x = self.fc(x)
        x = self.dropout(x)
        
        loss, hiddens = self.loss(x.reshape(-1, x.size(2)), labels.reshape(-1))
        loss = loss.reshape(labels.shape)
        return dict(loss=loss.mean(), state=state, hiddens=hiddens)

class UniversalTransformer(nn.Module):
    def __init__(self, c):
        super(Transformer, self).__init__()
        self.c = c.setdefault(layers=[{} for _ in range(c.n_layers)], light_conv=False)
        self.embed = AdaptiveEmbedding(c)

        self.dropout = nn.Dropout(c.dropout)

        self.should_halt = nn.Linear(c.n_embed, 1)

        self.layer = Decoder(c, 0)

        self.loss = ProjectedAdaptiveLogSoftmax(c)

        # tie output embedding weights to input embedding weights
        for layer_embed, layer_loss in zip(self.embed.layers, self.loss.layers):
            layer_loss.weight = layer_embed.weight

    def forward(self, inputs, labels, prevs=None):
        # inputs: (n_gs, n_b)

        c = self.c

        n_gs = inputs.size(0)
        n_s = c.n_seq
        if n_gs % n_s != 0:
            padding = torch.zeros((n_s - n_gs % n_s, inputs.size(1)), dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat((inputs, padding))
            labels = torch.cat((labels, padding))
        
        x = self.embed(inputs)
        x = self.dropout(x)

        p_halt = torch.zeros(n_gs, dtype=x.dtype, device=x.device)
        i_loop = 0
        while i_loops < c.n_loops and (p_halt < c.thres_halt).any():
            p_halt_i = self.should_halt(x).sigmoid()
            running = p_halt < 1
            
            new_halted = running & ((p_halt + p_halt_i) > c.thres_halt)
            running = running & ~new_halted

            p_halt = p_halt + p_halt_i
            i_loops += 1


        total_p_halt = 0





        prevs = prevs or [None] * len(self.layers)
        new_prevs = []
        for layer, prev in zip(self.layers, prevs):
            x, prev = layer(x, prev=prev)
            new_prevs.append(prev)
        
        x = self.dropout(x)

        loss, hiddens = self.loss(x.reshape(-1, x.size(2)), labels.reshape(-1))
        loss = loss.reshape(labels.shape)[:n_gs]
        return dict(loss=loss.mean(), state=new_prevs, hiddens=hiddens)

get_net = lambda c: eval(c.model_class)(c)
get_opt = lambda c, net: optim.Adam(net.parameters(), lr=c.lr)

def scheduler(c, opt, step):
    c.setdefault(
        scheduler=None,
        step_warmup=0,
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

def hebbian_weight_update(c, net, hiddens, counters, temp_counters):
    with torch.no_grad():
        for i, (hidden_i, target_i) in hiddens.items():
            target_i = target_i.reshape(-1)
            c_i = counters[i]
            n_i = temp_counters[i]
            n_cls = c_i.size(0)
            lam = (1 / c_i.float()).clamp(min=c.hebbian_gamma) * (c_i < c.hebbian_T).float()

            n_i.index_add_(0, target_i, torch.ones_like(target_i))

            weight = (net.module if c.distributed else net).loss.layers[i].weight.data
            
            h_sums = torch.zeros_like(weight)
            hidden_i = hidden_i * weight[target_i].norm(dim=1, keepdim=True).to(hidden_i.dtype) / hidden_i.norm(dim=1, keepdim=True)
            h_sums.index_add_(0, target_i, hidden_i.to(h_sums.dtype))

            if c.distributed:
                all_h_sums = [torch.zeros_like(h_sums) for _ in range(c.world_size)]
                torch.distributed.all_gather(all_h_sums, h_sums)
                
                all_n_is = [torch.zeros_like(n_i) for _ in range(c.world_size)]
                torch.distributed.all_gather(all_n_is, n_i)
                h_sums = sum(all_h_sums)
                n_i = sum(all_n_is)
            
            c_i += n_i # update total count

            mask = n_i > 0
            h_sums = h_sums[mask]
            n_i = n_i[mask]

            h_means = lam[mask][:, None] * h_sums / n_i[:, None].to(h_sums.dtype) # divide by mean then scale by lambda
            
            weight[mask].mul_(1 - lam[mask][:, None]) # scale by 1 - lambda
            weight[mask].add_(h_means)
            
            n_i[:] = 0

def train(c):
    c.setdefault(hebbian=False)
    net = get_net(c)

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

    if c.hebbian:
        counters = [torch.ones(end - start, dtype=torch.long, device=c.device) for start, end in zip([0] + c.cutoffs, c.cutoffs + [c.n_vocab])]
        temp_counters = [torch.zeros_like(x) for x in counters]

    best_val_loss = np.inf
    if s.results is not None and 'val_loss' in s.results.columns:
        best_val_loss = s.results['val_loss'].dropna().max()
    try:
        while step < s.step_max:
            step_lr(step)

            x = to_torch(next(iter_tr), c.device).t()

            t_s = time()
            inputs, labels = x[:-1], x[1:]
            preds = net(inputs, labels)
            loss = preds['loss']

            opt.zero_grad()
            skip_gradient = loss.abs() > 1e3
            if torch.isnan(loss):
                raise RuntimeError('Encountered nan loss during training')
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            if skip_gradient:
                c.log('Loss magnitude %s, skipping' % from_torch(loss))
                opt.zero_grad()
            torch.nn.utils.clip_grad_norm_(net.parameters(), c.get('clip_grad', 0.5))
            opt.step()

            if c.hebbian:
                hebbian_weight_update(c, net, preds['hiddens'], counters, temp_counters)

            time_model = np.round(time() - t_s, 5)

            loss = from_torch(loss)
            perplexity = np.nan if loss > 5 else np.e ** loss
            step_result = pd.Series(Dict(
                loss=loss,
                perplexity=perplexity,
                time=time_model
            )).add_prefix('train_')
            step_result['lr'] = next(iter(opt.param_groups))['lr']

            s.step = step = step + 1
            if step % c.step_eval == 0:
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

def evaluate(c, data, net):
    clear_gpu_memory()
    was_training = net.training
    net.eval()
    
    t_s = time()
    with torch.no_grad():
        weights = []
        losses = []
        prevs = None

        for batch in data:
            x = to_torch(batch, c.device).t()
            inputs, labels = x[:-1], x[1:]

            preds = net.forward(inputs, labels, prevs=prevs)
            losses.append(preds['loss'])
            weights.append(labels.size(0))
        weights = np.array(weights)
        weights = weights / weights.sum()
        loss = sum(x * w for x, w in zip(losses, weights))

    if c.distributed:
        gathered_losses = [torch.zeros_like(loss) for _ in range(c.world_size)]
        torch.distributed.all_gather(gathered_losses, loss)
        loss = sum(gathered_losses) / len(gathered_losses)
    if was_training:
        net.train()
    loss = from_torch(loss)
    perplexity = np.nan if loss > 5 else np.e ** loss
    return dict(loss=loss, perplexity=perplexity, time=np.round(time() - t_s))

if __name__ == '__main__':
    config = Config.from_args()
    if config.get('eval_only'):
        net, step = init(config, step=config.eval_only, train=False)
        gen = data_val(config)
        out = evaluate(config, gen, net)
        if config.main:
            (config.res / 'test.json').save(out)
            qq.d()
    else:
        train(config)