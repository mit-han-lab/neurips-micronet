from u import *
from modules import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax

mask_type = torch.uint8 if torch.__version__.startswith('1.1') else torch.bool

class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()

        n_embed = c.n_embed
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

        qkv = self.qkv(self.ln1(x)).reshape(n_g * n_s, n_b * n_h, 2 * n_k + n_v)
        q, kv = qkv.split([n_k, n_k + n_v], dim=-1)

        q = q.reshape(n_g, n_s, n_b * n_h, n_k)

        padding = prev if prev is not None else torch.zeros((n_s, n_b * n_h, n_k + n_v), dtype=kv.dtype, device=kv.device)
        kv = torch.cat((padding, kv))
        k, v = kv.unfold(0, 2 * n_s, n_s).split([n_k, n_v], dim=2) # (n_g, n_bh, n_kv, 2 * n_s)

        qk = torch.einsum('gsbk,gbkt->gbst', q, k) # (n_g, n_bh, n_s, 2 * n_s)
        qk = qk.reshape(n_g, n_b * n_h, -1).unfold(2, n_s + 1, 2 * n_s + 1) # (n_g, n_bh, n_s, n_s + 1)

        pos_emb = self.pos_emb
        qe = torch.einsum('gsbk,kt->gbst', q, pos_emb.to(q.dtype))

        attn = qk + qe
        attn.mul_(n_k ** -0.5)

        if prev is None:
            mask = torch.triu(torch.ones(attn.shape[2:], dtype=mask_type, device=attn.device), 1).flip([1])
            attn[0].masked_fill_(mask, -np.inf)
        attn = attn.softmax(dim=-1)

        attn = F.pad(attn, (0, n_s))
        attn = attn.reshape(n_g, n_b * n_h, -1).unfold(2, 2 * n_s, 2 * n_s) # (n_g, n_bh, n_s, 2 * n_s)

        attnv = torch.einsum('gbst,gbvt->gsbv', attn, v) # (n_g, n_s, n_bh, n_v)
        attn_out = self.out(attnv.reshape(n_g * n_s, n_b, n_h * n_v)) # (n_g * n_s, n_b, n_embed)
        attn_out = self.dropout(attn_out)

        out = x + attn_out

        next = kv[-n_s:].detach()
        out = out + self.fc(self.ln2(out))

        return out, next

class Transformer(nn.Module):
    def __init__(self, c):
        super(Transformer, self).__init__()
        self.c = c.setdefault(quantizing=False)
        self.embed = AdaptiveEmbedding(c)

        self.dropout = nn.Dropout(c.dropout)

        self.layers = nn.ModuleList(Decoder(c) for _ in range(c.n_layers))

        self.loss = ProjectedAdaptiveLogSoftmax(c)

        # tie output embedding weights to input embedding weights
        for layer_embed, layer_loss in zip(self.embed.layers, self.loss.layers):
            layer_loss.weight = layer_embed.weight

    def forward(self, inputs, labels, prevs=None, soft_labels=None, soft_probs=None, current_step=0.):
        # inputs: (n_group * n_seq, n_batch)
        # labels: (n_group * n_seq, n_batch)
        c = self.c

        n_gs = inputs.size(0)
        n_s = c.n_seq
        if n_gs % n_s != 0:
            padding = torch.zeros((n_s - n_gs % n_s, inputs.size(1)), dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat((inputs, padding))

        x = self.embed(inputs)
        x = self.dropout(x)

        prevs = prevs or [None] * c.n_layers
        nexts = []
        for layer, prev in zip(self.layers, prevs):
            x, prev = layer(x, prev=prev)
            nexts.append(prev)

        x = self.dropout(x)
        x = x[:n_gs]

        if c.get('distill') and self.training:
            soft_labels_reshape = soft_labels.reshape(-1, soft_labels.size(2))
            soft_probs_reshape = soft_probs.reshape(-1, soft_probs.size(2))
            loss, hiddens = self.loss(hidden=x.reshape(-1, x.size(2)), target=labels.reshape(-1),
                                      soft_labels=soft_labels_reshape, soft_probs=soft_probs_reshape,
                                      current_step=current_step)
            loss = loss.reshape(labels.shape)
            extras = {}
            if c.use_cache:
                extras['lambda'] = self.loss.last_lambda
                extras['theta'] = self.loss.last_theta
            return dict(loss=loss.mean(), state=nexts, hiddens=hiddens, current_step=current_step, **extras)

        loss, hiddens = self.loss(x.reshape(-1, x.size(2)), labels.reshape(-1), keep_order=c.get('keep_order', False))
        if c.get('gen_soft'):
            return loss, hiddens

        loss = loss.reshape(labels.shape)
        if not c.get('loss_no_mean'):
            loss = loss.mean()

        extras = {}
        if c.use_cache:
            extras['lambda'] = self.loss.last_lambda
            extras['theta'] = self.loss.last_theta
        if c.quantizing:
            return loss, nexts
        return dict(loss=loss, state=nexts, hiddens=hiddens, **extras)
