from u import *
from model import Transformer

class SequentialIteratorGenSoft:
    def __init__(self, c, batch_size):
        self.c = c
        self.tokens = self.tokens = (Cache / 'wikitext-103' / 'sorted_train.npy').load()
        self.batch_size = batch_size

    def __iter__(self):
        c = self.c
        bs = self.batch_size
        tokens = self.tokens

        n = len(tokens) - 1

        start, end = 0, n
        if c.distributed:
            start = n * c.local_rank // c.world_size
            end = n * (c.local_rank + 1) // c.world_size

        span_i = (end - start) // bs
        span = span_i * bs
        end = start + span

        starts = np.arange(start, end, span_i)
        for i in range(0, span_i, c.eval_chunk):
            starts_i = starts + i
            batch_i = np.array([tokens[s_i: s_i + c.eval_chunk + 1] for s_i in starts_i])
            yield batch_i.astype(np.int64)

    def __len__(self):
        n = len(self.tokens) - 1
        bs = self.batch_size
        start, end = 0, n
        span_i = (end - start) // bs

        return span_i // self.c.eval_chunk

def gen_soft_labels(c):
    c.setdefault(hebbian=False, distributed=False)
    net = Transformer(c)
    net, step = c.init_model(net, step='max', train=False)

    print('generating soft labels...')
    data_gen_tr = SequentialIteratorGenSoft(c, 1)
    net.eval()
    with torch.no_grad():
        i = 0
        for batch in tqdm(data_gen_tr):
            x = to_torch(batch, c.device).t()
            inputs, labels = x[:-1], x[1:]
            probs, _ = net(inputs, labels)

            values, indices = torch.topk(probs, c.topk, dim=1)

            indices_ = indices.cpu().numpy()
            values_ = values.cpu().numpy()
            labels_ = labels.cpu().numpy()

            if probs.size(0) != inputs.size(0):
                indices_ = indices_[-inputs.size(0):, :]
                values_ = values_[-inputs.size(0):, :]

            if i == 0:
                all_soft_indices = indices_
                all_soft_values = values_
            else:
                all_soft_indices = np.concatenate((all_soft_indices, indices_), axis=0)
                all_soft_values = np.concatenate((all_soft_values, values_), axis=0)

            i += 1
    all_soft_indices = np.concatenate((all_soft_indices[0:1, :], all_soft_indices), axis=0)
    all_soft_values = np.concatenate((all_soft_values[0:1, :], all_soft_values), axis=0)

    np.save(Cache / 'wikitext-103' / 'train_soft_labels.npy', all_soft_indices)
    np.save(Cache / 'wikitext-103' / 'train_soft_probs.npy', all_soft_values)
    print('Saved %s' % (Cache / 'wikitext-103' / 'train_soft_labels.npy'))
    print('Saved %s' % (Cache / 'wikitext-103' / 'train_soft_probs.npy'))

    cnt = 0.
    for k in range(len(data_gen_tr.tokens)):
        if data_gen_tr.tokens[k] in all_soft_indices[k]:
            cnt += 1
    print('%s%% of the tokens are predicted within the top %s logits' % (100 * cnt / len(data_gen_tr.tokens), c.topk))

if __name__ == '__main__':
    config = Config.from_args()
    gen_soft_labels(config)
