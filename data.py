from u import *

class SampleIterator:
    def __init__(self, c, batch_size, split):
        self.c = c.setdefault(train_chunk=c.n_seq)
        self.tokens = (Cache / 'wikitext-103' / 'sorted_' + split + '.npy').load()
        self.bs = batch_size
        np.random.seed(c.local_rank)

    def __iter__(self):
        c = self.c
        rand_max = (len(self.tokens) - c.train_chunk + 1) - 1 # -1 because actually returning train_chunk + 1 per sequence
        while True:
            starts = np.random.randint(rand_max, size=self.bs)
            batch = np.array([self.tokens[start: start + c.train_chunk + 1] for start in starts])
            yield batch.astype(np.int64)

class SequentialIterator:
    def __init__(self, c, batch_size, split):
        self.c = c
        self.tokens = (Cache / 'wikitext-103' / 'sorted_' + split + '.npy').load()
        self.batch_size = batch_size

        assert c.eval_chunk % c.n_seq == 0
        n = len(self.tokens) - 1

        start, end = 0, n
        if c.distributed:
            start = n * c.local_rank // c.world_size
            end = n * (c.local_rank + 1) // c.world_size

        if batch_size > 1:
            span_i = (end - start) // batch_size // c.eval_chunk * c.eval_chunk
            span = span_i * batch_size
            end = start + span
        else:
            span_i = end - start
        self.span_i = span_i
        self.starts = np.arange(start, end, span_i)

    def __iter__(self):
        c = self.c
        tokens = self.tokens
        starts = self.starts

        for i in range(0, self.span_i, c.eval_chunk):
            starts_i = starts + i
            batch_i = np.array([tokens[s_i: s_i + c.eval_chunk + 1] for s_i in starts_i])
            yield batch_i.astype(np.int64)

    def __len__(self):
        return len(range(0, self.span_i, self.c.eval_chunk))

class DistillationSampleIterator:
    def __init__(self, c, batch_size):
        self.c = c.setdefault(train_chunk=c.n_seq)
        self.tokens = (Cache / 'wikitext-103' / 'sorted_train.npy').load()
        self.soft_labels = (Cache / 'wikitext-103' / 'train_soft_labels.npy').load()
        self.soft_probs = (Cache / 'wikitext-103' / 'train_soft_probs.npy').load()
        self.bs = batch_size
        np.random.seed(c.local_rank)

    def __iter__(self):
        c = self.c
        rand_max = (len(
            self.tokens) - c.train_chunk + 1) - 1  # -1 because actually returning train_chunk + 1 per sequence
        while True:
            starts = np.random.randint(rand_max, size=self.bs)
            batch = np.array([self.tokens[start: start + c.train_chunk + 1] for start in starts])
            batch_soft_labels = np.array([self.soft_labels[start: start + c.train_chunk + 1] for start in starts])
            batch_soft_probs = np.array([self.soft_probs[start: start + c.train_chunk + 1] for start in starts])

            yield [batch.astype(np.int64), batch_soft_labels, batch_soft_probs]

def evaluate(c, data, net):
    clear_gpu_memory()
    was_training = net.training
    net.eval()
    net.loss.cache_keys = net.loss.cache_values = None

    t_s = time()
    with torch.no_grad():
        weights = []
        losses = []
        prevs = None

        for batch in data:
            x = to_torch(batch, c.device).t()
            inputs, labels = x[:-1], x[1:]

            preds = net.forward(inputs, labels, prevs=prevs)
            prevs = preds['state']
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