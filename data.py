import torch
import numpy as np
from ut import Cache

class SampleIterator:
    def __init__(self, c, batch_size, split):
        self.c = c.setdefault(train_chunk=c.n_seq, vocab_sorted=False)
        self.tokens = (Cache / 'wikitext-103' / ('sorted_' if c.vocab_sorted else '') + split + '.npy').load()
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
        self.c = c.setdefault(vocab_sorted=False)
        self.tokens = (Cache / 'wikitext-103' / ('sorted_' if c.vocab_sorted else '') + split + '.npy').load()
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

        span_i = (end - start) // bs // c.eval_chunk * c.eval_chunk
        span = span_i * bs
        end = start + span

        starts = np.arange(start, end, span_i)
        for i in range(0, span_i, c.eval_chunk):
            starts_i = starts + i
            batch_i = np.array([tokens[s_i: s_i + c.eval_chunk + 1] for s_i in starts_i])
            yield batch_i.astype(np.int64)

class DistillationSampleIterator:
    def __init__(self, c, batch_size, split):
        self.c = c.setdefault(train_chunk=c.n_seq)
        assert split == 'train'
        self.tokens = (Cache / 'wikitext-103' / ('sorted_' if c.vocab_sorted else '') + split + '.npy').load()
        print(Cache / 'wikitext-103' / ('sorted_' if c.vocab_sorted else '') + split + '.npy')
        self.soft_labels = (Cache / 'wikitext-103' / split + '_soft_labels.npy').load()
        print(self.soft_labels.shape)
        self.soft_probs = (Cache / 'wikitext-103' / split + '_soft_probs.npy').load()
        print(self.soft_probs.shape)
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
