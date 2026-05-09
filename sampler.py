import torch
import random
from torch.utils.data import Sampler



class BucketBatchSampler(Sampler):
    def __init__(self, metadata, batch_size, ratio):
        self.batch_size = batch_size
        self.buckets = {
            k: [i for i, m in enumerate(metadata) if m["bucket"] == k]
            for k in ratio
        }
        self.ratio = ratio

    def __iter__(self):
        while True:
            batch = []
            for k, r in self.ratio.items():
                k_count = int(self.batch_size * r)
                batch += random.sample(self.buckets[k], k_count)
            random.shuffle(batch)
            yield batch
