import math
import random
from typing import List, Iterator

from torch.utils.data import Sampler


class TokenBatchSampler(Sampler[List[int]]):
    """
    Groups dataset indices into batches such that
    total (src_len + tgt_len) per batch stays under a token budget.

    Uses precomputed length estimates.
    """

    def __init__(
        self,
        src_lengths: List[int],
        tgt_lengths: List[int],
        max_tokens_per_batch: int = 4000,
        shuffle: bool = True,
    ):
        assert len(src_lengths) == len(tgt_lengths)
        self.src_lengths = src_lengths
        self.tgt_lengths = tgt_lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.num_samples = len(src_lengths)

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(self.num_samples))

        if self.shuffle:
            random.shuffle(indices)

        current_batch = []
        current_tokens = 0

        for idx in indices:
            src_len = self.src_lengths[idx]
            tgt_len = self.tgt_lengths[idx]
            example_tokens = src_len + tgt_len

            # If single example already exceeds budget, just yield it alone
            if example_tokens > self.max_tokens_per_batch:
                if current_batch:
                    yield current_batch
                    current_batch = []
                    current_tokens = 0
                yield [idx]
                continue

            # If adding this example would exceed budget, yield current batch
            if current_tokens + example_tokens > self.max_tokens_per_batch:
                if current_batch:
                    yield current_batch
                current_batch = [idx]
                current_tokens = example_tokens
            else:
                current_batch.append(idx)
                current_tokens += example_tokens

        if current_batch:
            yield current_batch

    def __len__(self) -> int:
        # Approximate number of batches
        total_tokens = 0
        for s, t in zip(self.src_lengths, self.tgt_lengths):
            total_tokens += (s + t)
        return math.ceil(total_tokens / self.max_tokens_per_batch)
