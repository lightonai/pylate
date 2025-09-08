from __future__ import annotations

import os

import torch

from pylate.indexes.stanford_nlp.infra.config import ColBERTConfig
from pylate.indexes.stanford_nlp.infra.launcher import print_memory_stats
from pylate.indexes.stanford_nlp.infra.run import Run
from pylate.indexes.stanford_nlp.search.index_storage import IndexScorer


class Searcher:
    def __init__(
        self,
        index,
        checkpoint=None,
        collection=None,
        config=None,
        index_root=None,
        verbose: int = 3,
    ):
        self.verbose = verbose
        if self.verbose > 1:
            print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.config = ColBERTConfig.from_existing(self.index_config, initial_config)
        self.configure()

        self.use_gpu = self.config.total_visible_gpus > 0

        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and self.use_gpu:
            raise ValueError("Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, self.use_gpu, load_index_with_mmap)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def search(self, Q, k=10, filter_fn=None, full_length_search=False, pids=None):
        # print("Q", Q)
        # print("Q TYPE", Q[0].dtype)
        Q = torch.tensor(
            Q, dtype=torch.float16 if self.use_gpu else torch.float32
        ).unsqueeze(0)
        # Cast to bf16
        # Q = Q.to(torch.float16)
        # print("Q TYPE", Q.type())
        return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None, pids=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 8192))

        pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids)

        return pids[:k], list(range(1, k + 1)), scores[:k]
