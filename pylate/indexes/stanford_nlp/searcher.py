import os
from typing import Union

import torch
from tqdm import tqdm

from pylate.indexes.stanford_nlp.data import Collection, Queries, Ranking
from pylate.indexes.stanford_nlp.infra.config import ColBERTConfig
from pylate.indexes.stanford_nlp.infra.launcher import print_memory_stats
from pylate.indexes.stanford_nlp.infra.provenance import Provenance
from pylate.indexes.stanford_nlp.infra.run import Run
from pylate.indexes.stanford_nlp.modeling.checkpoint import Checkpoint
from pylate.indexes.stanford_nlp.search.index_storage import IndexScorer

TextQueries = Union[str, "list[str]", "dict[int, str]", Queries]


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

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(
            self.checkpoint_config, self.index_config, initial_config
        )
        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(
            self.checkpoint, colbert_config=self.config, verbose=self.verbose
        )
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and use_gpu:
            raise ValueError("Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, use_gpu, load_index_with_mmap)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries, full_length_search=False):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(
            queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search
        )

        return Q

    def search(self, Q, k=10, filter_fn=None, full_length_search=False, pids=None):
        # Q = self.encode(text, full_length_search=full_length_search)
        Q = torch.tensor(Q).unsqueeze(0)
        # Cast to bf16
        Q = Q.to(torch.float16)
        return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)

    def search_all(
        self,
        queries: TextQueries,
        k=10,
        filter_fn=None,
        full_length_search=False,
        qid_to_pids=None,
    ):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(
            queries, Q, k, filter_fn=filter_fn, qid_to_pids=qid_to_pids
        )

    def _search_all_Q(self, queries, Q, k, filter_fn=None, qid_to_pids=None):
        qids = list(queries.keys())

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        all_scored_pids = [
            list(
                zip(
                    *self.dense_search(
                        Q[query_idx : query_idx + 1],
                        k,
                        filter_fn=filter_fn,
                        pids=qid_to_pids[qid],
                    )
                )
            )
            for query_idx, qid in tqdm(enumerate(qids))
        ]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = "Searcher::search_all"
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

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
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids)

        return pids[:k], list(range(1, k + 1)), scores[:k]