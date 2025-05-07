from pylate.indexes.stanford_nlp.data.collection import Collection
from pylate.indexes.stanford_nlp.data.examples import Examples
from pylate.indexes.stanford_nlp.data.queries import Queries
from pylate.indexes.stanford_nlp.infra.config.config import ColBERTConfig
from pylate.indexes.stanford_nlp.modeling.reranker.tokenizer import RerankerTokenizer
from pylate.indexes.stanford_nlp.utils.utils import flatten, zipstar

# from pylate.indexes.stanford_nlp.utils.runs import Run


class RerankBatcher:
    def __init__(
        self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1
    ):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        assert (
            self.accumsteps == 1
        ), "The tensorizer doesn't support larger accumsteps yet --- but it's easy to add."

        self.tokenizer = RerankerTokenizer(
            total_maxlen=config.doc_maxlen, base=config.checkpoint
        )
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

        assert len(self.triples) > 0, "Received no triples on which to train."
        assert len(self.queries) > 0, "Received no queries on which to train."
        assert len(self.collection) > 0, "Received no collection on which to train."

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = (
            self.position,
            min(self.position + self.bsize, len(self.triples)),
        )
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            query, *pids = self.triples[position]
            pids = pids[: self.nway]

            query = self.queries[query]

            try:
                pids, scores = zipstar(pids)
            except:
                scores = []

            passages = [self.collection[pid] for pid in pids]

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)

        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        return self.collate(all_queries, all_passages, all_scores)

    def collate(self, queries, passages, scores):
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize

        queries = flatten([[query] * self.nway for query in queries])
        return [(self.tokenizer.tensorize(queries, passages), scores)]

    # def skip_to_batch(self, batch_idx, intended_batch_size):
    #     Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
    #     self.position = intended_batch_size * batch_idx
