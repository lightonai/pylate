from enum import Enum
from typing import Dict, Iterable

import torch.nn.functional as F
from torch import Tensor, nn
import torch 

from sentence_transformers.SentenceTransformer import SentenceTransformer


class ColBERTSimilarityMetric(Enum):
    """The metric for the contrastive loss"""

    def COLBERT_SIMILARITY(x, y, mask):
        # a num_queries, s queries_seqlen, h hidden_size, b num_documents, t documents_seqlen
        # Take make along the t axis (get max similarity for each query tokens), then sum over all the query tokens
        simis = torch.einsum("ash,bth->abst", x, y)
        expanded_mask = mask.unsqueeze(0).unsqueeze(2)
        expanded_mask = expanded_mask.expand(simis.size(0), -1, simis.size(2), -1)
        simis[expanded_mask == 0] = float("-inf")
        return simis.max(axis=3).values.sum(axis=2)
   
        return torch.einsum("ash,bth->abst", x, y).max(axis=3).values.sum(axis=2)



class ColBERTLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=ColBERTSimilarityMetric.COLBERT_SIMILARITY,
        size_average: bool = True,
    ):
        """
        Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
        two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

        Args:
            model: SentenceTransformer model
            distance_metric: Function that returns a distance between
                two embeddings. The class ColBERTDistanceMetric contains
                pre-defined metrices that can be used
            size_average: Average by the size of the mini-batch.

        References:
            * Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            * `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_

        Requirements:
            1. (anchor, positive/negative) pairs

        Relations:
            - :class:`OnlineContrastiveLoss` is similar, but uses hard positive and hard negative pairs.
            It often yields better results.

        Inputs:
            +-----------------------------------------------+------------------------------+
            | Texts                                         | Labels                       |
            +===============================================+==============================+
            | (anchor, positive/negative) pairs             | 1 if positive, 0 if negative |
            +-----------------------------------------------+------------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                    "label": [1, 0],
                })
                loss = losses.ContrastiveLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super(ColBERTLoss, self).__init__()
        self.distance_metric = distance_metric
        self.model = model
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(ColBERTSimilarityMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "ColBERTSimilarityMetric.{}".format(name)
                break

        return {"distance_metric": distance_metric_name, "size_average": self.size_average}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["token_embeddings"] for sentence_feature in sentence_features]
        masks = [sentence_feature["attention_mask"] for sentence_feature in sentence_features]
        # rep_anchor, rep_pos, rep_neg = reps
        # distances = self.distance_metric(rep_anchor, (torch.cat((rep_pos, rep_neg))))
        # Compute the distances between the anchor (0) and the positives (1) as well as the negatives (2)
        distances = torch.cat([self.distance_metric(reps[0], rep, mask) for rep, mask in zip(reps[1:], masks[1:])], dim=1)
        # create corresponding labels
        # labels = torch.arange(0, rep_anchor.size(0), device=rep_anchor.device)
        labels = torch.arange(0, reps[0].size(0), device=reps[0].device)
        # compute contrastive loss using cross-entropy over the distances
        loss = F.cross_entropy(distances, labels, reduction="mean" if self.size_average else "sum")

        return loss

    @property
    def citation(self) -> str:
        return """
    @inproceedings{santhanam-etal-2022-colbertv2,
        title = "{C}ol{BERT}v2: Effective and Efficient Retrieval via Lightweight Late Interaction",
        author = "Santhanam, Keshav  and
        Khattab, Omar  and
        Saad-Falcon, Jon  and
        Potts, Christopher  and
        Zaharia, Matei",
        editor = "Carpuat, Marine  and
        de Marneffe, Marie-Catherine  and
        Meza Ruiz, Ivan Vladimir",
        booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
        month = jul,
        year = "2022",
        address = "Seattle, United States",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.naacl-main.272",
        doi = "10.18653/v1/2022.naacl-main.272",
        pages = "3715--3734",
        abstract = "Neural information retrieval (IR) has greatly advanced search and other knowledge-intensive language tasks. While many neural IR methods encode queries and documents into single-vector representations, late interaction models produce multi-vector representations at the granularity of each token and decompose relevance modeling into scalable token-level computations. This decomposition has been shown to make late interaction more effective, but it inflates the space footprint of these models by an order of magnitude. In this work, we introduce ColBERTv2, a retriever that couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction. We evaluate ColBERTv2 across a wide range of benchmarks, establishing state-of-the-art quality within and outside the training domain while reducing the space footprint of late interaction models by 6{--}10x.",
    }
"""
