from __future__ import annotations

import torch

from pylate import utils


def test_colbert_collator_keeps_tensor_fields_before_metadata() -> None:
    def tokenize(texts, **kwargs):
        return {
            "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
        }

    collator = utils.ColBERTCollator(tokenize_fn=tokenize)
    batch = collator(
        [
            {
                "query": "query",
                "positive": "document",
                "dataset_name": "dataset",
            }
        ]
    )

    assert isinstance(next(iter(batch.values())), torch.Tensor)
    assert batch["return_loss"] is True
    assert batch["dataset_name"] == "dataset"
