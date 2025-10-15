# Overview

## evaluation


**Classes**

- [ColBERTDistillationEvaluator](../evaluation/ColBERTDistillationEvaluator)
- [ColBERTTripletEvaluator](../evaluation/ColBERTTripletEvaluator)
- [NanoBEIREvaluator](../evaluation/NanoBEIREvaluator)
- [PyLateInformationRetrievalEvaluator](../evaluation/PyLateInformationRetrievalEvaluator)

**Functions**

- [evaluate](../evaluation/evaluate)
- [get_beir_triples](../evaluation/get-beir-triples)
- [load_beir](../evaluation/load-beir)
- [load_custom_dataset](../evaluation/load-custom-dataset)

## hf_hub

- [PylateModelCardData](../hf-hub/PylateModelCardData)

## indexes

- [PLAID](../indexes/PLAID)
- [Voyager](../indexes/Voyager)

## losses

- [CachedContrastive](../losses/CachedContrastive)
- [Contrastive](../losses/Contrastive)
- [Distillation](../losses/Distillation)

## models

- [ColBERT](../models/ColBERT)
- [Dense](../models/Dense)

## rank


**Classes**

- [RerankResult](../rank/RerankResult)

**Functions**

- [rerank](../rank/rerank)

## retrieve

- [ColBERT](../retrieve/ColBERT)

## scores


**Classes**

- [SimilarityFunction](../scores/SimilarityFunction)

**Functions**

- [colbert_kd_scores](../scores/colbert-kd-scores)
- [colbert_scores](../scores/colbert-scores)
- [colbert_scores_pairwise](../scores/colbert-scores-pairwise)

## utils


**Classes**

- [ColBERTCollator](../utils/ColBERTCollator)
- [KDProcessing](../utils/KDProcessing)

**Functions**

- [all_gather](../utils/all-gather)
- [all_gather_with_gradients](../utils/all-gather-with-gradients)
- [convert_to_tensor](../utils/convert-to-tensor)
- [get_rank](../utils/get-rank)
- [get_world_size](../utils/get-world-size)
- [iter_batch](../utils/iter-batch)
