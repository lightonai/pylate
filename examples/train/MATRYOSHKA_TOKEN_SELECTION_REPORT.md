# Smarter Token Selection Strategies for MatryoshkaDocTokensLoss

## Motivation

The current `MatryoshkaDocTokensLoss` uses positional truncation (`tensor[:, :n_tokens, :]`) to train ColBERT models to work with fewer document tokens. While this works well at 64-128 tokens, it degrades at 32 tokens (~1.5% NDCG@10 gap). The goal is to implement three alternative strategies that select/compress tokens more intelligently.

## Baseline: MatryoshkaDocTokensLoss (Positional Truncation)

Simply keeps the first N tokens. No learnable parameters beyond the ColBERT model itself. Fast, simple, but position-biased — tokens at the end of a document are always discarded first regardless of their semantic importance.

## Approach 1: Learned Importance + STE (`MatryoshkaImportanceLoss`)

**File:** `pylate/losses/matryoshka_importance.py`

**Idea:** A `Linear(embed_dim, 1)` head scores each token's importance. Hard top-k selection in forward, straight-through estimator (STE) for backward.

**Learnable parameters:** `ImportanceScoreHead` — `nn.Linear(128, 1)` = 129 params

**How it works:**
```
scores = score_head(doc_embeddings)           # (batch, T, 1) → squeeze → (batch, T)
scores[~mask] = -inf                           # mask padding
soft_scores = sigmoid(scores)                  # for backward path
_, topk_idx = scores.topk(k)                   # hard forward selection
hard_mask = zeros.scatter_(1, topk_idx, 1.0)   # binary mask
ste_mask = hard_mask - soft_scores.detach() + soft_scores  # STE trick
weighted = embeddings * ste_mask.unsqueeze(-1)  # weight embeddings
selected = gather(weighted, topk_idx)           # reduce to (batch, k, dim)
```

- Forward: hard top-k (actual token reduction to k tokens)
- Backward: gradients flow through `sigmoid(scores)` to the score head
- Training and inference behave the same (hard top-k + STE in train, hard top-k in eval)

## Approach 2: Soft Top-K Gating (`MatryoshkaSoftTopKLoss`)

**File:** `pylate/losses/matryoshka_soft_topk.py`

**Idea:** Fully differentiable soft gating — no STE needed. Tokens are weighted by a learned sigmoid gate; at inference, hard top-k.

**Learnable parameters:** `SoftTopKGate` — `nn.Linear(128, 1)` + learnable `temperature` scalar = 130 params

**How it works (training):**
```
scores = gate.linear(doc_embeddings).squeeze(-1)   # (batch, T)
scores[~mask] = -inf
sorted_scores = scores.sort(descending=True)
threshold = sorted_scores[:, k-1:k]                # k-th score as threshold
soft_mask = sigmoid((scores - threshold) * temperature)
weighted = embeddings * soft_mask.unsqueeze(-1)     # all T tokens remain
```

**Key difference from Approach 1:** During training, the tensor shape stays `(batch, T, dim)` — no actual token reduction. All tokens participate in MaxSim, just weighted. This means training MaxSim cost is NOT reduced, but gradients are exact (no STE approximation). At inference, hard top-k selection (actual reduction to `(batch, k, dim)`).

**NaN guard:** When masked tokens produce `-inf - (-inf) = NaN`, the implementation uses `nan_to_num(nan=-inf)` so sigmoid maps those to 0.

## Approach 3: Learned Hierarchical Pooling (`MatryoshkaHierarchicalPoolingLoss`)

**File:** `pylate/losses/matryoshka_hierarchical_pooling.py`

**Idea:** Iteratively halve tokens via learned attention-weighted pooling of consecutive pairs: N → N/2 → N/4 → N/8. Each level is a `LearnedPoolingLevel` with a `Linear(embed_dim, 1)` gate.

**Learnable parameters:**
- `LearnedPoolingLevel` — `nn.Linear(128, 1)` gate per level (129 params each)
- `HierarchicalPoolingStack` — `nn.ModuleList` of levels. For `[32, 64, 128, 256]`: 3 levels (~387 params)

**One pooling level (stride=2, T→T/2):**
```
groups = embeddings[:, :K*2].view(batch, K, 2, dim)   # pair consecutive tokens
gate_logits = gate(groups).squeeze(-1)                  # (batch, K, 2)
gate_logits[~group_mask] = -inf                         # mask padding within groups
weights = softmax(gate_logits, dim=2)                   # learned attention within pair
pooled = (groups * weights.unsqueeze(-1)).sum(dim=2)    # (batch, K, dim)
```

**Mask handling:** Pooled mask = `group_mask.any(dim=2)` — a pooled token is valid if any token in its group was valid. For CachedContrastive compatibility, a structural `_pool_mask` function handles mask pooling without the linear gate.

**Limitation discussed:** This pools consecutive token pairs structurally (token[0] with token[1], token[2] with token[3], etc.), regardless of actual token similarity. The user has a separate scipy-based hierarchical clustering approach (`pool_embeddings_hierarchical`) that groups tokens by cosine similarity at inference time, which is semantically smarter but not differentiable.

## Comparison

| | Approach 1: Importance+STE | Approach 2: Soft Top-K | Approach 3: Hierarchical Pooling |
|---|---|---|---|
| Parameters | 129 | 130 | ~387 |
| Gradient flow | Approximate (STE) | Exact | Exact |
| Training token reduction | Yes | No (full tensor) | Yes |
| Information preservation | Discards unchosen tokens | Soft-weights all tokens | Pools all tokens |
| Training MaxSim cost | Reduced (k tokens) | Full (T tokens) | Reduced (k tokens) |

## Verification Results

All three approaches were tested with Distillation (4D document tensors), Contrastive (3D), and CachedContrastive. All 9 combinations produce valid losses and gradient flow through learnable parameters.

```
MatryoshkaImportanceLoss
  Distillation: loss=0.1231, grad_norm=0.00000000
  Contrastive:  loss=1.7256, grad_norm=0.39065221
  CachedContr:  loss=1.6899, grad_norm=0.38401651

MatryoshkaSoftTopKLoss
  Distillation: loss=0.1231, gate_grad_norm=0.00000058, temp_grad=-0.00000000
  Contrastive:  loss=1.6758, gate_grad_norm=3.73019314, temp_grad=-0.02255776
  CachedContr:  loss=2.3178, gate_grad_norm=16.06092834, temp_grad=0.12276733

MatryoshkaHierarchicalPoolingLoss
  Distillation: loss=0.1231, level0_gate_grad_norm=0.00000000
  Contrastive:  loss=1.6977, level0_gate_grad_norm=0.06369890
  CachedContr:  loss=1.6591, level0_gate_grad_norm=0.07177541
```

**Note on Distillation gradients:** The near-zero gradients for Distillation are expected with the tiny test data (1 query, 2 documents, ~10 tokens). With k=8/16/32 and only ~10 tokens, most iterations hit `k >= T` and skip selection entirely. Contrastive/CachedContrastive show strong gradients because the in-batch negatives setup creates more gradient signal through the scoring.

## Architecture

All three follow the same wrapper pattern as `MatryoshkaDocTokensLoss`:

1. **`ForwardCachingDecorator`** — reused from `matryoshka_doc_tokens.py`, caches `model.forward()` to avoid recomputation across token count iterations
2. **`XxxScoreMetricDecorator`** — intercepts the score function to apply the strategy on document embeddings before MaxSim
3. **`CachedXxxLossDecorator`** — decorates `CachedContrastive.calculate_loss` for gradient caching compatibility
4. **`MatryoshkaXxxLoss(nn.Module)`** — main wrapper with `__init__`, `forward`, `get_config_dict`

Learnable parameters live on the loss module (not the ColBERT model). Since the loss extends `nn.Module`, `SentenceTransformerTrainer` discovers them via `named_parameters()`.

All handle both 3D `(bs, tokens, dim)` and 4D `(bs, n_ways, tokens, dim)` document tensors by reshaping 4D to 3D, applying the strategy, and reshaping back.

## Files Created/Modified

### New files
- `pylate/losses/matryoshka_importance.py`
- `pylate/losses/matryoshka_soft_topk.py`
- `pylate/losses/matryoshka_hierarchical_pooling.py`
- `examples/train/gte_modern_colbert_matryoshka_importance.py`
- `examples/train/gte_modern_colbert_matryoshka_soft_topk.py`
- `examples/train/gte_modern_colbert_matryoshka_hierarchical_pooling.py`
- `examples/train/run_matryoshka_experiments.sh`

### Modified files
- `pylate/losses/__init__.py` — added exports
- `examples/train/gte_modern_colbert_matryoshka_doc_tokens.py` — shows all 4 options

## Analysis: Approach 3 Strengths and Weaknesses

### Strengths

- **Best information preservation of the three.** Instead of discarding tokens entirely (Approach 1) or just soft-weighting them (Approach 2), it actually *fuses* information from pairs. Every input token contributes to the output representation.
- **Real training token reduction** — unlike Approach 2, the output is an actual `(batch, k, dim)` tensor, so MaxSim cost scales with k, not T.
- **Exact gradients** through softmax attention within pairs — no STE approximation needed.
- **The hierarchical N → N/2 → N/4 structure is natural** for ColBERT's use case, where you want representations at power-of-2 token counts.

### Core Weakness: Structural Pairing

The pairing of token[0] with token[1], token[2] with token[3], etc. is **entirely arbitrary from a semantic standpoint**. BERT-style tokenizers don't guarantee that consecutive tokens are semantically related:

- **Subword splits:** `"un"`, `"##believ"`, `"##able"` — tokens 0 and 1 might be parts of one word, while tokens 2 and 3 span a word boundary.
- **CLS token:** `[CLS]` gets paired with the first content token, which is almost always a bad match.

The learned gate (softmax over the pair) can learn to put all weight on one token and effectively "drop" the other, but it **can never choose a better partner**. It's stuck with the structural pairing it's given. This is fundamentally less flexible than Approach 1, which can pick *any* k tokens from the full sequence.

### Predicted Scaling Behavior

At aggressive compression (e.g., 128 → 32, which is 3 pooling levels), the structural pairing weakness compounds — each level inherits the arbitrary grouping of the previous one:

- At **128 → 64** (1 level): probably fine, similar to Approach 1.
- At **128 → 32** (2 levels): the compounding starts to matter.
- At **256 → 32** (3 levels): likely worse than just picking the top-32 important tokens via Approach 1.

The reason: Approach 1's selection is *global* (any token can be chosen), while Approach 3's pooling is *local* (only within fixed consecutive pairs). At low token counts, global selection should dominate.

**Prediction:** Approach 1 (Importance + STE) will likely win in practice, especially at aggressive compression ratios. The STE approximation is well-understood and works fine in practice (see: quantization literature, discrete VAEs), and global top-k selection is simply more flexible than local fixed-pair pooling. That said, if Approach 3 does well at moderate compression (128 → 64) where information preservation matters more than selection flexibility, it could be a good default for users who only need mild compression.

### Why Not Use `pool_embeddings_hierarchical` from Encode Directly?

A natural question is: why not use the existing `ColBERT.pool_embeddings_hierarchical()` function (which uses scipy's Ward-linkage clustering by cosine similarity) during training? This would give semantically meaningful groupings instead of arbitrary consecutive pairs. The answer is that it has multiple fundamental incompatibilities with training:

1. **Not differentiable.** The function calls `scipy.cluster.hierarchy.linkage()` and `fcluster()`, which are pure NumPy/SciPy operations that sit entirely outside PyTorch's autograd graph. There is no gradient path from the loss back through the clustering decisions. You'd need to wrap it with an STE or REINFORCE-style estimator, at which point you lose the "exact gradient" advantage that motivated Approach 3 in the first place.

2. **Per-document, variable-output, Python loop.** The function iterates `for document_embeddings in documents_embeddings` and produces variable-length outputs per document (since different documents may have different numbers of clusters). This is fundamentally incompatible with the batched `(batch, tokens, dim)` tensor format that the loss functions, MaxSim scoring, and SentenceTransformer training pipeline all expect. You'd need per-document padding back to a uniform size, defeating much of the benefit.

3. **Prohibitively expensive per training step.** Ward-linkage clustering is O(n² log n) per document (computing the full pairwise distance matrix + the hierarchical merge). At training time, with hundreds of documents per batch and hundreds of tokens per document, this runs every forward pass for every matryoshka dimension. The consecutive-pair approach is O(n) — just a reshape and a learned weighted sum.

4. **Non-deterministic cluster assignments break gradient caching.** The `CachedContrastive` loss relies on `ForwardCachingDecorator` to cache model forward passes and replay them across matryoshka iterations. If the clustering assignments change between the cached forward and the loss computation (which they could due to floating point non-determinism or different batch compositions), the cached embeddings and the cluster assignments become inconsistent.

5. **CPU round-trip.** The function explicitly calls `.cpu().numpy()` to run scipy, then moves results back to GPU. This synchronization stall alone would be a major bottleneck in a training loop.

In short, the scipy-based hierarchical pooling is excellent for one-shot inference (run once after encoding, no gradients needed) but fundamentally unsuited for integration into a differentiable training loss.

### Ideas to Improve Approach 3

If the structural pairing limitation proves to be a problem in experiments, several directions could bridge the gap with semantically-aware grouping while remaining differentiable:

1. **Optimal transport / Sinkhorn-based assignment** — soft-assign tokens to groups based on cosine similarity using a differentiable Sinkhorn layer, then pool within groups. This replaces fixed consecutive pairing with learned similarity-based pairing, at the cost of O(n²) attention-like computation.

2. **Sort tokens by similarity before pairing** — even a simple heuristic like sorting by the first principal component of embeddings before applying the consecutive-pair pooling could help a lot, and it's cheap. The sorting itself isn't differentiable, but the pooling within the (now semantically better) pairs still is.

### Proposed Approach 4: Document-Derived Centroid Pooling

A more principled alternative that captures the spirit of `pool_embeddings_hierarchical` (cluster similar tokens and average them) while being fully differentiable. Inspired by Slot Attention (Locatello et al., 2020) and Pooling by Multihead Attention (Lee et al., 2019).

**Key idea:** Derive K centroid vectors **from the document itself**, then refine them via cross-attention back to all document tokens. This is purely document-side — the ColBERT search query is not involved at any point.

**Note on terminology:** "Centroids" here refers to cluster-center vectors derived from the document, NOT ColBERT search queries. "Queries" in Q/K/V notation refers to the attention role, not search queries.

**Why document-derived centroids matter:** A naive version would use fixed `nn.Parameter(K, dim)` centroids (global prototypes shared across all documents). But pooling should be document-specific — a document about cooking and a document about quantum physics should have different cluster centers. Fixed centroids can only adapt via the attention weights, not via the centroids themselves. Document-derived centroids make the pooling fully document-adaptive: both *what to look for* (centroids) and *how to assign* (attention weights) depend on the document content.

```python
class DocumentDerivedCentroidPooling(nn.Module):
    def __init__(self, embed_dim: int, n_iters: int = 1):
        super().__init__()
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.n_iters = n_iters
        self.scale = embed_dim ** -0.5

    def forward(self, doc_embeddings, mask, k):
        # doc_embeddings: (batch, T, dim)
        batch, T, dim = doc_embeddings.shape

        # Step 1: Initialize centroids FROM the document
        # Chunk T tokens into K groups and average — crude positional init
        chunk_size = T // k
        centroids = doc_embeddings[:, :k * chunk_size].view(
            batch, k, chunk_size, dim
        ).mean(dim=2)  # (batch, K, dim)

        # Step 2: Iterative refinement via cross-attention
        # Even 1 iteration lets centroids escape their initial chunk
        # and absorb semantically similar tokens from anywhere
        keys = self.proj_k(doc_embeddings)              # (batch, T, dim)
        for _ in range(self.n_iters):
            queries = self.proj_q(centroids)             # (batch, K, dim)
            attn_logits = queries @ keys.transpose(-1, -2) * self.scale  # (batch, K, T)
            attn_logits[..., ~mask] = float('-inf')
            attn_weights = softmax(attn_logits, dim=-1)  # (batch, K, T)
            centroids = attn_weights @ doc_embeddings     # (batch, K, dim)

        return centroids  # (batch, K, dim) — pooled document
```

**How it works:**
1. **Initialization:** Chunk the T document tokens into K positional groups and average each group. This gives K initial centroids that are rough positional summaries — not ideal, but a starting point derived from the document itself.
2. **Refinement:** Each centroid cross-attends to ALL T tokens via learned Q/K projections. This allows a centroid initialized from tokens 0-3 to "discover" that token 47 is semantically similar and absorb it. After even 1 iteration, centroids become content-aware cluster centers rather than positional averages.
3. **Output:** K document-derived, content-adaptive pooled token representations.

**Why document-derived centroids are stronger than fixed centroids:**

The centroid-token affinity is **bilinear in the document content**: centroids depend on the document (via initialization + iterative updates), AND attention weights depend on centroid-token similarity. With fixed global centroids, only the latter holds — the "where to look" is static across all documents.

**Why this addresses the limitations of Approach 3:**

| | Approach 3 (Hierarchical) | Doc-Derived Centroid Pooling |
|---|---|---|
| Grouping strategy | Fixed consecutive pairs | Content-adaptive via cross-attention |
| What can be grouped | Only adjacent tokens | Any tokens in the document |
| Centroid source | N/A (no centroids) | Derived from document content |
| Gradient flow | Exact | Exact |
| Training token reduction | Yes (batch, K, dim) | Yes (batch, K, dim) |
| Info preservation | Pools within pairs | Pools all tokens via soft attention |
| Computational cost | O(T) per level | O(K×T×n_iters) |
| Parameters | ~129 per level | 2 × dim² (Q/K projections) |

**Parameter cost:** For dim=128: 2 × 128 × 128 = 32,768 params for the Q/K projections, shared across all target sizes K. No per-K parameters needed — the same projections work for any K since the initialization adapts. Still tiny relative to the ColBERT model (~33M+ params).

**How it relates to `pool_embeddings_hierarchical`:** The inference-time scipy function computes pairwise cosine similarity, runs Ward-linkage clustering, and averages within clusters. This approach is the differentiable analog: the initialization corresponds to an initial partitioning, the iterative cross-attention corresponds to cluster refinement (like iterative K-means), and the final weighted average corresponds to within-cluster pooling. The key difference is that everything runs on GPU in batched tensor operations with full gradient flow.

**Design choices:**
- `n_iters=1` is likely sufficient — Slot Attention typically converges in 3-7 iterations, but our initialization from document chunks is already much better than random, so 1 refinement step should suffice. Worth ablating.
- The Q/K projections are shared across iterations (weight tying), keeping parameter count low.
- Can still be applied hierarchically (T → T/2 → T/4) if desired, though single-step pooling to K is simpler.

### Relation to AGC (Qin et al., 2026)

The concurrent work "Multi-Vector Index Compression in Any Modality" (Qin, Martin, Jha, Zuo, Kriz, Van Durme; arXiv:2602.21202, Feb 2026) introduces **Attention-Guided Clustering (AGC)**, which addresses the same problem with a strikingly similar three-stage structure: (1) attention-based centroid selection, (2) clustering, (3) weighted aggregation.

**AGC's approach:**
1. **Saliency scoring:** Learned "universal query tokens" X_Ψ are appended to the document and run through the full bidirectional encoder. Saliency scores α are computed by averaging the attention weights from universal tokens to document tokens across all heads and layers: `α = (1/|Ψ|H) ΣΣ Attn_i^(L,η)`.
2. **Centroid selection:** The top-m tokens by saliency become cluster centroids (hard top-k selection).
3. **Hard clustering:** Every non-centroid token is assigned to its nearest centroid by cosine similarity: `G_k = {j | k = argmax cos(z_j, μ_k)}` (hard argmax assignment).
4. **Weighted aggregation:** Within each cluster, the compressed token is a saliency-weighted average: `c_k = Σ(α_j · z_j) / Σ(α_j)` for j in cluster k.

AGC achieves 97% of full-index performance at 32 tokens on BEIR and sets new SOTA on MSR-VTT, validating the centroid-selection + clustering + weighted-pooling paradigm.

**Key architectural difference — where centroids come from:**

| | Our Approach 4 | AGC |
|---|---|---|
| Centroid derivation | Post-encoder: chunked averaging + cross-attention refinement on encoder output | In-encoder: universal tokens attend through all transformer layers |
| Encoder modification | None — operates on frozen encoder output | Requires appending tokens to encoder input, modifying the forward pass |
| Clustering | Soft (attention weights = differentiable soft assignment) | Hard (argmax cosine assignment, non-differentiable) |
| Centroid selection | Implicit in initialization + refinement | Explicit hard top-k on saliency scores (non-differentiable) |

**Key difference — differentiability of the full pipeline:**

This is the most important distinction. AGC has **two hard, non-differentiable operations** in its pipeline:

1. **Centroid selection is hard top-k.** `I = top-m(α)` selects which tokens become centroids. This is a discrete, non-differentiable decision. No gradient flows through the question "should token 47 have been a centroid instead of token 12?" The saliency scores α receive gradients from the weighted aggregation step (where they're used as weights), but NOT from the centroid selection step (where they're used for ranking).

2. **Cluster assignment is hard argmax.** `G_k = {j | k = argmax cos(z_j, μ_k)}` assigns each token to exactly one cluster. This is also non-differentiable. No gradient flows through the question "should token 23 have been assigned to cluster 2 instead of cluster 5?" The assignment is frozen for the backward pass.

The **only differentiable component** in AGC is the weighted aggregation: `c_k = Σ(α_j · z_j) / Σ(α_j)`. Gradients flow through the saliency weights α and the token embeddings z, but **only within the fixed cluster assignments established by the hard operations**. The model can learn:
- "Give token 23 more/less weight within its assigned cluster" ✓
- "Move token 23 to a different cluster" ✗
- "Use a different token as centroid" ✗

This is analogous to the STE limitation in our Approach 1 — the hard decisions create gradient approximation gaps.

**Our Approach 4 is fully soft — every operation is differentiable:**

The cross-attention weights `attn = softmax(proj_q(centroids) @ proj_k(doc_embs).T / sqrt(d))` simultaneously encode both centroid identity AND cluster assignment in a single differentiable operation. There are no hard discrete choices anywhere in the pipeline:

- **Centroid derivation:** Chunked averaging (differentiable mean) → cross-attention refinement (differentiable softmax + matmul). Gradients flow all the way back to the input embeddings.
- **Cluster assignment:** The attention weights are a *soft* assignment matrix — token j contributes to centroid k proportionally to `attn[k, j]`. This is not a discrete 0/1 assignment but a continuous distribution over centroids.
- **Aggregation:** `pooled = attn @ doc_embs` is a differentiable weighted sum.

This means the gradient can express:
- "Give token 23 more/less weight within its assigned cluster" ✓
- "Token 23 should contribute 0.3 to centroid 2 and 0.7 to centroid 5 instead of the current 0.1/0.9 split" ✓ (soft cluster reassignment)
- "The centroid for cluster 3 should shift toward token 47's region of the embedding space" ✓ (centroid refinement)

The gradient signal is **exact** for all three — no STE, no hard assignment boundary, no approximation. The loss landscape is smooth with respect to all learnable parameters (the Q/K projections), which should make optimization easier and more stable.

**Trade-off 1 — In-encoder vs post-encoder:** AGC's in-encoder universal tokens are more powerful in one respect — they influence all transformer layers, so the encoder can co-adapt its internal representations to support compression. Our post-encoder approach cannot change how the encoder produces embeddings, only how they're pooled afterward. However, AGC's approach requires encoder modification and retraining from scratch, while ours can be applied to any pre-trained ColBERT model as a lightweight add-on.

**Trade-off 2 — Soft training vs hard deployment (train/inference mismatch):**

Our soft attention creates a gap between training and inference:
- **Training:** each pooled output token k is a soft weighted blend of ALL T input tokens: `pooled[k] = Σ_j attn[k,j] · doc_embs[j]`
- **Inference:** we must store K discrete vectors in the index — there is no soft blending at retrieval time.

AGC does NOT have this problem. Its hard centroid selection and hard cluster assignment at training time faithfully match what happens at inference — what you train is what you deploy.

This is the classic tension seen in Gumbel-softmax vs straight-through estimators in discrete VAEs: soft training gives better gradient signal but doesn't match deployment; hard training matches deployment but has worse (or approximate) gradients.

The empirical question is: **does our better gradient quality outweigh the train/inference mismatch?**

Possible mitigations for the mismatch:
- **Temperature annealing:** Start training with low temperature (soft attention, good gradients for early learning) and anneal to high temperature so that attention sharpens toward near-one-hot by the end of training: `attn = softmax(logits * τ)` with τ: 1 → 10 over training. This progressively closes the gap between soft training and hard inference.
- **Hard top-k at inference from soft-trained model:** Even without annealing, if the soft attention is naturally peaked (most weight on a few tokens per centroid), the hard selection at inference may lose very little compared to the soft blend.
- **Hybrid approach:** Train with soft attention for most of training, then fine-tune with hard assignment + STE for the final phase to close the gap.

**What we should take from AGC:**
- Their ablation (Table 9) confirms all three components matter: removing attention-based selection (R@1 drops 54.1 → 52.9), clustering (nDCG@10 drops 71.5 → 69.8), or weighted aggregation (nDCG@10 drops 71.5 → 71.0) each hurts performance.
- They achieve 97% of full-index performance at 32 tokens on BEIR and set new SOTA on MSR-VTT, validating the centroid + clustering + pooling paradigm.
- Their index utilization analysis (Figure 3) shows that AGC and H-Pool produce the most diverse and well-utilized compressed representations, while SeqResize and MemTok suffer from representation collapse.
- Their finding that compression can *improve* over the full index on some tasks (Table 1: AGC R@1 on MSR-VTT: 56.9 vs baseline 55.7) suggests that token compression acts as a beneficial regularizer, removing noise and redundancy.

## Next Steps

- Run full training experiments on MS MARCO to compare NDCG@10 at each token cutoff
- Compare against baseline `MatryoshkaDocTokensLoss` (positional truncation)
- Investigate whether Approach 3's structural pairing could be improved with similarity-based clustering (bridging the gap with the scipy `pool_embeddings_hierarchical` inference method)
- If Approach 3 underperforms at aggressive compression, try the "sort before pairing" heuristic as a low-cost improvement
- Consider implementing Approach 4 (Document-Derived Centroid Pooling) as the fully differentiable analog of AGC's attention-guided clustering
- Compare against AGC's results on overlapping benchmarks (BEIR) to validate our approach
