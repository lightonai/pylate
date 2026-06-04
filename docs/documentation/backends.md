# Scoring Backends

PyLate's late-interaction scoring (`colbert_scores`, `colbert_scores_pairwise`,
`colbert_kd_scores`) ships with a pluggable **backend selector**. The default
PyTorch path always works; optional fused-GPU kernels can be enabled with one
extra `pip install` and a single kwarg.

## Available backends

| `backend=` | Backend | When it fires |
|---|---|---|
| `"auto"` (default) | best available | Tries `flash` on CUDA; falls back to `torch` |
| `"torch"` | the original `einsum` + `max` + `sum` reduction | Always available; CPU + CUDA |
| `"flash"` | [`flash-maxsim`](https://github.com/roipony/flash-maxsim) fused Triton kernels | CUDA-only; requires `pip install pylate[flash-maxsim]` |

The selector is non-breaking: omitting `backend=` keeps the legacy `torch`
behaviour unless the optional dependency is installed *and* the inputs are
CUDA tensors of supported shape.

## Installation

```bash
# Default PyTorch backend — no extras needed
pip install pylate

# Add the flash-maxsim backend
pip install pylate[flash-maxsim]
```

## Usage

### Per-call

```python
from pylate.scores import colbert_scores

scores = colbert_scores(queries, documents, backend="flash")
```

### Global override

Set `PYLATE_SCORES_BACKEND` once; every subsequent score call reads it:

```bash
export PYLATE_SCORES_BACKEND=flash    # or "torch", "auto"
```

The env var is read at call time, so flipping it at runtime (e.g.\
`os.environ["PYLATE_SCORES_BACKEND"] = "torch"`) takes effect immediately
without re-importing.

### Trainer integration

The trainer's loss path uses the same dispatch. Set `PYLATE_SCORES_BACKEND=flash`
before launching training to use the fused kernels throughout — this is the
recommended setup for contrastive training where the in-batch-negatives score
tensor dominates memory.

## What `auto` falls back on

`backend="auto"` (the default) dispatches to `flash` only when **all** of the
following hold:

- The inputs are CUDA tensors
- `flash-maxsim` is importable
- The shape is one the kernel supports (`_inputs_supported` — checks dtype,
  contiguity, length bounds)

If any precondition fails, `auto` silently falls back to `torch`. If the user
explicitly requested `backend="flash"`, the same preconditions are checked
but a precondition failure raises (rather than silently dropping back) so the
caller knows.

The only exception we silently catch is `FlashUnsupported` (raised by
`_inputs_supported`); kernel assertions, OOMs, and runtime bugs propagate as
real errors.

## What you actually get

For the GPU-fitting shapes (any real ColBERT or ColPali workload):

- **Forward** is ~2-4× faster than `einsum + max + sum` on cross-product
  in-batch scoring; OOM-unlocks corpus sizes the materialised `[B,Lq,Ld]`
  similarity tensor can't fit (the headline contribution of `flash-maxsim`).
- **Training** (contrastive in-batch) is dominated by memory: the fused
  backward removes the `[B,B,Lq,Ld]` gradient tensor that caps the batch
  size, giving 6-22× memory reduction at ColBERT/ColPali shapes.
- **Variable-length** (ragged document corpora) gets up to 4.6× speedup
  via the padding-free `cu_seqlens` variant.

See [`flash-maxsim` benchmarks](https://github.com/roipony/flash-maxsim) for
per-shape numbers.

## Other late-interaction kernels

PyLate is designed to accommodate multiple fused-MaxSim implementations.
[`late-interaction-kernels` (LIK)](https://github.com/hcompai/late-interaction-kernels)
is a parallel project covered by PR
[#222](https://github.com/lightonai/pylate/pull/222); when both are installed
the selector can be extended (`backend="lik"`) so users pick at runtime.
