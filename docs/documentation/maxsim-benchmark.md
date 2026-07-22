# Accelerated MaxSim benchmark

Empirical comparison of accelerated MaxSim backends against PyLate's
`torch.einsum + max.sum` baseline (`colbert_scores`, `colbert_kd_scores`,
`colbert_scores_pairwise`). Source write-up:
[#224](https://github.com/lightonai/pylate/issues/224).

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (sm_90) |
| PyTorch | 2.11.0 + CUDA 12.8 |
| Triton | 3.6.0 |
| dtype | fp16, dim = 128 (ColBERT default) |
| Baseline | `pylate.scores.colbert_scores` / `colbert_kd_scores` / `colbert_scores_pairwise` |

Packages compared:

- **flash** — [`flash-maxsim`](https://github.com/roipony/flash-maxsim) (PR [#212](https://github.com/lightonai/pylate/pull/212))
- **lik** — [`late-interaction-kernels`](https://github.com/hcompai/late-interaction-kernels) (PR [#222](https://github.com/lightonai/pylate/pull/222))
- **hf** — `erikkaum/maxsim` via Hugging Face `kernels` (forward-only)

Timings are median over 10 runs (3 warmup) with `torch.cuda.Event`. Numerical
agreement vs the fp16 PyLate baseline stayed within fp16 accumulation noise.

See also [Scoring backends](backends.md) for installation and `backend=` usage.

## In-batch inference (`colbert_scores`)

`(Nq, Lq, d) × (B, Ld, d) → (Nq, B)`.

| scenario | baseline | flash | lik | hf | flash× | lik× |
|---|---:|---:|---:|---:|---:|---:|
| contrastive B=128 Lq=32 Ld=180 | 0.88 ms | 0.18 ms | 0.31 ms | 48.66 ms | 4.8× | 2.8× |
| contrastive B=512 Lq=32 Ld=180 | 14.94 ms | 2.60 ms | 2.93 ms | 777.70 ms | 5.7× | 5.1× |
| long-doc B=128 Lq=128 Ld=512 | 7.78 ms | 0.60 ms | 0.79 ms | 551.39 ms | 12.9× | 9.8× |
| ColPali B=32 Lq=1024 Ld=1024 | 7.41 ms | 0.57 ms | 0.81 ms | 566.60 ms | 13.0× | 9.1× |

- **flash** and **lik** both beat the baseline; flash leads ~30–50% over lik.
- **hf** is far slower on H100 (no Hopper tuning; scalar PTX fallback).

## KD inference (`colbert_kd_scores`)

`(Nq, Lq, d) × (Nq, K, Ld, d) → (Nq, K)`.

| scenario | baseline | flash | lik | flash× | lik× |
|---|---:|---:|---:|---:|---:|
| KD B=32 K=16 Lq=32 Ld=180 | 0.08 ms | 0.11 ms | 6.29 ms | 0.7× | 0.01× |
| KD B=16 K=8 Lq=128 Ld=512 | 0.09 ms | 0.10 ms | 3.11 ms | 0.9× | 0.03× |
| KD B=32 K=16 Lq=128 Ld=512 | 0.30 ms | 0.12 ms | 6.15 ms | 2.6× | 0.05× |

- **lik** is 20–70× slower on KD when it falls back to a Python per-query loop
  (no native KD kernel).
- **flash** uses a single fused KD kernel (`shared_docs=False`) and scales better
  on longer documents.

## In-batch training (forward + backward)

| scenario | baseline | flash | lik | flash speed / mem | lik speed / mem |
|---|---:|---:|---:|---|---|
| ColBERT B=256 Lq=32 Ld=180 | 7.74 ms | 2.75 ms | 2.38 ms | 2.8× / mem×6.0 | 3.3× / mem×9.3 |
| ColBERT B=1K Lq=32 Ld=180 | 127.86 ms | 24.48 ms | 23.53 ms | 5.2× / mem×9.0 | 5.4× / mem×16.3 |
| ColPali B=16 Lq=1024 Ld=1024 | 5.25 ms | 1.56 ms | 1.48 ms | 3.4× / mem×10.3 | 3.6× / mem×11.2 |
| ColPali B=32 Lq=1024 Ld=1024 | 18.97 ms | 4.00 ms | 2.94 ms | 4.7× / mem×22.6 | 6.4× / mem×31.4 |

- erikkaum/maxsim has **no backward**, so it is omitted here.
- On large batches, **lik** slightly edges **flash** on speed/memory; both unlock
  batch sizes where the einsum path OOMs.
- On tiny batches, tile setup can make both kernels slower than einsum.

## Pairwise / packed rerank (`colbert_scores_pairwise`)

| scenario | baseline | flash | lik | hf | flash× |
|---|---:|---:|---:|---:|---:|
| B=256 text | 10.10 ms | 0.09 ms | 0.69 ms | 0.85 ms | 113× |
| B=1000 text | 39.57 ms | 0.11 ms | 13.38 ms | 3.11 ms | 360× |
| B=5000 text | 196.24 ms | 0.11 ms | 355.19 ms | 14.76 ms | 1784× |
| B=500 ColPali | 20.56 ms | 0.53 ms | 217.98 ms | 283.71 ms | 39× |

**flash** dominates packed/varlen workloads. **lik** varlen can regress past a
few hundred pairs on Hopper.

## Precision notes

At **fp16/bf16** inputs (typical training/serving dtypes), fused kernels that
accumulate in fp32 are usually **more accurate** than the einsum baseline that
accumulates in the input dtype. Rankings stayed aligned in sanity checks.

Caveats from [#224](https://github.com/lightonai/pylate/issues/224):

- flash-maxsim's **bf16 + masked** path showed a long-doc precision tail; prefer
  **fp16**, or **lik**, for masked bf16 long-document workloads until fixed upstream.
- Gradient direction cosine vs fp32 autograd stayed ≥ 0.997 for flash and lik;
  large per-element `max|Δ|` is mostly argmax-tie sparsity, not a bulk bug.

## Summary by use-case

| | In-batch inference | In-batch training | KD inference | KD training | Pairwise / rerank | Backward |
|---|---|---|---|---|---|---|
| flash (#212) | **5–13×** | 4–5× (mem 6–22×) | **1–3×** | ~1× (avoids OOM) | **100–2000×** | yes |
| lik (#222) | 4–10× | **5–6×** (mem 9–31×) | 0.01–0.05× (loop) | 0.05–0.2× (loop) | regressive past B≈500 | yes |
| hf (erikkaum) | 0.01–0.04× on H100 | n/a | mixed | n/a | OK small text | **no** |

## Practical recommendation

For PyLate's pluggable backends today:

1. Prefer **`backend="auto"`** (tries flash, then lik, then torch) or set
   `PYLATE_SCORES_BACKEND` explicitly — see [Scoring backends](backends.md).
2. **flash-maxsim** covers in-batch, KD, and packed-varlen with real fused kernels
   and is the strongest default fast path on CUDA, especially for pairwise/rerank
   and KD.
3. **lik** is competitive for large-batch in-batch training and adds an MPS path;
   avoid relying on it for KD or huge pairwise batches until upstream ships a
   native KD / Hopper-tuned varlen path.
4. Hugging Face `erikkaum/maxsim` is useful as a precision reference but is not a
   practical H100 training/inference backend (forward-only, no Hopper tuning).
