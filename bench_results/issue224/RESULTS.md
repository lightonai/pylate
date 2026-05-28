# Issue #224 benchmark — pylate dispatcher vs. einsum baseline

Reproduces the timing layout of [issue #224](https://github.com/lightonai/pylate/issues/224) but
swept against the pylate dispatcher routing through
[`late-interaction-kernels`](https://github.com/hcompai/late-interaction-kernels) (LIK) `0.3.0`.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 (capability 9.0) |
| CUDA | 12.4 (runtime image), torch 2.11.0+cu130 |
| dtype | fp16 |
| iters / warmup | 10 / 3 |
| seed | 0 |
| LIK | 0.3.0 (PyPI) |
| Harness | [`scripts/benchmark.py`](../../scripts/benchmark.py), launched via [`scripts/sky_benchmark.yaml`](../../scripts/sky_benchmark.yaml) |

`baseline` = `torch.einsum("ash,bth->abst").max(-1).sum(-1)` reference path.
`lik` = pylate dispatcher routing through `late_interaction_kernels.autograd.maxsim`
(the same entrypoint handles both in-batch and KD layouts via shape dispatch).

## §1 — In-batch forward (`Nq × Lq × d`  vs  `Nd × Ld × d`)

| label | shape (Nq, Lq, Nd, Ld) | baseline ms | LIK ms | speedup | peak GiB (base → LIK) |
| --- | --- | ---: | ---: | ---: | --- |
| contrastive_B32   | (32, 32, 32, 180)    |   0.173 |  0.157 | 1.10× | 0.06 → 0.03 |
| contrastive_B128  | (128, 32, 128, 180)  |   1.100 |  0.312 | 3.53× | 0.39 → 0.04 |
| contrastive_B256  | (256, 32, 256, 180)  |   4.194 |  0.844 | 4.97× | 1.45 → 0.04 |
| contrastive_B512  | (512, 32, 512, 180)  |  17.645 |  2.968 | 5.95× | 5.68 → 0.06 |
| longdoc_B64       | (64, 128, 64, 512)   |   2.484 |  0.305 | 8.14× | 1.04 → 0.04 |
| longdoc_B128      | (128, 128, 128, 512) |   9.830 |  0.805 | 12.21× | 4.05 → 0.05 |
| colpali_B16       | (16, 1024, 16, 1024) |   2.414 |  0.307 | 7.86× | 1.04 → 0.04 |
| colpali_B32       | (32, 1024, 32, 1024) |   9.299 |  0.820 | 11.34× | 4.05 → 0.05 |

## §2 — KD forward (`Nq × K × Ld × d`, per-query candidate lists)

| label | shape (Nq, K, Lq, Ld) | baseline ms | LIK ms | speedup |
| --- | --- | ---: | ---: | ---: |
| kd_B16_K8_short  | (16, 8, 32, 180)    | 0.160 | 0.139 | 1.15× |
| kd_B32_K16_short | (32, 16, 32, 180)   | 0.164 | 0.169 | 0.97× |
| kd_B32_K32_short | (32, 32, 32, 180)   | 0.176 | 0.177 | 0.99× |
| kd_B64_K32_short | (64, 32, 32, 180)   | 0.252 | 0.174 | 1.45× |
| kd_B16_K8_long   | (16, 8, 128, 512)   | 0.179 | 0.148 | 1.21× |
| kd_B32_K16_long  | (32, 16, 128, 512)  | 0.427 | 0.175 | 2.44× |

The §2 regression flagged in the issue (Python for-loop over the K axis) is gone — KD now
routes through the same `autograd.maxsim` entrypoint as in-batch via the kd_layout path.

## §3 — In-batch training (fwd + bwd)

| label | shape (Nq, Lq, Nd, Ld) | baseline ms | LIK ms | speedup | peak GiB (base → LIK) |
| --- | --- | ---: | ---: | ---: | --- |
| colbert_B128       | (128, 32, 128, 180)  |   2.492 |  0.767 |  3.25× | 0.42 → 0.09 |
| colbert_B256       | (256, 32, 256, 180)  |   9.402 |  1.868 |  5.03× | 1.48 → 0.16 |
| colbert_B512       | (512, 32, 512, 180)  |  38.417 |  7.228 |  5.32× | 5.71 → 0.41 |
| colbert_B1K        | (1024, 32, 1024, 180)| 153.435 | 29.310 |  5.23× | 22.62 → 1.39 |
| colpali_train_B8   | (8, 1024, 8, 1024)   |   1.599 |  0.477 |  3.35× | 0.32 → 0.08 |
| colpali_train_B16  | (16, 1024, 16, 1024) |   5.944 |  0.719 |  8.27× | 1.07 → 0.09 |
| colpali_train_B32  | (32, 1024, 32, 1024) |  23.683 |  2.153 | 11.00× | 4.08 → 0.13 |

## §4 — KD training (fwd + bwd)

| label | shape (Nq, K, Lq, Ld) | baseline ms | LIK ms | speedup |
| --- | --- | ---: | ---: | ---: |
| kdtrain_B32_K16_short | (32, 16, 32, 180)  | 0.570 | 0.495 | 1.15× |
| kdtrain_B32_K32_short | (32, 32, 32, 180)  | 0.617 | 0.485 | 1.27× |
| kdtrain_B64_K32_short | (64, 32, 32, 180)  | 0.809 | 0.548 | 1.48× |
| kdtrain_B16_K8_long   | (16, 8, 128, 512)  | 0.542 | 0.471 | 1.15× |

## §5 — Pairwise forward (baseline only)

pylate's `colbert_scores_pairwise` uses a Python loop with `einsum("sh,th->st")` and
does not route through the dispatcher, so the LIK column is intentionally omitted.

| label | shape (batch, Lq, Ld) | baseline ms |
| --- | --- | ---: |
| pairwise_B256_text    | (256, 32, 180)   |  22.025 |
| pairwise_B1000_text   | (1000, 32, 180)  |  86.278 |
| pairwise_B5000_text   | (5000, 32, 180)  | 440.777 |
| pairwise_B500_colpali | (500, 1024, 1024)|  46.358 |

## §6 — Precision (vs fp64 reference, in-batch forward)

The kernel accumulates in fp32 internally, the baseline accumulates in fp16.

| label | shape | kernel | max abs Δ | mean abs Δ | max rel Δ | mean rel Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| short   | (8, 32, 32, 180)    | baseline | 4.71e-01 | 1.28e-01 | 4.53e-04 | 1.31e-04 |
| short   | (8, 32, 32, 180)    | lik      | **2.88e-04** | **1.58e-04** | **2.78e-07** | **1.62e-07** |
| longdoc | (8, 128, 32, 512)   | baseline | 2.16e+00 | 1.00e+00 | 4.89e-04 | 2.28e-04 |
| longdoc | (8, 128, 32, 512)   | lik      | **1.21e-03** | **7.46e-04** | **2.80e-07** | **1.70e-07** |
| colpali | (4, 1024, 16, 1024) | baseline | 1.57e+01 | 7.76e+00 | 4.22e-04 | 2.07e-04 |
| colpali | (4, 1024, 16, 1024) | lik      | **1.15e-02** | **6.41e-03** | **3.06e-07** | **1.71e-07** |

LIK is 3–4 orders of magnitude closer to the fp64 reference than the fp16 baseline.

## Raw data

- [`fast.json`](./fast.json) — §1, §2, §5, §6 (38 cells)
- [`train.json`](./train.json) — §3, §4 (22 cells)
