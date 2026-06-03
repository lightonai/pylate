# Issue #224 benchmark — pylate dispatcher vs. einsum baseline

Reproduces the timing layout of [issue #224](https://github.com/lightonai/pylate/issues/224) but
swept against the pylate dispatcher routing through
[`late-interaction-kernels`](https://github.com/hcompai/late-interaction-kernels) (LIK) `0.4.0`.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 (capability 9.0) |
| CUDA | 12.4 (runtime image), torch 2.11.0+cu130 |
| dtype | fp16 |
| iters / warmup | 10 / 3 |
| seed | 0 |
| LIK | 0.4.0 (PyPI) |
| Harness | [`scripts/benchmark.py`](../../scripts/benchmark.py), launched via [`scripts/sky_benchmark.yaml`](../../scripts/sky_benchmark.yaml) |

`baseline` = `torch.einsum("ash,bth->abst").max(-1).sum(-1)` reference path.
`lik` = pylate dispatcher routing through `late_interaction_kernels.autograd.maxsim`
(the same entrypoint handles both in-batch and KD layouts via shape dispatch).

## §1 — In-batch forward (`Nq × Lq × d`  vs  `Nd × Ld × d`)

| label | shape (Nq, Lq, Nd, Ld) | baseline ms | LIK ms | speedup | peak GiB (base → LIK) |
| --- | --- | ---: | ---: | ---: | --- |
| contrastive_B32   | (32, 32, 32, 180)    |   0.178 |  0.157 | 1.14× | 0.06 → 0.03 |
| contrastive_B128  | (128, 32, 128, 180)  |   1.107 |  0.319 | 3.47× | 0.39 → 0.04 |
| contrastive_B256  | (256, 32, 256, 180)  |   4.276 |  0.849 | 5.03× | 1.45 → 0.04 |
| contrastive_B512  | (512, 32, 512, 180)  |  17.752 |  2.940 | 6.04× | 5.68 → 0.06 |
| longdoc_B64       | (64, 128, 64, 512)   |   2.476 |  0.303 | 8.17× | 1.04 → 0.04 |
| longdoc_B128      | (128, 128, 128, 512) |   9.826 |  0.798 | 12.32× | 4.05 → 0.05 |
| colpali_B16       | (16, 1024, 16, 1024) |   2.416 |  0.323 | 7.47× | 1.04 → 0.04 |
| colpali_B32       | (32, 1024, 32, 1024) |   9.409 |  0.785 | 11.99× | 4.05 → 0.05 |

## §2 — KD forward (`Nq × K × Ld × d`, per-query candidate lists)

| label | shape (Nq, K, Lq, Ld) | baseline ms | LIK ms | speedup |
| --- | --- | ---: | ---: | ---: |
| kd_B16_K8_short  | (16, 8, 32, 180)    | 0.164 | 0.143 | 1.15× |
| kd_B32_K16_short | (32, 16, 32, 180)   | 0.168 | 0.160 | 1.05× |
| kd_B32_K32_short | (32, 32, 32, 180)   | 0.174 | 0.168 | 1.03× |
| kd_B64_K32_short | (64, 32, 32, 180)   | 0.253 | 0.184 | 1.37× |
| kd_B16_K8_long   | (16, 8, 128, 512)   | 0.182 | 0.146 | 1.25× |
| kd_B32_K16_long  | (32, 16, 128, 512)  | 0.428 | 0.174 | 2.46× |

The §2 regression flagged in the issue (Python for-loop over the K axis) is gone — KD now
routes through the same `autograd.maxsim` entrypoint as in-batch via the kd_layout path.

## §3 — In-batch training (fwd + bwd)

| label | shape (Nq, Lq, Nd, Ld) | baseline ms | LIK ms | speedup | peak GiB (base → LIK) |
| --- | --- | ---: | ---: | ---: | --- |
| colbert_B128       | (128, 32, 128, 180)  |   2.509 |  0.647 |  3.88× | 0.42 → 0.09 |
| colbert_B256       | (256, 32, 256, 180)  |   9.475 |  1.530 |  6.19× | 1.48 → 0.10 |
| colbert_B512       | (512, 32, 512, 180)  |  38.191 |  5.870 |  6.51× | 5.71 → 0.15 |
| colbert_B1K        | (1024, 32, 1024, 180) | 153.511 | 23.925 |  6.42× | 22.62 → 0.30 |
| colpali_train_B8   | (8, 1024, 8, 1024)   |   1.634 |  0.676 |  2.42× | 0.32 → 0.08 |
| colpali_train_B16  | (16, 1024, 16, 1024) |   6.078 |  0.658 |  9.23× | 1.07 → 0.09 |
| colpali_train_B32  | (32, 1024, 32, 1024) |  23.705 |  1.538 | 15.41× | 4.08 → 0.13 |

LIK 0.4.0's `auto` backward routes these gradient-heavy in-batch squares through the new
`lowmem` path (deterministic, no fp32 grad buffer), so the LIK training peak drops sharply
vs the 0.3.0 dispatcher (e.g. B=1K: 1.39 → 0.30 GiB) on top of the autotuned-launch speedup.

## §4 — KD training (fwd + bwd)

| label | shape (Nq, K, Lq, Ld) | baseline ms | LIK ms | speedup |
| --- | --- | ---: | ---: | ---: |
| kdtrain_B32_K16_short | (32, 16, 32, 180)  | 0.648 | 0.776 | 0.84× |
| kdtrain_B32_K32_short | (32, 32, 32, 180)  | 0.654 | 0.561 | 1.16× |
| kdtrain_B64_K32_short | (64, 32, 32, 180)  | 0.813 | 0.574 | 1.42× |
| kdtrain_B16_K8_long   | (16, 8, 128, 512)  | 0.623 | 0.578 | 1.08× |

`auto` routes the KD backward to `lowmem` (memory-optimal, deterministic). At the smallest
KD-train shape (B=32 K=16) that trades a little speed for lower memory and lands just behind
the einsum baseline; the larger KD shapes stay ahead.

## §5 — Pairwise forward (baseline only)

pylate's `colbert_scores_pairwise` uses a Python loop with `einsum("sh,th->st")` and
does not route through the dispatcher, so the LIK column is intentionally omitted.

| label | shape (batch, Lq, Ld) | baseline ms |
| --- | --- | ---: |
| pairwise_B256_text    | (256, 32, 180)    |  23.355 |
| pairwise_B1000_text   | (1000, 32, 180)   |  90.139 |
| pairwise_B5000_text   | (5000, 32, 180)   | 447.749 |
| pairwise_B500_colpali | (500, 1024, 1024) |  48.010 |

## §6 — Precision (vs fp64 reference, in-batch forward)

The kernel accumulates in fp32 internally, the baseline accumulates in fp16.

| label | shape | kernel | max abs Δ | mean abs Δ | max rel Δ | mean rel Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| short   | (8, 32, 32, 180)    | baseline | 4.71e-01 | 1.28e-01 | 4.53e-04 | 1.31e-04 |
| short   | (8, 32, 32, 180)    | lik      | **2.88e-04** | **1.58e-04** | **2.78e-07** | **1.62e-07** |
| longdoc | (8, 128, 32, 512)   | baseline | 2.16e+00 | 1.00e+00 | 4.89e-04 | 2.28e-04 |
| longdoc | (8, 128, 32, 512)   | lik      | **1.21e-03** | **7.46e-04** | **2.80e-07** | **1.70e-07** |
| colpali | (4, 1024, 16, 1024) | baseline | 1.57e+01 | 7.76e+00 | 4.22e-04 | 2.07e-04 |
| colpali | (4, 1024, 16, 1024) | lik      | **1.15e-02** | **6.41e-03** | **3.09e-07** | **1.71e-07** |

LIK is 3–4 orders of magnitude closer to the fp64 reference than the fp16 baseline.

## Raw data

- [`fast.json`](./fast.json) — §1, §2, §5, §6 (38 cells)
- [`train.json`](./train.json) — §3, §4 (22 cells)
