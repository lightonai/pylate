# Issue #224 benchmark — pylate `backend=` torch vs lik

Reproduces the timing layout of [issue #224](https://github.com/lightonai/pylate/issues/224),
comparing `backend="torch"` (the einsum reference) against `backend="lik"`
([`late-interaction-kernels`](https://github.com/hcompai/late-interaction-kernels) `0.4.0`).

## Setup

| | |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 (capability 9.0) |
| CUDA | 12.4 (runtime image), torch 2.11.0+cu130 |
| dtype | fp16 |
| iters / warmup | 10 / 3 |
| LIK | 0.4.0 (PyPI) |
| Harness | [`scripts/benchmark.py`](../../scripts/benchmark.py) via [`scripts/sky_benchmark.yaml`](../../scripts/sky_benchmark.yaml) |

`baseline` = `backend="torch"` (einsum reference). `lik` = `backend="lik"`
(`autograd.maxsim` for in-batch/KD, `maxsim_pairs` for pairwise).

## §1 — In-batch forward

| label | shape (Nq, Lq, Nd, Ld) | baseline ms | LIK ms | speedup | peak GiB (base → LIK) |
| --- | --- | ---: | ---: | ---: | --- |
| contrastive_B32   | (32, 32, 32, 180)    |   0.167 |  0.180 | 0.93× | 0.06 → 0.03 |
| contrastive_B128  | (128, 32, 128, 180)  |   1.110 |  0.325 | 3.41× | 0.39 → 0.04 |
| contrastive_B256  | (256, 32, 256, 180)  |   4.171 |  0.853 | 4.89× | 1.45 → 0.04 |
| contrastive_B512  | (512, 32, 512, 180)  |  17.753 |  2.950 | 6.02× | 5.68 → 0.06 |
| longdoc_B64       | (64, 128, 64, 512)   |   2.515 |  0.313 | 8.03× | 1.04 → 0.04 |
| longdoc_B128      | (128, 128, 128, 512) |   9.627 |  0.803 | 12.00× | 4.05 → 0.05 |
| colpali_B16       | (16, 1024, 16, 1024) |   2.401 |  0.325 | 7.38× | 1.04 → 0.04 |
| colpali_B32       | (32, 1024, 32, 1024) |   9.359 |  0.791 | 11.84× | 4.05 → 0.05 |

## §2 — KD forward

| label | shape (Nq, K, Lq, Ld) | baseline ms | LIK ms | speedup |
| --- | --- | ---: | ---: | ---: |
| kd_B16_K8_short  | (16, 8, 32, 180)    | 0.165 | 0.152 | 1.09× |
| kd_B32_K16_short | (32, 16, 32, 180)   | 0.170 | 0.174 | 0.98× |
| kd_B32_K32_short | (32, 32, 32, 180)   | 0.175 | 0.173 | 1.01× |
| kd_B64_K32_short | (64, 32, 32, 180)   | 0.252 | 0.183 | 1.38× |
| kd_B16_K8_long   | (16, 8, 128, 512)   | 0.181 | 0.150 | 1.20× |
| kd_B32_K16_long  | (32, 16, 128, 512)  | 0.429 | 0.186 | 2.31× |

## §3 — In-batch training (fwd + bwd)

| label | shape (Nq, Lq, Nd, Ld) | baseline ms | LIK ms | speedup | peak GiB (base → LIK) |
| --- | --- | ---: | ---: | ---: | --- |
| colbert_B128       | (128, 32, 128, 180)  |   2.477 |  0.696 |  3.56× | 0.42 → 0.09 |
| colbert_B256       | (256, 32, 256, 180)  |   9.404 |  1.602 |  5.87× | 1.48 → 0.10 |
| colbert_B512       | (512, 32, 512, 180)  |  38.416 |  5.901 |  6.51× | 5.71 → 0.15 |
| colbert_B1K        | (1024, 32, 1024, 180) | 153.518 | 23.806 |  6.45× | 22.62 → 0.30 |
| colpali_train_B8   | (8, 1024, 8, 1024)   |   1.639 |  0.671 |  2.44× | 0.32 → 0.08 |
| colpali_train_B16  | (16, 1024, 16, 1024) |   6.071 |  0.643 |  9.44× | 1.07 → 0.09 |
| colpali_train_B32  | (32, 1024, 32, 1024) |  23.816 |  1.518 | 15.69× | 4.08 → 0.13 |

## §4 — KD training (fwd + bwd)

| label | shape (Nq, K, Lq, Ld) | baseline ms | LIK ms | speedup |
| --- | --- | ---: | ---: | ---: |
| kdtrain_B32_K16_short | (32, 16, 32, 180)  | 0.529 | 0.561 | 0.94× |
| kdtrain_B32_K32_short | (32, 32, 32, 180)  | 0.601 | 0.547 | 1.10× |
| kdtrain_B64_K32_short | (64, 32, 32, 180)  | 0.788 | 0.543 | 1.45× |
| kdtrain_B16_K8_long   | (16, 8, 128, 512)  | 0.525 | 0.622 | 0.84× |

Sub-millisecond KD-train shapes: `auto` routes the KD backward to LIK's deterministic
`lowmem` path, which trades a little speed for lower memory and can land just behind the
einsum baseline at these tiny shapes. KD step time is dominated by the forward (§2).

## §5 — Pairwise forward (`colbert_scores_pairwise`)

Torch path is pylate's per-pair Python `for` loop; LIK routes through `maxsim_pairs`
(one fused diagonal launch, no loop), so the speedup mostly reflects removing the
Python-loop overhead.

| label | shape (batch, Lq, Ld) | baseline ms | LIK ms | speedup |
| --- | --- | ---: | ---: | ---: |
| pairwise_B256_text    | (256, 32, 180)    |   22.862 | 0.118 | 193× |
| pairwise_B1000_text   | (1000, 32, 180)   |   88.011 | 0.141 | 624× |
| pairwise_B5000_text   | (5000, 32, 180)   |  441.509 | 0.201 | 2194× |
| pairwise_B500_colpali | (500, 1024, 1024) |   47.010 | 0.499 | 94× |

## §6 — Precision (vs fp64 reference, in-batch forward)

| label | shape | kernel | max abs Δ | mean abs Δ | max rel Δ | mean rel Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| short   | (8, 32, 32, 180)    | baseline | 4.71e-01 | 1.28e-01 | 4.53e-04 | 1.31e-04 |
| short   | (8, 32, 32, 180)    | lik      | **2.88e-04** | **1.58e-04** | **2.78e-07** | **1.62e-07** |
| longdoc | (8, 128, 32, 512)   | baseline | 2.16e+00 | 1.00e+00 | 4.89e-04 | 2.28e-04 |
| longdoc | (8, 128, 32, 512)   | lik      | **1.21e-03** | **7.46e-04** | **2.80e-07** | **1.70e-07** |
| colpali | (4, 1024, 16, 1024) | baseline | 1.57e+01 | 7.76e+00 | 4.22e-04 | 2.07e-04 |
| colpali | (4, 1024, 16, 1024) | lik      | **1.15e-02** | **6.41e-03** | **3.09e-07** | **1.71e-07** |

LIK accumulates in fp32 internally, staying 3–4 orders of magnitude closer to the fp64
reference than the fp16 baseline.

## Raw data

- [`fast.json`](./fast.json) — §1, §2, §5, §6 (42 cells)
- [`train.json`](./train.json) — §3, §4 (22 cells)
