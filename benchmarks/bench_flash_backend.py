"""Benchmark the flash-maxsim backend vs the default torch path.

Runs the three public scoring functions in `pylate.scores` on realistic shapes
and reports:

  - inference latency (forward-only, no grad)
  - training step latency (forward + backward)
  - peak activation memory
  - top-k ranking agreement with the torch reference

Each scoring path is exercised through the PR's `backend=` kwarg so the
numbers reflect exactly what ships in the PR.

Usage:
    python benchmarks/bench_flash_backend.py
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pylate.scores import (
    colbert_kd_scores,
    colbert_scores,
    colbert_scores_pairwise,
)

DEVICE = "cuda"
DTYPE = torch.float16


def _reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _bench(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    _reset()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2], torch.cuda.max_memory_allocated() / (1024**3)


def _try_bench(fn, warmup=3, runs=10):
    try:
        return ("OK",) + _bench(fn, warmup, runs)
    except torch.cuda.OutOfMemoryError:
        _reset()
        return ("OOM", float("nan"), float("nan"))


def _random_masked(n, lq, d, lq_range=None):
    embs = F.normalize(torch.randn(n, lq, d, dtype=DTYPE, device=DEVICE), dim=-1)
    if lq_range is None:
        return embs, torch.ones(n, lq, dtype=DTYPE, device=DEVICE)
    lo, hi = lq_range
    lens = torch.randint(lo, hi + 1, (n,), device=DEVICE)
    pos = torch.arange(lq, device=DEVICE).unsqueeze(0)
    mask = (pos < lens.unsqueeze(1)).to(DTYPE)
    return embs * mask.unsqueeze(-1), mask


def _topk_match(a, b, k=5):
    ta = a.argsort(dim=-1, descending=True)[..., :k]
    tb = b.argsort(dim=-1, descending=True)[..., :k]
    n = ta.reshape(-1, k).shape[0]
    agree = sum(
        1
        for ra, rb in zip(ta.reshape(-1, k), tb.reshape(-1, k))
        if set(ra.tolist()) == set(rb.tolist())
    )
    return agree, n


# ---------------------------------------------------------------------------
# 1) colbert_scores — inference (no grad)
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    Nq: int
    B: int
    Lq: int
    Ld: int
    d: int = 128


INFERENCE_SCENARIOS = [
    Scenario("contrastive B=32  Lq=32  Ld=180", 32, 32, 32, 180),
    Scenario("contrastive B=128 Lq=32  Ld=180", 128, 128, 32, 180),
    Scenario("ReasonCol   B=256 Lq=32  Ld=180", 256, 256, 32, 180),
    Scenario("long-query  B=64  Lq=128 Ld=512", 64, 64, 128, 512),
    Scenario("ColPali     B=32  Lq=1024 Ld=1024", 32, 32, 1024, 1024),
]


def bench_inference():
    print("=" * 110)
    print(" colbert_scores   (inference, forward-only, no grad)")
    print("=" * 110)
    print(
        f"  {'scenario':<32} {'torch_ms':>10} {'flash_ms':>10} {'speedup':>8} "
        f"{'torch_GB':>10} {'flash_GB':>10}  top-5 match"
    )
    print("  " + "-" * 108)

    for s in INFERENCE_SCENARIOS:
        torch.manual_seed(0)
        Q, Qm = _random_masked(s.Nq, s.Lq, s.d, lq_range=(max(1, s.Lq - 4), s.Lq))
        D, Dm = _random_masked(s.B, s.Ld, s.d, lq_range=(max(1, s.Ld // 2), s.Ld))

        def torch_step(Q=Q, D=D, Qm=Qm, Dm=Dm):
            return colbert_scores(Q, D, Qm, Dm, backend="torch")

        def flash_step(Q=Q, D=D, Qm=Qm, Dm=Dm):
            return colbert_scores(Q, D, Qm, Dm, backend="flash")

        st_t, t_ms, t_gb = _try_bench(torch_step)
        st_f, f_ms, f_gb = _try_bench(flash_step)

        if st_t == "OK" and st_f == "OK":
            with torch.no_grad():
                s_t = colbert_scores(Q, D, Qm, Dm, backend="torch")
                s_f = colbert_scores(Q, D, Qm, Dm, backend="flash")
            agree, n = _topk_match(s_t, s_f, k=min(5, s.B))
            match = f"{agree}/{n}"
            spd = f"{t_ms / f_ms:>6.2f}×"
        else:
            match = f"{st_t}/{st_f}"
            spd = "   n/a"

        t_ms_s = f"{t_ms:>8.2f}" if st_t == "OK" else "       OOM"
        t_gb_s = f"{t_gb:>7.2f}GB" if st_t == "OK" else "      OOM"
        f_ms_s = f"{f_ms:>8.2f}" if st_f == "OK" else "       OOM"
        f_gb_s = f"{f_gb:>7.2f}GB" if st_f == "OK" else "      OOM"
        print(
            f"  {s.name:<32} {t_ms_s:>10} {f_ms_s:>10} {spd:>8} "
            f"{t_gb_s:>10} {f_gb_s:>10}  {match}"
        )
        _reset()


# ---------------------------------------------------------------------------
# 2) colbert_scores — TRAINING step (forward + backward)
# ---------------------------------------------------------------------------
TRAIN_SCENARIOS = [
    # ColBERT-scale contrastive
    Scenario("ColBERT B=128 Lq=32  Ld=180", 128, 128, 32, 180),
    Scenario("ColBERT B=512 Lq=32  Ld=180", 512, 512, 32, 180),
    Scenario("ColBERT B=1K  Lq=32  Ld=180", 1024, 1024, 32, 180),
    Scenario("ColBERT B=2K  Lq=32  Ld=180", 2048, 2048, 32, 180),
    # ColPali-scale
    Scenario("ColPali B=8  Lq=1024 Ld=1024", 8, 8, 1024, 1024),
    Scenario("ColPali B=16 Lq=1024 Ld=1024", 16, 16, 1024, 1024),
    Scenario("ColPali B=32 Lq=1024 Ld=1024", 32, 32, 1024, 1024),
    Scenario("ColPali B=64 Lq=1024 Ld=1024", 64, 64, 1024, 1024),
]


def _training_step(backend: str, s: Scenario):
    torch.manual_seed(0)
    Q = F.normalize(
        torch.randn(s.Nq, s.Lq, s.d, dtype=DTYPE, device=DEVICE), dim=-1
    ).requires_grad_(True)
    D = F.normalize(
        torch.randn(s.B, s.Ld, s.d, dtype=DTYPE, device=DEVICE), dim=-1
    ).requires_grad_(True)
    qmask = torch.ones(s.Nq, s.Lq, dtype=DTYPE, device=DEVICE)
    dmask = torch.ones(s.B, s.Ld, dtype=DTYPE, device=DEVICE)
    scores = colbert_scores(Q, D, qmask, dmask, backend=backend)
    loss = scores.diagonal().sum()
    loss.backward()
    return loss.detach()


def bench_training():
    print()
    print("=" * 110)
    print(" colbert_scores   (TRAINING step — forward + backward)")
    print("=" * 110)
    print(
        f"  {'scenario':<32} {'torch_ms':>10} {'flash_ms':>10} {'speedup':>8} "
        f"{'torch_peak':>12} {'flash_peak':>12}  memory ratio"
    )
    print("  " + "-" * 108)

    for s in TRAIN_SCENARIOS:

        def torch_step():
            _training_step("torch", s)

        def flash_step():
            _training_step("flash", s)

        st_t, t_ms, t_gb = _try_bench(torch_step, warmup=1, runs=3)
        st_f, f_ms, f_gb = _try_bench(flash_step, warmup=1, runs=3)

        if st_t == "OK" and st_f == "OK":
            spd = f"{t_ms / f_ms:>6.2f}×"
            mem = f"{t_gb / max(f_gb, 1e-6):>6.1f}×"
        else:
            spd = "  n/a"
            mem = f"{'OOM→OK' if st_t == 'OOM' and st_f == 'OK' else '  n/a'}"

        t_ms_s = f"{t_ms:>8.2f}" if st_t == "OK" else "       OOM"
        t_gb_s = f"{t_gb:>9.2f}GB" if st_t == "OK" else "       OOM"
        f_ms_s = f"{f_ms:>8.2f}" if st_f == "OK" else "       OOM"
        f_gb_s = f"{f_gb:>9.2f}GB" if st_f == "OK" else "       OOM"
        print(
            f"  {s.name:<32} {t_ms_s:>10} {f_ms_s:>10} {spd:>8} "
            f"{t_gb_s:>12} {f_gb_s:>12}  {mem}"
        )
        _reset()


# ---------------------------------------------------------------------------
# 3) colbert_scores_pairwise — Python loop vs varlen
# ---------------------------------------------------------------------------
@dataclass
class PairScenario:
    name: str
    B: int
    Lq: int
    Ld: int
    d: int = 128


PAIR_SCENARIOS = [
    PairScenario("B=256  text", 256, 32, 180),
    PairScenario("B=1000 text", 1000, 32, 180),
    PairScenario("B=5000 text", 5000, 32, 180),
    PairScenario("B=500  ColPali", 500, 1024, 1024),
]


def bench_pairwise():
    print()
    print("=" * 110)
    print(" colbert_scores_pairwise   (Python loop → varlen one-shot)")
    print("=" * 110)
    print(
        f"  {'scenario':<32} {'torch_ms':>10} {'flash_ms':>10} {'speedup':>8}  max|Δ|"
    )
    print("  " + "-" * 108)

    for s in PAIR_SCENARIOS:
        torch.manual_seed(0)
        Q = F.normalize(torch.randn(s.B, s.Lq, s.d, dtype=DTYPE, device=DEVICE), dim=-1)
        D = F.normalize(torch.randn(s.B, s.Ld, s.d, dtype=DTYPE, device=DEVICE), dim=-1)

        def torch_step(Q=Q, D=D):
            return colbert_scores_pairwise(Q, D, backend="torch")

        def flash_step(Q=Q, D=D):
            return colbert_scores_pairwise(Q, D, backend="flash")

        st_t, t_ms, _ = _try_bench(torch_step, warmup=1, runs=3)
        st_f, f_ms, _ = _try_bench(flash_step)

        if st_t == "OK" and st_f == "OK":
            s_t = colbert_scores_pairwise(Q, D, backend="torch")
            s_f = colbert_scores_pairwise(Q, D, backend="flash")
            err = (s_t.float() - s_f.float()).abs().max().item()
            spd = f"{t_ms / f_ms:>6.2f}×"
        else:
            err = float("nan")
            spd = "  n/a"

        t_ms_s = f"{t_ms:>8.2f}" if st_t == "OK" else "       OOM"
        f_ms_s = f"{f_ms:>8.2f}" if st_f == "OK" else "       OOM"
        err_s = f"{err:.2e}" if not (err != err) else "n/a"
        print(f"  {s.name:<32} {t_ms_s:>10} {f_ms_s:>10} {spd:>8}  {err_s}")
        _reset()


# ---------------------------------------------------------------------------
# 4) colbert_kd_scores — per-query doc set (distillation)
# ---------------------------------------------------------------------------
KD_SCENARIOS = [
    ("KD B=16 K=8  Lq=32 Ld=180", 16, 8, 32, 180),
    ("KD B=16 K=32 Lq=32 Ld=180", 16, 32, 32, 180),
    ("KD B=32 K=16 Lq=32 Ld=180", 32, 16, 32, 180),
    ("KD B=16 K=8  Lq=128 Ld=512", 16, 8, 128, 512),
]

KD_TRAIN_SCENARIOS = [
    # Realistic KD training shapes — teacher-student distillation
    ("KDtrain B=32 K=32  Lq=32 Ld=180", 32, 32, 32, 180),
    ("KDtrain B=64 K=32  Lq=32 Ld=180", 64, 32, 32, 180),
    ("KDtrain B=64 K=64  Lq=32 Ld=180", 64, 64, 32, 180),
    ("KDtrain B=32 K=128 Lq=32 Ld=180", 32, 128, 32, 180),
    ("KDtrain B=16 K=8   Lq=128 Ld=512", 16, 8, 128, 512),
    ("KDtrain B=32 K=16  Lq=128 Ld=512", 32, 16, 128, 512),
]


def bench_kd():
    print()
    print("=" * 110)
    print(" colbert_kd_scores   (distillation — per-query document set)")
    print("=" * 110)
    print(
        f"  {'scenario':<32} {'torch_ms':>10} {'flash_ms':>10} {'speedup':>8} "
        f"{'torch_GB':>10} {'flash_GB':>10}  max|Δ|"
    )
    print("  " + "-" * 108)

    for name, Nq, K, Lq, Ld in KD_SCENARIOS:
        torch.manual_seed(0)
        Q, Qm = _random_masked(Nq, Lq, 128, lq_range=(max(1, Lq - 4), Lq))
        D = F.normalize(torch.randn(Nq, K, Ld, 128, dtype=DTYPE, device=DEVICE), dim=-1)
        d_lens_flat = torch.randint(
            Ld // 2, Ld + 1, (Nq * K,), device=DEVICE, dtype=torch.int32
        )
        pos = torch.arange(Ld, device=DEVICE)
        Dm = (pos.unsqueeze(0) < d_lens_flat.unsqueeze(1)).reshape(Nq, K, Ld).to(DTYPE)
        D = D * Dm.unsqueeze(-1)

        def torch_step(Q=Q, D=D, Qm=Qm, Dm=Dm):
            return colbert_kd_scores(Q, D, Qm, Dm, backend="torch")

        def flash_step(Q=Q, D=D, Qm=Qm, Dm=Dm):
            return colbert_kd_scores(Q, D, Qm, Dm, backend="flash")

        st_t, t_ms, t_gb = _try_bench(torch_step)
        st_f, f_ms, f_gb = _try_bench(flash_step)

        if st_t == "OK" and st_f == "OK":
            s_t = colbert_kd_scores(Q, D, Qm, Dm, backend="torch")
            s_f = colbert_kd_scores(Q, D, Qm, Dm, backend="flash")
            err = (s_t.float() - s_f.float()).abs().max().item()
            spd = f"{t_ms / f_ms:>6.2f}×"
        else:
            err = float("nan")
            spd = "  n/a"

        t_ms_s = f"{t_ms:>8.2f}" if st_t == "OK" else "       OOM"
        f_ms_s = f"{f_ms:>8.2f}" if st_f == "OK" else "       OOM"
        t_gb_s = f"{t_gb:>7.2f}GB" if st_t == "OK" else "      OOM"
        f_gb_s = f"{f_gb:>7.2f}GB" if st_f == "OK" else "      OOM"
        err_s = f"{err:.2e}" if not (err != err) else "n/a"
        print(
            f"  {name:<32} {t_ms_s:>10} {f_ms_s:>10} {spd:>8} "
            f"{t_gb_s:>10} {f_gb_s:>10}  {err_s}"
        )
        _reset()


def _kd_training_step(backend: str, Nq, K, Lq, Ld, d=128):
    torch.manual_seed(0)
    Q = F.normalize(
        torch.randn(Nq, Lq, d, dtype=DTYPE, device=DEVICE), dim=-1
    ).requires_grad_(True)
    D = F.normalize(
        torch.randn(Nq, K, Ld, d, dtype=DTYPE, device=DEVICE), dim=-1
    ).requires_grad_(True)
    qmask = torch.ones(Nq, Lq, dtype=DTYPE, device=DEVICE)
    dmask = torch.ones(Nq, K, Ld, dtype=DTYPE, device=DEVICE)
    scores = colbert_kd_scores(Q, D, qmask, dmask, backend=backend)  # [Nq, K]
    # KD-style target: log-softmax KL against a uniform teacher for timing
    loss = scores.sum()
    loss.backward()
    return loss.detach()


def bench_kd_training():
    print()
    print("=" * 110)
    print(" colbert_kd_scores   (TRAINING step — forward + backward, realistic shapes)")
    print("=" * 110)
    print(
        f"  {'scenario':<32} {'torch_ms':>10} {'flash_ms':>10} {'speedup':>8} "
        f"{'torch_peak':>12} {'flash_peak':>12}  memory ratio"
    )
    print("  " + "-" * 108)

    for name, Nq, K, Lq, Ld in KD_TRAIN_SCENARIOS:

        def torch_step():
            _kd_training_step("torch", Nq, K, Lq, Ld)

        def flash_step():
            _kd_training_step("flash", Nq, K, Lq, Ld)

        st_t, t_ms, t_gb = _try_bench(torch_step, warmup=1, runs=3)
        st_f, f_ms, f_gb = _try_bench(flash_step, warmup=1, runs=3)

        if st_t == "OK" and st_f == "OK":
            spd = f"{t_ms / f_ms:>6.2f}×"
            mem = f"{t_gb / max(f_gb, 1e-6):>6.1f}×"
        else:
            spd = "  n/a"
            mem = f"{'OOM→OK' if st_t == 'OOM' and st_f == 'OK' else '  n/a'}"

        t_ms_s = f"{t_ms:>8.2f}" if st_t == "OK" else "       OOM"
        t_gb_s = f"{t_gb:>9.2f}GB" if st_t == "OK" else "       OOM"
        f_ms_s = f"{f_ms:>8.2f}" if st_f == "OK" else "       OOM"
        f_gb_s = f"{f_gb:>9.2f}GB" if st_f == "OK" else "       OOM"
        print(
            f"  {name:<32} {t_ms_s:>10} {f_ms_s:>10} {spd:>8} "
            f"{t_gb_s:>12} {f_gb_s:>12}  {mem}"
        )
        _reset()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}  |  dtype={DTYPE}")
    import flash_maxsim

    print(f"flash_maxsim {flash_maxsim.__version__}\n")

    bench_inference()
    bench_training()
    bench_pairwise()
    bench_kd()
    bench_kd_training()


if __name__ == "__main__":
    main()
