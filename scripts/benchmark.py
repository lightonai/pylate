"""Reproduce the issue #224 benchmark (baseline vs LIK only).

Covers the five timing sections of
https://github.com/lightonai/pylate/issues/224:

  §1 ``colbert_scores`` in-batch inference (forward only)
  §2 ``colbert_kd_scores`` distillation (forward only)
  §3 ``colbert_scores`` in-batch training (forward + backward)
  §4 ``colbert_kd_scores`` distillation training (forward + backward)
  §5 ``colbert_scores_pairwise`` packed / pair-rerank
  §6 forward precision vs an fp64 ground truth

Each cell is the median of ``--iters`` timed runs (``--warmup`` warmup) using
``torch.cuda.Event`` for GPU timing. The baseline run forces the einsum path
by setting ``PYLATE_DISABLE_LIK=1`` in-process; both kernels share the same
interpreter so module-import cost does not skew the numbers.

Output: one JSON file with metadata + a record per cell. Run with one or more
``--sections`` (e.g. ``--sections 1 2 5``); omit to run all sections.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from pylate.scores import colbert_kd_scores, colbert_scores, colbert_scores_pairwise


_DTYPE_LOOKUP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# Shapes from the issue (Nq column is always the query batch).
_INBATCH_SHAPES: list[dict] = [
    {"label": "contrastive_B32",   "nq": 32,  "lq": 32,   "nd": 32,  "ld": 180},
    {"label": "contrastive_B128",  "nq": 128, "lq": 32,   "nd": 128, "ld": 180},
    {"label": "contrastive_B256",  "nq": 256, "lq": 32,   "nd": 256, "ld": 180},
    {"label": "contrastive_B512",  "nq": 512, "lq": 32,   "nd": 512, "ld": 180},
    {"label": "longdoc_B64",       "nq": 64,  "lq": 128,  "nd": 64,  "ld": 512},
    {"label": "longdoc_B128",      "nq": 128, "lq": 128,  "nd": 128, "ld": 512},
    {"label": "colpali_B16",       "nq": 16,  "lq": 1024, "nd": 16,  "ld": 1024},
    {"label": "colpali_B32",       "nq": 32,  "lq": 1024, "nd": 32,  "ld": 1024},
]

_KD_SHAPES: list[dict] = [
    {"label": "kd_B16_K8_short",   "nq": 16, "k": 8,  "lq": 32,  "ld": 180},
    {"label": "kd_B32_K16_short",  "nq": 32, "k": 16, "lq": 32,  "ld": 180},
    {"label": "kd_B32_K32_short",  "nq": 32, "k": 32, "lq": 32,  "ld": 180},
    {"label": "kd_B64_K32_short",  "nq": 64, "k": 32, "lq": 32,  "ld": 180},
    {"label": "kd_B16_K8_long",    "nq": 16, "k": 8,  "lq": 128, "ld": 512},
    {"label": "kd_B32_K16_long",   "nq": 32, "k": 16, "lq": 128, "ld": 512},
]

_TRAIN_INBATCH_SHAPES: list[dict] = [
    {"label": "colbert_B128",       "nq": 128,  "lq": 32,   "nd": 128,  "ld": 180},
    {"label": "colbert_B256",       "nq": 256,  "lq": 32,   "nd": 256,  "ld": 180},
    {"label": "colbert_B512",       "nq": 512,  "lq": 32,   "nd": 512,  "ld": 180},
    {"label": "colbert_B1K",        "nq": 1024, "lq": 32,   "nd": 1024, "ld": 180},
    {"label": "colpali_train_B8",   "nq": 8,    "lq": 1024, "nd": 8,    "ld": 1024},
    {"label": "colpali_train_B16",  "nq": 16,   "lq": 1024, "nd": 16,   "ld": 1024},
    {"label": "colpali_train_B32",  "nq": 32,   "lq": 1024, "nd": 32,   "ld": 1024},
]

_TRAIN_KD_SHAPES: list[dict] = [
    {"label": "kdtrain_B32_K16_short",  "nq": 32, "k": 16, "lq": 32,  "ld": 180},
    {"label": "kdtrain_B32_K32_short",  "nq": 32, "k": 32, "lq": 32,  "ld": 180},
    {"label": "kdtrain_B64_K32_short",  "nq": 64, "k": 32, "lq": 32,  "ld": 180},
    {"label": "kdtrain_B16_K8_long",    "nq": 16, "k": 8,  "lq": 128, "ld": 512},
]

_PAIRWISE_SHAPES: list[dict] = [
    {"label": "pairwise_B256_text",     "batch": 256,  "lq": 32,   "ld": 180},
    {"label": "pairwise_B1000_text",    "batch": 1000, "lq": 32,   "ld": 180},
    {"label": "pairwise_B5000_text",    "batch": 5000, "lq": 32,   "ld": 180},
    {"label": "pairwise_B500_colpali",  "batch": 500,  "lq": 1024, "ld": 1024},
]

_PRECISION_SHAPES: list[dict] = [
    {"label": "short",   "nq": 8,  "lq": 32,   "nd": 32, "ld": 180},
    {"label": "longdoc", "nq": 8,  "lq": 128,  "nd": 32, "ld": 512},
    {"label": "colpali", "nq": 4,  "lq": 1024, "nd": 16, "ld": 1024},
]


def _set_lik(enabled: bool) -> None:
    """Toggle the LIK dispatch via the env-var kill switch."""
    if enabled:
        os.environ.pop("PYLATE_DISABLE_LIK", None)
        os.environ.pop("LIK_DISABLE", None)
        return
    os.environ["PYLATE_DISABLE_LIK"] = "1"


def _time_callable(
    fn: Callable[[], None], iters: int, warmup: int
) -> tuple[float, float, float, int]:
    """Median/mean/stddev wall-time (ms) and peak GPU bytes for ``fn``."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    peak_bytes: int = torch.cuda.max_memory_allocated()
    stddev: float = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return statistics.median(times_ms), statistics.mean(times_ms), stddev, peak_bytes


def _record(shape: dict, kernel: str, op: str, dtype: str, stats: tuple) -> dict:
    """Pack one timing cell into a JSON-friendly dict."""
    median_ms, mean_ms, stddev_ms, peak_bytes = stats
    return {
        "label": shape["label"],
        "shape": {k: v for k, v in shape.items() if k != "label"},
        "kernel": kernel,
        "op": op,
        "dtype": dtype,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "stddev_ms": stddev_ms,
        "peak_gpu_bytes": peak_bytes,
        "peak_gpu_gib": peak_bytes / (1024**3),
    }


def _section1_inbatch_fwd(dtype: torch.dtype, iters: int, warmup: int) -> list[dict]:
    """§1 — ``colbert_scores`` in-batch, forward only, ones-mask."""
    records: list[dict] = []
    for shape in _INBATCH_SHAPES:
        nq, lq, nd, ld = shape["nq"], shape["lq"], shape["nd"], shape["ld"]
        for use_lik in (False, True):
            _set_lik(use_lik)
            torch.cuda.empty_cache()
            query = torch.randn(nq, lq, 128, device="cuda", dtype=dtype)
            doc = torch.randn(nd, ld, 128, device="cuda", dtype=dtype)
            qmask = torch.ones(nq, lq, device="cuda", dtype=torch.bool)
            dmask = torch.ones(nd, ld, device="cuda", dtype=torch.bool)

            def _step() -> None:
                with torch.no_grad():
                    colbert_scores(query, doc, qmask, dmask)

            stats = _time_callable(_step, iters, warmup)
            kernel = "lik" if use_lik else "baseline"
            records.append(_record(shape, kernel, "fwd", str(dtype).removeprefix("torch."), stats))
            print(
                f"  §1 {shape['label']:20s} {kernel:9s} median={stats[0]:7.3f}ms"
            )
    return records


def _section2_kd_fwd(dtype: torch.dtype, iters: int, warmup: int) -> list[dict]:
    """§2 — ``colbert_kd_scores``, forward only, ones-mask."""
    records: list[dict] = []
    for shape in _KD_SHAPES:
        nq, k, lq, ld = shape["nq"], shape["k"], shape["lq"], shape["ld"]
        for use_lik in (False, True):
            _set_lik(use_lik)
            torch.cuda.empty_cache()
            query = torch.randn(nq, lq, 128, device="cuda", dtype=dtype)
            doc = torch.randn(nq, k, ld, 128, device="cuda", dtype=dtype)
            qmask = torch.ones(nq, lq, device="cuda", dtype=torch.bool)
            dmask = torch.ones(nq, k, ld, device="cuda", dtype=torch.bool)

            def _step() -> None:
                with torch.no_grad():
                    colbert_kd_scores(query, doc, qmask, dmask)

            stats = _time_callable(_step, iters, warmup)
            kernel = "lik" if use_lik else "baseline"
            records.append(_record(shape, kernel, "fwd", str(dtype).removeprefix("torch."), stats))
            print(
                f"  §2 {shape['label']:20s} {kernel:9s} median={stats[0]:7.3f}ms"
            )
    return records


def _section3_inbatch_train(dtype: torch.dtype, iters: int, warmup: int) -> list[dict]:
    """§3 — ``colbert_scores`` training (forward + backward)."""
    records: list[dict] = []
    for shape in _TRAIN_INBATCH_SHAPES:
        nq, lq, nd, ld = shape["nq"], shape["lq"], shape["nd"], shape["ld"]
        for use_lik in (False, True):
            _set_lik(use_lik)
            torch.cuda.empty_cache()
            query = torch.randn(nq, lq, 128, device="cuda", dtype=dtype, requires_grad=True)
            doc = torch.randn(nd, ld, 128, device="cuda", dtype=dtype, requires_grad=True)
            qmask = torch.ones(nq, lq, device="cuda", dtype=torch.bool)
            dmask = torch.ones(nd, ld, device="cuda", dtype=torch.bool)

            def _step() -> None:
                scores = colbert_scores(query, doc, qmask, dmask)
                scores.sum().backward()
                query.grad = None
                doc.grad = None

            stats = _time_callable(_step, iters, warmup)
            kernel = "lik" if use_lik else "baseline"
            records.append(_record(shape, kernel, "fwdbwd", str(dtype).removeprefix("torch."), stats))
            print(
                f"  §3 {shape['label']:22s} {kernel:9s} median={stats[0]:7.3f}ms "
                f"peak={stats[3] / (1024**3):.2f}GiB"
            )
    return records


def _section4_kd_train(dtype: torch.dtype, iters: int, warmup: int) -> list[dict]:
    """§4 — ``colbert_kd_scores`` training (forward + backward)."""
    records: list[dict] = []
    for shape in _TRAIN_KD_SHAPES:
        nq, k, lq, ld = shape["nq"], shape["k"], shape["lq"], shape["ld"]
        for use_lik in (False, True):
            _set_lik(use_lik)
            torch.cuda.empty_cache()
            query = torch.randn(nq, lq, 128, device="cuda", dtype=dtype, requires_grad=True)
            doc = torch.randn(nq, k, ld, 128, device="cuda", dtype=dtype, requires_grad=True)
            qmask = torch.ones(nq, lq, device="cuda", dtype=torch.bool)
            dmask = torch.ones(nq, k, ld, device="cuda", dtype=torch.bool)

            def _step() -> None:
                scores = colbert_kd_scores(query, doc, qmask, dmask)
                scores.sum().backward()
                query.grad = None
                doc.grad = None

            stats = _time_callable(_step, iters, warmup)
            kernel = "lik" if use_lik else "baseline"
            records.append(_record(shape, kernel, "fwdbwd", str(dtype).removeprefix("torch."), stats))
            print(
                f"  §4 {shape['label']:22s} {kernel:9s} median={stats[0]:7.3f}ms "
                f"peak={stats[3] / (1024**3):.2f}GiB"
            )
    return records


def _section5_pairwise(dtype: torch.dtype, iters: int, warmup: int) -> list[dict]:
    """§5 — ``colbert_scores_pairwise`` packed / pair-rerank.

    ``colbert_scores_pairwise`` does not go through ``pylate.utils.maxsim`` —
    it has its own Python for-loop. The dispatcher therefore has no LIK
    route for this workload today, so we only time the baseline here and
    skip the LIK column entirely. (Routing pair-rerank through the inbatch
    kernel + ``.diagonal()`` would compute an [B, B] matrix and waste B²/B
    of the work, which is not a fair comparison.)
    """
    records: list[dict] = []
    for shape in _PAIRWISE_SHAPES:
        batch, lq, ld = shape["batch"], shape["lq"], shape["ld"]
        _set_lik(False)
        torch.cuda.empty_cache()
        q_pair = torch.randn(batch, lq, 128, device="cuda", dtype=dtype)
        d_pair = torch.randn(batch, ld, 128, device="cuda", dtype=dtype)

        def _baseline_step() -> None:
            with torch.no_grad():
                colbert_scores_pairwise(q_pair, d_pair)

        stats = _time_callable(_baseline_step, iters, warmup)
        records.append(_record(shape, "baseline", "fwd", str(dtype).removeprefix("torch."), stats))
        print(f"  §5 {shape['label']:24s} baseline  median={stats[0]:8.3f}ms")
    return records


def _section6_precision(dtype: torch.dtype) -> list[dict]:
    """§6 — forward precision against an fp64 ground truth.

    Ground truth: ``einsum("ash,bth->abst").max(-1).sum(-1)`` in fp64 over the
    full ``[Nq, Nd]`` matrix, chunked one query at a time to bound memory.
    """
    records: list[dict] = []
    for shape in _PRECISION_SHAPES:
        nq, lq, nd, ld = shape["nq"], shape["lq"], shape["nd"], shape["ld"]
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        query = torch.randn(nq, lq, 128, device="cuda", dtype=dtype)
        doc = torch.randn(nd, ld, 128, device="cuda", dtype=dtype)

        # fp64 reference, chunked over Nq to avoid an [Nq, Nd, Lq, Ld] tensor.
        ref = torch.empty(nq, nd, device="cuda", dtype=torch.float64)
        query_fp64 = query.to(torch.float64)
        doc_fp64 = doc.to(torch.float64)
        for q_index in range(nq):
            scores_q = torch.einsum("sh,bth->bst", query_fp64[q_index], doc_fp64)
            ref[q_index] = scores_q.max(dim=-1).values.sum(dim=-1)

        for use_lik in (False, True):
            _set_lik(use_lik)
            with torch.no_grad():
                got = colbert_scores(query, doc).to(torch.float64)
            abs_err = (got - ref).abs()
            rel_err = abs_err / ref.abs().clamp_min(1e-12)
            kernel = "lik" if use_lik else "baseline"
            records.append({
                "label": shape["label"],
                "shape": {k: v for k, v in shape.items() if k != "label"},
                "kernel": kernel,
                "op": "fwd",
                "dtype": str(dtype).removeprefix("torch."),
                "max_abs": abs_err.max().item(),
                "mean_abs": abs_err.mean().item(),
                "max_rel": rel_err.max().item(),
                "mean_rel": rel_err.mean().item(),
            })
            print(
                f"  §6 {shape['label']:10s} {kernel:9s} "
                f"max|Δ|={abs_err.max().item():.3e} mean|Δ|={abs_err.mean().item():.3e}"
            )
    return records


_SECTION_DISPATCH: dict[int, Callable] = {
    1: _section1_inbatch_fwd,
    2: _section2_kd_fwd,
    3: _section3_inbatch_train,
    4: _section4_kd_train,
    5: _section5_pairwise,
    6: _section6_precision,
}


def _run_section(section: int, dtype: torch.dtype, iters: int, warmup: int) -> list[dict]:
    """Dispatch to a section handler. §6 ignores iters/warmup."""
    if section == 6:
        return _SECTION_DISPATCH[section](dtype)
    return _SECTION_DISPATCH[section](dtype, iters, warmup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench_results/issue224.json")
    parser.add_argument(
        "--sections", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
        choices=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPE_LOOKUP), default="fp16")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = _DTYPE_LOOKUP[args.dtype]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"capability: {torch.cuda.get_device_capability()}")
    print(f"torch: {torch.__version__}")
    print(f"dtype: {args.dtype}  iters: {args.iters}  warmup: {args.warmup}")

    started_at = time.time()
    all_records: list[dict] = []
    for section in args.sections:
        print(f"\n=== Section {section} ===")
        all_records.extend(_run_section(section, dtype, args.iters, args.warmup))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "device_name": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "torch_version": torch.__version__,
                "dtype": args.dtype,
                "iters": args.iters,
                "warmup": args.warmup,
                "seed": args.seed,
                "sections": args.sections,
                "elapsed_s": time.time() - started_at,
                "records": all_records,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path} ({len(all_records)} cells)")


if __name__ == "__main__":
    main()
