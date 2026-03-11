"""Test script for contrastive vs cached-contrastive with colbert/xtr scoring.

Single-run mode:
    python test_cached_xtr.py --score xtr --loss cached --batch_size 64 --mini_batch_size 16

Sweep mode (grid over batch_size x mini_batch_size for each score/loss combo):
    python test_cached_xtr.py --sweep
    python test_cached_xtr.py --sweep --sweep_scores xtr --sweep_losses cached
    python test_cached_xtr.py --sweep --sweep_batch_sizes 16 32 64 128 --sweep_mini_batch_sizes 8 16 32

Validate mode (compare contrastive vs cached per-step losses):
    python test_cached_xtr.py --validate
    python test_cached_xtr.py --validate --validate_scores colbert xtr --validate_batch_size 16 --validate_steps 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

from pylate import evaluation, losses, models, scores, utils


class LossLoggerCallback(TrainerCallback):
    """Captures per-step training loss."""

    def __init__(self):
        self.step_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[{state.global_step}] loss: {logs['loss']}")
            self.step_losses.append(logs["loss"])


def run_single(args: argparse.Namespace) -> None:
    """Train a single configuration. Prints peak GPU memory and per-step losses."""
    run_name = (
        f"{args.score}-{args.loss}-bs{args.batch_size}"
        f"-mbs{args.mini_batch_size}-t{args.temperature}"
    )
    output_dir = f"output/{run_name}"

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model = models.ColBERT(model_name_or_path=args.model_name)

    score_fn = scores.xtr_scores if args.score == "xtr" else scores.colbert_scores

    if args.loss == "cached":
        train_loss = losses.CachedContrastive(
            model=model,
            score_metric=score_fn,
            mini_batch_size=args.mini_batch_size,
            temperature=args.temperature,
        )
    else:
        train_loss = losses.Contrastive(
            model=model,
            score_metric=score_fn,
            temperature=args.temperature,
        )

    dataset = load_dataset(
        "bclavie/msmarco-10m-triplets", split="train"
    )
    splits = dataset.train_test_split(test_size=1000, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        fp16=True,
        run_name=run_name,
        learning_rate=3e-6,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        report_to="wandb" if args.wandb else "none",
        seed=args.seed if args.seed is not None else 42,
        data_seed=args.seed if args.seed is not None else 42,
    )

    loss_logger = LossLoggerCallback()

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
        callbacks=[loss_logger],
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    # Print machine-readable result lines for the sweep/validate driver to parse
    print(f"__RESULT__:{json.dumps({'peak_mem_gb': round(peak_gb, 2)})}")
    print(f"__LOSSES__:{json.dumps(loss_logger.step_losses)}")


def run_sweep(args: argparse.Namespace) -> None:
    """Sweep over a grid of configs, running each as a subprocess."""
    configs = []
    for score in args.sweep_scores:
        for loss in args.sweep_losses:
            for bs in args.sweep_batch_sizes:
                if loss == "cached":
                    for mbs in args.sweep_mini_batch_sizes:
                        if mbs > bs:
                            continue
                        configs.append((score, loss, bs, mbs))
                else:
                    # mini_batch_size is irrelevant for non-cached
                    configs.append((score, loss, bs, 0))

    results = []
    for score, loss, bs, mbs in configs:
        label = f"{score:>7s}  {loss:>12s}  bs={bs:<4d}"
        if loss == "cached":
            label += f"  mbs={mbs:<4d}"
        else:
            label += "  mbs= -  "
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, __file__,
            "--score", score,
            "--loss", loss,
            "--batch_size", str(bs),
            "--mini_batch_size", str(mbs) if loss == "cached" else "1",
            "--max_steps", str(args.max_steps),
            "--temperature", str(args.temperature),
            "--model_name", args.model_name,
        ]

        t0 = time.time()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0

        # Parse result from subprocess stdout
        peak_mem = None
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT__:"):
                    data = json.loads(line.split(":", 1)[1])
                    peak_mem = data.get("peak_mem_gb")
            status = "OK"
        else:
            status = "OOM/FAIL"
            # Print last few lines of stderr for debugging
            stderr_lines = proc.stderr.strip().splitlines()
            for line in stderr_lines[-5:]:
                print(f"  stderr: {line}")

        results.append({
            "score": score, "loss": loss, "bs": bs, "mbs": mbs,
            "status": status, "peak_mem_gb": peak_mem,
            "elapsed_s": round(elapsed, 1),
        })
        mem_str = f"{peak_mem:.2f}" if peak_mem else "-"
        print(f"  => {status}  peak={mem_str} GB  time={elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'score':>7s}  {'loss':>12s}  {'bs':>4s}  {'mbs':>4s}  {'status':>8s}  {'peak_GB':>8s}  {'time_s':>7s}")
    print("-" * 80)
    for r in results:
        mbs_str = str(r["mbs"]) if r["loss"] == "cached" else "-"
        mem_str = f"{r['peak_mem_gb']:.2f}" if r["peak_mem_gb"] else "-"
        print(
            f"{r['score']:>7s}  {r['loss']:>12s}  {r['bs']:>4d}  {mbs_str:>4s}"
            f"  {r['status']:>8s}  {mem_str:>8s}  {r['elapsed_s']:>7.1f}"
        )


def _run_subprocess(score, loss, bs, mbs, max_steps, temperature, model_name, seed):
    """Run a single config as a subprocess, return (losses, peak_mem, returncode)."""
    cmd = [
        sys.executable, __file__,
        "--score", score,
        "--loss", loss,
        "--batch_size", str(bs),
        "--mini_batch_size", str(mbs),
        "--max_steps", str(max_steps),
        "--temperature", str(temperature),
        "--model_name", model_name,
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    step_losses = None
    peak_mem = None
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith("__LOSSES__:"):
                step_losses = json.loads(line.split(":", 1)[1])
            elif line.startswith("__RESULT__:"):
                data = json.loads(line.split(":", 1)[1])
                peak_mem = data.get("peak_mem_gb")
    return step_losses, peak_mem, proc.returncode, proc.stderr


def run_validate(args: argparse.Namespace) -> None:
    """Compare per-step losses between contrastive and cached contrastive.

    For each score type, runs contrastive as the baseline, then runs cached
    contrastive at each mini_batch_size in validate_mini_batch_sizes and
    compares per-step losses within tolerance.
    """
    bs = args.validate_batch_size
    steps = args.validate_steps
    seed = args.seed if args.seed is not None else 42
    tol = args.validate_tolerance
    mbs_list = args.validate_mini_batch_sizes
    all_passed = True

    for score in args.validate_scores:
        print(f"\n{'='*60}")
        print(f"Validating: {score} scoring  (bs={bs}, steps={steps}, seed={seed}, tol={tol})")
        print(f"{'='*60}")

        # Run contrastive as baseline (once per score type)
        print(f"\n  Running {score} contrastive (baseline)...")
        losses_c, _, rc_c, stderr_c = _run_subprocess(
            score, "contrastive", bs, bs, steps, args.temperature, args.model_name, seed
        )
        if rc_c != 0:
            print(f"  FAILED (contrastive): {stderr_c.strip().splitlines()[-3:]}")
            all_passed = False
            continue

        # Test each mini_batch_size
        for mbs in mbs_list:
            if mbs > bs:
                continue
            print(f"\n  Running {score} cached (mbs={mbs})...")
            losses_cc, _, rc_cc, stderr_cc = _run_subprocess(
                score, "cached", bs, mbs, steps, args.temperature, args.model_name, seed
            )
            if rc_cc != 0:
                print(f"  FAILED (cached mbs={mbs}): {stderr_cc.strip().splitlines()[-3:]}")
                all_passed = False
                continue

            # Compare
            n = min(len(losses_c), len(losses_cc))
            if n == 0:
                print("  WARNING: No losses captured. Check logging_steps.")
                all_passed = False
                continue

            header = f"  {'step':>4s}  {'contrastive':>14s}  {'cached(mbs='+str(mbs)+')':>14s}  {'diff':>12s}  {'status':>6s}"
            print(f"\n{header}")
            print(f"  {'-'*len(header)}")
            step_passed = True
            max_diff = 0.0
            for i in range(n):
                diff = abs(losses_c[i] - losses_cc[i])
                max_diff = max(max_diff, diff)
                ok = diff <= tol
                if not ok:
                    step_passed = False
                status = "OK" if ok else "FAIL"
                print(
                    f"  {i+1:>4d}  {losses_c[i]:>14.6f}  {losses_cc[i]:>14.6f}"
                    f"  {diff:>12.2e}  {status:>6s}"
                )

            if step_passed:
                print(f"\n  PASSED (mbs={mbs}): All {n} steps match within tol={tol} (max_diff={max_diff:.2e})")
            else:
                print(f"\n  FAILED (mbs={mbs}): max_diff={max_diff:.2e} exceeds tol={tol}")
                all_passed = False

    if all_passed:
        print(f"\n{'='*60}")
        print("ALL VALIDATIONS PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("SOME VALIDATIONS FAILED")
        print(f"{'='*60}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    # Single-run args
    parser.add_argument("--score", choices=["colbert", "xtr"], default="colbert")
    parser.add_argument("--loss", choices=["contrastive", "cached"], default="contrastive")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--model_name", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    # Sweep args
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep_scores", nargs="+", default=["colbert", "xtr"])
    parser.add_argument("--sweep_losses", nargs="+", default=["contrastive", "cached"])
    parser.add_argument("--sweep_batch_sizes", nargs="+", type=int, default=[128, 196, 256, 300, 384, 450, 512])
    parser.add_argument("--sweep_mini_batch_sizes", nargs="+", type=int, default=[8, 16, 32])
    # Validate args
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate_scores", nargs="+", default=["colbert", "xtr"])
    parser.add_argument("--validate_batch_size", type=int, default=16)
    parser.add_argument("--validate_mini_batch_sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--validate_steps", type=int, default=10)
    parser.add_argument("--validate_tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    if args.validate:
        run_validate(args)
    elif args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()