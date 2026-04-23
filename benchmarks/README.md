# Benchmarks

## `bench_flash_backend.py`

Compares the default PyTorch path (`backend="torch"`) against the optional
`flash-maxsim` backend (`backend="flash"`) for the three public scoring
functions in `pylate.scores`:

- `colbert_scores` — inference (fwd-only) and training (fwd+bwd)
- `colbert_scores_pairwise` — Python loop → fused varlen kernel
- `colbert_kd_scores` — distillation (per-query teacher set), inference and
  training

Measures median latency (10 warm-up runs dropped, 10 timed runs), peak
activation memory, top-k ranking agreement, and correctness (max|Δ|).

### Run it

```bash
pip install "pylate[flash]"          # installs flash-maxsim
python benchmarks/bench_flash_backend.py
```

Requires a CUDA GPU. Tested on A100 80GB with flash-maxsim 0.2.1 and
torch 2.9 (fp16). See `PR_DESCRIPTION.md` for reference numbers.

### Notes

- All shapes match realistic ColBERT / ColPali / ReasonColBERT scenarios
  used in the pylate losses and evaluators.
- The training rows intentionally include batch sizes where the torch path
  OOMs on 80 GB — the point is to show flash unlocks them.
- KD wins are modest (~1.5×) and sometimes uses slightly more memory; see
  the write-up.
