# 0011: `brasskey bench`: GuideLLM runs, intervals and verdicts

Date: 2026-11-08
Status: accepted

## Context

Milestone (h) adds `brasskey bench <endpoint>`. Brasskey's measurement rules: repeated runs,
confidence intervals on every metric, and verdicts of PASS, FAIL or INCONCLUSIVE that never claim a
gain the data can't support. GuideLLM (Apache-2.0, vLLM project) is the chosen load generator. It
must run outside Brasskey's process, because it pins torch and transformers, and it must not keep
prompt or response text.

## Options

- **Load generator:** GuideLLM, or a load generator of Brasskey's own. GuideLLM is maintained,
  SLO-aware, and does the token accounting.
- **Statistics:** SciPy (a large dependency for one quantile function), or pure Python (Student
  t from the regularized incomplete beta function, about 60 lines, checked against published
  tables).
- **Comparison scale:** absolute differences, or log ratios. Log ratios mean "12% faster" means
  the same at any speed, and they suit metrics that can't go below zero.

## Decision

- **GuideLLM 0.8.1.**
  - The `guidellm` command on PATH is used if present.
  - Otherwise a pinned `uv tool run` environment: dependencies resolved as of 2026-11-01, Python
    3.12, CPU-only torch (`--torch-backend cpu`; GuideLLM only tokenizes). About 300 MB to
    download, 1 GB on disk, set up once after the user confirms. Later runs start offline in
    0.2 s.
  - Otherwise an install hint (`brew install uv`).
- **Each run:**
  - synthetic prompts (`synthetic_text`) with a fixed token shape, a fixed number of concurrent
    streams (`concurrent` profile), a fixed duration and a seed;
  - `--metrics sample_size=0`, so GuideLLM keeps no prompt or response text;
  - the API key passed as `GUIDELLM__SPEC__BACKEND__API_KEY` (GuideLLM's settings variable),
    never on argv; GuideLLM masks it in its report;
  - run inside a private folder (0700), so a stray `.env` file can't change it. The raw report
    is deleted right after parsing, and the folder is removed at the end;
  - its log is private (0600).

  Verified against llama.cpp and MLX: the report holds no request text and no key.
- **Report parser.** It reads GuideLLM's report format 2 (`metadata.version`) and refuses any
  other version with a plain error, so an upgrade can't silently change what the numbers mean.
  Per run it keeps:
  - output tokens/s and requests/s (means);
  - time to first token (median, p95);
  - inter-token latency (median);
  - request latency (median, p95);
  - request counts.
- **Workload defaults:**
  - 256-token prompts and 128-token answers;
  - 4 concurrent requests;
  - 3 measured runs of 30 s after one 10 s warm-up per endpoint;
  - seed 42.

  Run i uses seed 42 + i, so a prefix cache can't flatter later runs. Every option can be set,
  and the plan and time estimate are confirmed first (`--yes` skips the prompt).
- **Comparisons.**
  - `--compare URL` measures a baseline now. The endpoints alternate and the order flips every
    repeat (A B, B A, …), so drift affects both. Both get the same tokenizer and seeds, so the
    prompts are identical.
  - `--baseline FILE` uses results saved with `--save`, and requires the same workload.
- **Statistics** (`bench/stats.py`, pure Python):
  - every metric: mean ± Student-t 95% interval; a run-to-run CV above 10% is flagged as noisy;
  - comparisons: Welch's t interval on log values. Improvement Δ is oriented so that positive
    means better. The target t is ln(1 + margin) for `--goal better` and ln(1 − margin) for
    `--goal not-worse`. Then PASS if lo ≥ t, FAIL if hi < t, otherwise INCONCLUSIVE.
  - Throughputs are rates of finished requests, so one run can't resolve less than one
    request: a relative step of 1/successful. Its variance (step² / 12, a uniform rounding error)
    is added to the run-to-run variance in every interval and comparison. Without it, runs that
    finished the same number of requests differ only by timing jitter and claim false precision
    (measured: 13 vs 10 requests per 10 s run gave "−23.1% (−23.1% to −23.1%)"; with it,
    −31.6% to −13.4%, still FAIL).
  - Fewer than two runs, a zero value, or identical values in every run with no known resolution
    (the latency metrics) give INCONCLUSIVE with the reason, never a zero-width interval.
  - The primary metric (`--metric`, default output throughput) decides the verdict. The other
    metrics are compared and shown for information.
  - Exit codes are 0 for PASS (or no comparison), 3 for FAIL and 4 for INCONCLUSIVE (ADR 0001).
- **Tokenizer.**
  - `--tokenizer` (a folder or a Hugging Face id) when given.
  - Otherwise, for catalog models, the pinned and hashed tokenizer files from the catalog: a
    cached copy, or a download of those files only (a few MB, confirmed).
  - Unknown models need `--tokenizer`.
- **Saving.** `--save FILE` writes `brasskey.bench/v1` JSON (numbers only). `--lock` appends a
  `LockedBenchmark` to `brasskey.lock` (an additive lock v1 field). Both are checked before any
  run, so a wrong path or model costs no time.

## Consequences

- No new Python dependencies. GuideLLM's dependencies stay in its own environment.
- uv labels `--torch-backend` experimental. If a uv release changes it, the GuideLLM setup fails
  with uv's message, and the fallback is `uv tool install guidellm==0.8.1` (GuideLLM on PATH).
- A run costs about 10 s of GuideLLM overhead beyond its duration, which is part of the estimate.
- Short runs give coarse request counts. The defaults (30 s × 3), the resolution term and the
  INCONCLUSIVE rule keep that honest: longer runs narrow the interval.
- Measured on the M3 Pro, Qwen3 0.6B, 2 concurrent, 3 × 8 s: MLX 4-bit gave +33% output
  throughput over llama.cpp Q8_0 (95% interval +17% to +51%): PASS. The quantizations differ, so
  this compares setups, not engines alone.
- Later: replaying real traffic shapes (GuideLLM trace replay), SLO goodput, and comparing
  several candidates at once.
