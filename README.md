# ODMA-URA

Research testbed for unsourced random access (URA), structured encoders and learned/model-based multiuser decoding.

Matched small-B tests show a large receiver gap on unchanged ODMA–polar codebooks. Polar-symbol energy shaping
approaches dense/D2 performance. The current focus is deriving nonlearned multiuser decoding from a shared posterior
on the fixed, deterministically balanced polar encoder. Further encoder geometry work is deferred; neither dense
optimality nor a scalable URA receiver is established.
The broad exact-derivation phase is complete: a finite Bayes reference and useful exact reductions are documented,
with no established tractable exact decoder at the intended scale. Approximation design now follows the Bayes route with algebraic cost/error derivations only. Exact foundations
are in [Report 7](docs/reports/07_polar_decoding_from_posterior.pdf); assumptions have their separate home in
[Report 9](docs/reports/09_polar_bayesian_approximations.pdf). Implementation/testing and PUPE panels are deferred. See the [current state](docs/CURRENT_STATE.md); the shared mixer is retired.
Primary PUPE now matches message multiplicities. Read the [11 October metric correction](docs/PUPE_METRIC.md)
before using historical IID scores; distinct-message matched comparisons retain their PUPE.

Global D0–D2 remain exponential in payload length. Generated hash/amplitude encoders and section-domain backends are
retained for historical reproduction, not active searches. Compact encoding or candidate-only refinement does not
establish scalable message search.

## Start Here

- [`docs/README.md`](docs/README.md): document ownership and retained technical history.
- [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md): fixed decoder problem, stage status and agreed algebra-only approach.
- [`docs/EXPERIMENT_BANK.md`](docs/EXPERIMENT_BANK.md): script-to-question/algebra/setup mapping, results and reproduction commands.
- [Exact research log](docs/POLAR_DECODING_RESEARCH_LOG.md) and [approximation research log](docs/POLAR_APPROXIMATION_LOG.md).
- [`results/03_results.md`](results/03_results.md): explicit dense-versus-ODMA and oracle-support evidence.
- [`results/04_results.md`](results/04_results.md): detailed framework, decoder, and sectioned-experiment evidence.
- [`jobs/README.md`](jobs/README.md): private HPC workflow and job commands.
- [`Job 033`](jobs/033_matched_d2/README.md): completed matched D2 controls, building on the published comparisons in 032.
- [`Job 034`](jobs/034_polar_mapping/README.md): 12/12 complete; symbol-energy shaping approaches dense D2, shared mixing does not help.

## Repository Layout

- `src/`: original ODMA scenario, classical/model-based decoders, sweeps, metrics, and bounds.
- `framework/`: factorised encoders, hash-skeleton generators, section-domain backend, the D0--D4 decoder ladder, outer
  code/BP, training, and analysis.
- `baselines/`: separate ODMA–polar, dynamic-CS and CCS-AMP native encoder/receiver chains.
- `benchmarks/`: matched native/explicit comparisons, result checks, plots and labelled bound calculations.
- `tests/`: executable experiments, merge/plot scripts, smoke tests, and algebraic regression tests.
- `jobs/`: numbered HPC manifests, scripts, logs, checkpoints, and returned outputs.
- `results/`: detailed result ledgers and generated local outputs.
- `docs/reports/`: LaTeX reports: exact Bayesian foundations in 7, fixed encoder in 8, approximation algebra in 9.

## Setup

```bash
uv sync
```

## Core Verification

```bash
uv run python -m compileall src framework baselines benchmarks tests
uv run python -m tests.framework_sectioned_refactor_test
uv run python -m tests.framework_sectioned_energy_test
uv run python -m tests.framework_hash_skeleton_test
uv run python -m tests.framework_prototype_amplitude_test
uv run python -m tests.framework_early_stopping_test
uv run python -m tests.single_test --decoders NNOMP SIC BlockMAP \
  --n 32 --d 8 --num-blocks 4 --num-codewords 16 \
  --num-devices-active 4 --num-antennas 2 --esn0-db 5
```

Generated outputs under `jobs/`, `results/`, and most of `docs/` are intentionally ignored by git. Do not interpret a
submitted job as a result: inspect summaries, logs, completion, numerical diagnostics, and merged plots first.
