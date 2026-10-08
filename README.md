# ODMA-URA

Research testbed for unsourced random access (URA), structured encoders and learned/model-based multiuser decoding.

Matched small-B tests show a large receiver gap on unchanged ODMA–polar codebooks; dense D2 still retains an edge.
The current question is how to retain strong explicit-codebook recovery with tractable inference. Polar coding and
resource placement are separate design choices: using polar structure need not commit the framework to a fixed ODMA
construction. Job 034 tests learned polar-symbol energy and a shared pre-placement mixer with unchanged D2;
a new tractable receiver is still a derivation question. See the [current handoff](docs/CURRENT_STATE.md).

Global D0–D2 remain exponential in payload length. Generated hash/amplitude encoders and section-domain backends are
retained as research controls; compact encoding or candidate-only refinement does not establish scalable message search.

## Start Here

- [`docs/README.md`](docs/README.md): chronological research narrative and document map.
- [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md): neutral handoff for a new conversation or supervisor meeting.
- [`docs/EXPERIMENT_BANK.md`](docs/EXPERIMENT_BANK.md): concise operational record of the latest experiment banks.
- [`results/03_results.md`](results/03_results.md): explicit dense-versus-ODMA and oracle-support evidence.
- [`results/04_results.md`](results/04_results.md): detailed framework, decoder, and sectioned-experiment evidence.
- [`jobs/README.md`](jobs/README.md): private HPC workflow and job commands.
- [`Job 033`](jobs/033_matched_d2/README.md): completed matched D2 controls, building on the published comparisons in 032.
- [`Job 034`](jobs/034_polar_mapping/README.md): matched polar mapping extensions; local checks pass, HPC results pending.

## Repository Layout

- `src/`: original ODMA scenario, classical/model-based decoders, sweeps, metrics, and bounds.
- `framework/`: factorised encoders, hash-skeleton generators, section-domain backend, the D0--D4 decoder ladder, outer
  code/BP, training, and analysis.
- `baselines/`: separate ODMA–polar, dynamic-CS and CCS-AMP native encoder/receiver chains.
- `benchmarks/`: matched native/explicit comparisons, result checks, plots and labelled bound calculations.
- `tests/`: executable experiments, merge/plot scripts, smoke tests, and algebraic regression tests.
- `jobs/`: numbered HPC manifests, scripts, logs, checkpoints, and returned outputs.
- `results/`: detailed result ledgers and generated local outputs.
- `docs/reports/`: chronological LaTeX reports and the supervisor meeting summary; not the live roadmap.

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
