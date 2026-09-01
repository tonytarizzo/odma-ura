# ODMA-URA

Research testbed for unsourced random access (URA), structured codebooks, support recovery, and scalable section-domain
encoding/decoding.

The repository now separates two execution regimes:

- `framework.Encoder`: a small-`B` global-message backend with an implicit factorisation of
  `Phi in C^(n x M)`, where `M=2^B`. It avoids materialising `Phi` during normal computation but still keeps
  `M`-length message states.
- `framework.SectionedEncoder`: a scalable backend with local states of total size `sum_l N_l`, a procedural outer
  encoder, exact unit-energy construction, local D0 evidence, optional outer BP, and complete-path list extraction.

The current research conclusion is deliberately narrower than “sectioning works” or “sectioning fails.” Small-`B`
tests show that independently distributed sparse-global supports retain near-dense performance over a broad density
range, while a small reused ODMA support bank performs much worse even at equal density. The scalable backend is exactly
compatible with the global backend at `L=1`, but the current `L>1` route loses complete-message association as local
occupancy rises. The `B=128` implementation executes without a `2^B` object and satisfies unit energy, but its current
local decoder saturates and does not learn useful recovery.

## Start Here

- [`docs/README.md`](docs/README.md): chronological research narrative and document map.
- [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md): neutral handoff for a new conversation or supervisor meeting.
- [`docs/EXPERIMENT_BANK.md`](docs/EXPERIMENT_BANK.md): concise operational record of the latest experiment banks.
- [`results/03_results.md`](results/03_results.md): explicit dense-versus-ODMA and oracle-support evidence.
- [`results/04_results.md`](results/04_results.md): detailed framework, decoder, and sectioned-experiment evidence.
- [`jobs/README.md`](jobs/README.md): private HPC workflow and job commands.

## Repository Layout

- `src/`: original ODMA scenario, classical/model-based decoders, sweeps, metrics, and bounds.
- `framework/`: factorised encoders, section-domain backend, learned D0/D1 decoders, outer code/BP, training, and analysis.
- `tests/`: executable experiments, merge/plot scripts, smoke tests, and algebraic regression tests.
- `jobs/`: numbered HPC manifests, scripts, logs, checkpoints, and returned outputs.
- `results/`: detailed result ledgers and generated local outputs.
- `docs/reports/`: four supervisor-facing chronological LaTeX reports and their verified PDFs.

## Setup

```bash
uv sync
```

## Core Verification

```bash
uv run python -m compileall src framework tests
uv run python -m tests.framework_sectioned_refactor_test
uv run python -m tests.framework_sectioned_energy_test
uv run python -m tests.single_test --decoders NNOMP SIC BlockMAP \
  --n 32 --d 8 --num-blocks 4 --num-codewords 16 \
  --num-devices-active 4 --num-antennas 2 --esn0-db 5
```

Generated outputs under `jobs/`, `results/`, and most of `docs/` are intentionally ignored by git. Do not interpret a
submitted job as a result: inspect summaries, logs, completion, numerical diagnostics, and merged plots first.
