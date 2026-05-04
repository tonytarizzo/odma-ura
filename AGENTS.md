# Agent Instructions

This repository is a research testbed for ODMA-aware unsourced random access decoding. Treat correctness of the mathematical/modeling assumptions as more important than making a run appear to work.

## Working Style

- Keep implementations simple, concise, and robust.
- Avoid bloat, broad abstractions, and unnecessary fallback paths.
- Add fallback logic only when a real project-relevant failure mode justifies it.
- Prefer explicit assumptions over hidden convenience behavior.
- Do not paper over numerical/model issues; surface them clearly.
- Preserve the experiment contract: decoders should be comparable through `run(scenario, **params) -> (counts, meta)`.
- When proposing alternatives, prioritize low-complexity options first.

## Code Style

- Use the available horizontal space; do not break lines prematurely.
- Keep short function signatures, argparse arguments, and compact dict/list literals on one line when they remain readable.
- Split lines only when they would otherwise become genuinely long, roughly over 130 characters.
- Keep comments sparse and useful: explain modeling assumptions or non-obvious numerical choices, not mechanical code behavior.

## Research Priorities

- Prioritize exactness of the ODMA+URA scenario, decoder target, oracle assumptions, and metrics.
- Label oracle-aided baselines clearly.
- Keep generated plots/results out of git unless explicitly requested.
- Avoid turning the repo into production software infrastructure; this is a research framework.

## Environment And Commands

- Use `uv` for environment management.
- Use `uv run python ...` for Python commands once `uv.lock` exists.
- If `uv` is unavailable in the current shell, say so once and use the best available local environment for verification.
- If a run fails twice due to environment/runtime issues, stop retrying and report the blocker briefly.
- Useful checks:
  - `uv run python -m compileall src tests`
  - `uv run python -m tests.single_test --decoders NNOMP SIC BlockMAP --n 32 --d 8 --num-blocks 4 --num-codewords 16 --num-devices-active 4 --num-antennas 2 --esn0-db 5`

## Git

- The primary branch is `main`.
- Do not assume `master`.
