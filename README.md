# ODMA-URA

Research testbed for decoding **unsourced random access (URA)** signals with **on-off division multiple access (ODMA)** structure.

The current model uses a shared codebook of unit-norm message codewords. Each message is assigned to an ODMA block, each block embeds a length-`d` codeword into a sparse subset of `n` resources, and the receiver observes a noisy multi-antenna superposition. The decoding target is the global message-count vector: which messages were sent, and how many times, without recovering device identities.

The main research question is whether a decoder that uses both structures jointly can outperform generic sparse-recovery baselines such as OMP/SIC or standard per-block variants.

## Repository Layout

- `src/scenario.py` builds one reproducible ODMA+URA trial.
- `src/decoders/` contains comparable decoder implementations registered in `src/decoders/registry.py`.
- `src/sweep.py`, `src/cache.py`, and `src/plotting.py` run cached experiments and generate plots.
- `tests/single_test.py` runs a single scenario and writes plots/summary files.
- `tests/sweep_test.py` runs or replots parameter sweeps.

Generated outputs are written under `results/`, which is ignored by git.

## Setup

This repo is configured for `uv`:

```bash
uv sync
```

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Run

Single scenario:

```bash
uv run python -m tests.single_test --decoders Graph-BP ADMM-Poisson ADMM-Multinom Residual-MAP NNOMP \
  --num-devices-active 20 --esn0-db 0 --seed 42
```

Sweep:

```bash
uv run python -m tests.sweep_test --sweeps K SNR \
  --decoders ADMM-Poisson ADMM-Multinom Residual-MAP NNOMP --num-seeds 3
```

Common parameters include `--n`, `--d`, `--num-blocks`, `--num-codewords`, `--num-devices-active`, `--num-antennas`, and `--esn0-db`.

## Quick Check

```bash
uv run python -m compileall src tests
uv run python -m tests.single_test --decoders NNOMP SIC BlockMAP \
  --n 32 --d 8 --num-blocks 4 --num-codewords 16 \
  --num-devices-active 4 --num-antennas 2 --esn0-db 5
```

## Decoders

The registry currently includes structure-aware candidates (`Graph-BP`, `ADMM-Poisson`, `ADMM-Multinom`, `Residual-MAP`, `BlockMAP`), global sparse-recovery baselines (`NNOMP`, `SIC`), AMP/VAMP variants, and oracle LMMSE diagnostics. Add new decoders by implementing `run(scenario, **params) -> (counts, meta)` under `src/decoders/` and registering it in `src/decoders/registry.py`.
