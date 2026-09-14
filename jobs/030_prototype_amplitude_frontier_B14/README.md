# Job 030: compact amplitude frontier at B=14

The selected affine hash support is held fixed while amplitude freedom is varied through
`J = 0, 2, 4, 8, 10, 12, 14`. Each `J` has a fixed-prototype run (decoder trained) and a jointly learned run
(`V` and decoder trained); `A`, `b`, and `P_J` never train. Runs are paired by support, projection,
prototype initialisation, data streams, D0/D1, and seed.

`J=14` has the same unrestricted per-message amplitude model class as the previous hash encoder.
The extra equal (`J=0`) and Rademacher (`J=14`) fixed controls distinguish amplitude sharing from
amplitude distribution. Rows 1--48 are the completed base sweep; rows 49--64 append the missing
Gaussian `J=10,12` cases without changing any prior array index. All runs use at most 120 epochs and
restore the best validation checkpoint; training stops after five consecutive epochs without lower
validation loss.

Local checks:

```bash
bash jobs/030_prototype_amplitude_frontier_B14/local_smoke.sh
bash jobs/030_prototype_amplitude_frontier_B14/local_mini.sh
```

Submit:

```bash
qsub jobs/030_prototype_amplitude_frontier_B14/030_prototype_amplitude_frontier_B14.sh
```

Submit only the `J=10,12` extension:

```bash
qsub -J 49-64 jobs/030_prototype_amplitude_frontier_B14/030_prototype_amplitude_frontier_B14.sh
```

After all rows return:

```bash
uv run python -m tests.framework_prototype_amplitude_merge \
  --results-root jobs/030_prototype_amplitude_frontier_B14/results \
  --manifest jobs/030_prototype_amplitude_frontier_B14/manifest.tsv \
  --out-dir jobs/030_prototype_amplitude_frontier_B14/results/merged
```

This job measures amplitude model-class loss under the current exact small-B D0/D1 receiver. It does
not test scalable inversion; the saved generator is compact, but D0/D1 still allocate length `2^B` state.
