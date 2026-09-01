# Job 027: Sparse-Global Density Frontier

This is a controlled continuation of jobs `021/022`, not a sectioned/outer-code experiment. It fixes
`B=12,n=256`, the old load and SNR contract, D0/D1, and the exact unit-energy sparse-global family. Only the integer
codeword support size changes:

```text
s = 256,192,128,96,64,48,32,24,16,12,8,6,4,3,2,1
p = s/256
```

Two seeds are coupled across `s`: within a seed, each message uses one random ordering of the 256 resources and one
Gaussian amplitude vector, so reducing `s` takes a nested prefix before renormalisation. Each `s` still has the same
marginal distribution as the earlier independently supported sparse-global control. Dense `p=1` and four-mask ODMA
`p=0.25` controls are trained separately. Codebook/init, training-data, and evaluation-data seeds are separated; all
families and both decoders within one replication seed use identical held-out messages and noise. The full manifest has
72 rows.

The primary outcome is mean PUPE over 8 and 12 dB and all four evaluated loads. Geometry diagnostics record exact support
mask repetition, row load/energy balance, 30,000 sampled column correlations, active-set row occupancy, and unit-energy
error. No Gram matrix is materialised.

Run the short local pipeline check with:

```bash
bash jobs/027_sparse_density_frontier/local_smoke.sh
```

Submit on HPC from the repository root with:

```bash
qsub jobs/027_sparse_density_frontier/027_sparse_density_frontier.sh
```

After retrieving the complete results, merge and plot with:

```bash
uv run python -m tests.framework_sparsity_sweep_merge \
  --result-root jobs/027_sparse_density_frontier/results \
  --manifest jobs/027_sparse_density_frontier/manifest.tsv \
  --out-dir jobs/027_sparse_density_frontier/results/merged
```

The x-axis is the nonzero fraction `p=s/n`, reversed on a base-two log scale. Zero is deliberately absent: it cannot be
shown on a log axis and a zero-support codeword cannot satisfy unit energy. The smallest physical point is `p=1/n`.

## Returned-artifact audit (31 August 2026)

All 72 task logs were returned. Sixty-eight tasks wrote both `summary.json` and `checkpoint.pt`; array indices
`3,4,67,68` completed evaluation and then failed in the sparsity diagnostic. Seed 2702 generated one exactly zero
float32 Gaussian entry in each full-density construction, so one column had numerical support 255 while all others had
support 256. Gaussian initialisation now resamples exact zeros to preserve the intended exact-support contract.

After pushing/pulling that fix, rerun only the failed indices:

```bash
qsub -J 3-4,67-68 jobs/027_sparse_density_frontier/027_sparse_density_frontier.sh
```

Until those artifacts return, the strict merger correctly refuses the manifest. A provisional audit plot can be made
with `--allow-incomplete`; do not relabel it as a complete 72-run aggregate.
