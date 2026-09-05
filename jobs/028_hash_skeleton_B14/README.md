# Job 028: B=14 Hash-Skeleton Certification

This batch asks one controlled question: does replacing an arbitrary size-`T` sparse support by exactly one resource in
each of `T` disjoint tables damage the useful codebook geometry or learned-decoder PUPE at a scale where the full
`n x 2^B` matrix is still available for certification?

The fixed operating point is `B=14,n=256`, with the jobs `021/022` load rule (`K=7..22` for training and
`K in {7,15,22,26}` for evaluation), the same `{-4,0,4,8,12}` dB grid, D0/D1, and two seeds. The manifest contains:

- dense reference (`T=256`);
- iid arbitrary sparse support;
- balanced random table assignment, isolating the one-per-table constraint;
- random full-rank binary linear hashes;
- binary linear hashes selected offline from 128 candidates by their exact XOR-difference collision spectrum.

The four sparse families are evaluated at `T in {16,32}`. Thus `(T,R,r)=(16,16,4)` or `(32,8,3)`, where `R=n/T`
and `r=log2 R`. At fixed `T` and seed, every sparse family receives the same Gaussian amplitude array before exact
column normalisation. Codebook structure, training data, and evaluation data use separate deterministic streams.

This is a small-`B` certification experiment. The affine support rule is stored compactly as
`A in GF(2)^(T x r x B), b in GF(2)^(T x r)`, but the present D0/D1 interface still materialises the generated
`Phi in R^(n x 2^B)`. A favourable result establishes that the support restriction is not damaging at this operating
point; it does not yet establish a scalable receiver or a fully procedural amplitude generator.

Local verification:

```bash
bash jobs/028_hash_skeleton_B14/local_smoke.sh
bash jobs/028_hash_skeleton_B14/local_mini.sh
```

Submit all 36 tasks from the HPC repository root:

```bash
qsub jobs/028_hash_skeleton_B14/028_hash_skeleton_B14.sh
```

If a full array is too large for the current queue, the two support sizes and dense controls occupy indices `1-16`,
`17-32`, and `33-36`, respectively:

```bash
qsub -J 1-16 jobs/028_hash_skeleton_B14/028_hash_skeleton_B14.sh
qsub -J 17-32 jobs/028_hash_skeleton_B14/028_hash_skeleton_B14.sh
qsub -J 33-36 jobs/028_hash_skeleton_B14/028_hash_skeleton_B14.sh
```

After pulling results, enforce manifest completeness and create both performance and geometry plots:

```bash
uv run python -m tests.framework_hash_skeleton_merge \
  --result-root jobs/028_hash_skeleton_B14/results \
  --manifest jobs/028_hash_skeleton_B14/manifest.tsv \
  --out-dir jobs/028_hash_skeleton_B14/results/merged
```

Primary decision: compare each structured family with iid sparse at the same `T`, decoder, and seed. Dense is context,
not a density-matched control. Do not infer superiority from the local mini suite or from geometry alone.
