# 033 — Matched D2: published codebooks, hash supports and generated amplitudes

Two questions remain after 032: does the explicit advantage persist when the published codebooks also use D2,
and does our compressed encoder retain that advantage under the same receiver? D2 scores the full alphabet;
there is no candidate-list confound and no claim of scalable decoding.

## Returned evidence — 6 October 2026

All **66/66** rows pass the strict audit and reuse all 42 required 032 references. Same-codebook ODMA–polar D2 closes
most of the native receiver gap; dense D2 retains an edge. Coordinate J8 remains competitive with the unrestricted
hash parent. Some learned calibrations are unstable, so post-hoc diagnostics do not replace the production ranking.
See the [experiment ledger](../../docs/EXPERIMENT_BANK.md) for conclusions and the local
[full audit](results/analysis/review_2026-10-06/REPORT.md) for numerical tables. The contracts and replay commands below
are retained for reproduction; this is not a request to resubmit the bank.

## Array

| Rows | Encoder | Training | What this isolates |
|---|---|---|---|
| 1–18 | ODMA–polar, CCS-AMP, exact-energy block CCS | D2 only | Receiver versus codebook limitations |
| 19–66 | Selected affine hash, unrestricted amplitudes (`J=B`), or coordinate banks `J=4,8,10` | Fixed amplitudes or joint amplitudes + D2 | Support restriction, then amplitude compression |

All use `B=12/14,n=256,T=32` where applicable, seeds 3201/3202/3203, and the **unchanged 032 contract**:
8 layers, 200 epochs maximum, patience 10, best-pair restoration, 100 batches/epoch of 8, 16 validation batches.
Train distinct messages at `K=7,15,22,26`, physical `Eb/N0=-4..10 dB`; evaluate both distinct and iid messages,
`-4:2:10 dB`, 256 frames/cell. Decoder/encoder learning rates remain 0.002/0.001.

Published codebooks retain the original pilot selections and ODMA cap30 override; none is retrained or retuned.
The existing 24 dense/sparse D2 rows and 18 native rows are reused, not rerun. Dynamic CS remains excluded.

For hash rows, `h_t(w)=A_t w+b_t`, `g_t(w)=V[t,P_t w]`, and `alpha(w)=g(w)/||g(w)||`.
All variants with a given B/seed share the same selected support (128 candidates, geometry-only selection).
`A,b,P_t` stay fixed. Compact banks are centered/unit-RMS **at initialization only**; their magnitudes remain free.
`J=B` uses raw Gaussian initialization and is exactly the arbitrary stored-amplitude model on those supports,
with `T*2^B` parameters. Compact banks have `T*2^J` parameters: 512/8192/32768 at J=4/8/10.
Every generated codeword has unit energy. Fixed/joint pairs start from identical physical codebooks.

Both hash endpoints use raw-bank parameters and forward normalization, so their amplitude-learning comparison has
the same parameterization. Historical 032 dense/sparse joint rows instead project explicit columns after updates:
joint gaps against those rows also reflect optimization, not just model class. Fixed-codebook comparisons are cleaner.

## Interpretation

1. Published D2 versus its native receiver measures the receiver gap on identical waveforms.
2. Sparse-global D2 versus hash-full D2 tests the hash-support restriction under a stronger receiver.
3. Hash-full versus coordinate J tests amplitude compression; compare fixed and joint routes separately.
4. Dense D2 remains the primary reference. Report per-load/SNR gaps and all three seeds, not only one pooled score.

Job 031 has returned 72/72 rows. Balanced scalar banks were close to the parent under D0/D1, but that job used
different data seeds, iid training, load distribution, training budgets and a legacy noise convention. Its results
cannot be inserted as matched 032 curves. The new batch tests whether that encouraging result survives D2.
Neither small-B equality nor isolated-message injectivity establishes B=100 multiuser inversion.

D0/D1 code and historical results are unchanged. The diagnosed D0 count-to-list shortfall remains a labelled
readout limitation; affected high-SNR tails cannot establish an encoder defect or a clean D0-versus-D2 gap.
Skipping their re-evaluation does not invalidate the D2/native comparisons above.

## Run and merge

Do not update an HPC checkout while 032 is running: its source fingerprints include benchmark files.
Wait for it to finish or use a separate checkout, with the same completed pilot `selected.json` available there.
From the repository root with the environment synchronized:

```bash
qsub jobs/033_matched_d2/033_matched_d2.sh
```

Optional split (instead of the whole array): `qsub -J 1-18 .../033_matched_d2.sh` for published controls and
`qsub -J 19-66 .../033_matched_d2.sh` for hash rows. Never submit both the full array and these splits.

```bash
uv run python jobs/033_matched_d2/merge.py
```

The merger checks complete grids, unchanged selected codebooks, paired support maps and per-codeword energy.
It reuses the 42 required 032 rows without waiting for D4, and produces separate paper and amplitude plots for
each B/sampling mode. It accepts only the two explicitly audited source-file hash transitions for this extension;
other changes require a new audit, not a silent override. Original results are never modified. Plots include
Polyanskiy's achievable-error upper bound, matched to B,n,K and message sampling. It is not a converse or a
performance floor. Nominal-energy CCS is labelled as not satisfying the bound's per-codeword energy constraint.

Local checks:

```bash
uv run python -m tests.matched_d2_test
uv run python -m tests.matched_d2_learning --out-dir jobs/033_matched_d2/results_local_smoke
uv run python -m tests.matched_d2_learning --mini --out-dir jobs/033_matched_d2/results_local_mini
```

Output directories must be new. Mini/smoke overrides are execution/learning checks, never production results.

Verified locally: nine B6 smoke paths; four B8 minis (40 epochs). Validation loss falls from 0.27425 to 0.16726/0.14457
for fixed/joint unrestricted amplitudes, and from 0.20167 to 0.13927/0.12912 for fixed/joint coordinate J4.
Unit energy and best-state restoration pass. Two reduced-budget B14,n256 joint executions also pass (unrestricted
and J4), including the production 128-candidate support search. All five B14 reference families reconstruct the
original seed-3201 matrix fingerprints exactly. These checks do not rank the production constructions.
