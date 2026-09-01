# Framework Experiment Bank

This is the concise operational ledger. The chronological explanation is in [`docs/README.md`](README.md), the current
neutral handoff is in [`docs/CURRENT_STATE.md`](CURRENT_STATE.md), and full numerical evidence is in
[`results/04_results.md`](../results/04_results.md).

## Explicit `L=1` contract (`021--022`)

- Message space: `M=2^B=Q*V`; every `(q,v)` pair is legal, so `U=T=I`.
- Product encoder: `Phi=[R_1 C|...|R_Q C]`, `d=n`, fixed real diagonal sign masks `R_q`, and fixed or learned `C`.
- Compact operators: diagonal masks are stored as `(Q,n)`. Normal channel and decoder operations call implicit `matvec/rmatvec`; they do not construct `Phi` or an `(M,n,d)` intermediate.
- Controls: fixed/learned dense global codebooks, fixed legacy ODMA placement, and a fixed sparse global codebook without product sharing.
- Antennas: one by default. The framework still supports multiple known-channel antennas, but they are irrelevant to this first factorisation test.
- D0: unrolled exact implicit gradient steps, Bernoulli posterior-inspired scalar calibration, per-layer parameters, differentiable mass normalisation to `K_a`, and hard integer count projection only at output.
- D1: D0 plus latent-slot attention within each pattern `q`, across patterns for each shared local index `v`, and globally. It has no flattened-index convolution or adjacency assumption. Its correction head is zero-initialised, so it starts with exactly the D0 forward map; training tests whether nonlocal factor context helps.
- Conservative control: `UnrolledNonnegativeISTA` remains available. It is an exact-gradient nonnegative ISTA-style baseline, not classical LISTA with learned dense matrices.
- Loss: balanced support BCE at all layers, Smooth-L1 count loss for multiplicities, and D1 analysis/inverse symmetry. There is no data-consistency loss, count-sum loss, ranking loss, or L1 sparsity penalty.
- Tiny certification: `exact_count_ml` enumerates count multisets, includes message collisions, uses implicit encoding, and refuses problems above a hypothesis limit.

## Training and evaluation grid

The default training load is sampled once per batch from

`K_min=round(0.4*n/B)` through `K_max=round(1.2*n/B)` (inclusive).

| B | n | train K | fixed evaluation K | extrapolation K |
|---:|---:|---:|---:|---:|
| 12 | 128 | 4-13 | 4, 9, 13 | 15 |
| 12 | 256 | 9-26 | 9, 17, 26 | 30 |
| 14 | 128 | 4-11 | 4, 7, 11 | 13 |
| 14 | 256 | 7-22 | 7, 15, 22 | 26 |

Training samples one `Eb/N0` value per batch uniformly in dB over `[-4,12]`. Evaluation uses fixed `[-4,0,4,8,12]` dB points. Every result row records empirical and theoretical probability of any message collision and expected users per pattern `K_a/Q`.

Learned encoders are trained first for 40 epochs through the same D0 surrogate. The encoder is then frozen before the selected decoder phase. D0 receives 80 decoder epochs because the local convergence control showed it was still improving after 20; D1 receives 20 because it plateaued around epochs 10-15. Paired D0/D1 jobs use the same encoder seed, preventing decoder architecture from silently changing the learned encoding under comparison.

The pilot array has 20 rows at `(B,n)=(12,128)`: ten encoders times D0/D1. The scale array has 60 rows for `(12,256)`,
`(14,128)`, and `(14,256)`. Both arrays are complete and inspected.

## Receiver knowledge audit

| Quantity | Used by receiver? | Status in this bank |
|---|---:|---|
| Full factorised encoder (`C`, `R`, `U`, `T`) | yes | Known system design; accessed by forward/adjoint operations. |
| Realised `K_a` | yes | Oracle-known. It sets the Bernoulli prior, mass normalisation, and final count total. Unknown-`K_a` decoding is not tested here. |
| AWGN variance / selected `Eb/N0` | yes | Oracle-known scalar used for calibration. A noise-blind ablation is not yet included. |
| Channel coefficient `H` | yes | Known. With the default single constant antenna, `H=1`, so this is trivial. |
| Message prior | yes | Uniform i.i.d. user messages, yielding `rho=K_a/M`. |
| Active message identities and multiplicities | no | These are the decoder targets and are visible only to the training loss/evaluator. |
| Noise realisation | no | Never exposed. |
| Collision locations | no | Not exposed; multiplicities are learned through the count target. |
| Per-pattern occupancy | no | D1 computes soft occupancy from its current iterate; true occupancy is not supplied. |

These `L=1` jobs compare encoders and decoder expressiveness under known `K_a`, known noise level, known codebook, and low collision. They do not establish robustness to unknown activity/noise, learned `R`, restricted legal pairs, or paper-faithful CCS. The later section-domain jobs test `L>1` separately. Existing CCS-AMP jobs remain the external matched-decoder benchmark and should not be relabelled as a generic-framework control.

## Verification checklist

- [x] Exact implicit forward versus materialised `Phi` for real sign and complex phase operators.
- [x] Numerical adjoint identity and selected-column materialisation.
- [x] Legacy ODMA algebraic equivalence after implicit operator changes.
- [x] Variable `K_a` per batch, single antenna, D0/D1 forward/backward, and collided count target.
- [x] Tiny noiseless collision-aware exact ML recovery.
- [x] PBS shell syntax and 20/60 manifest row counts.
- [x] Tiny end-to-end learned-encoder plus D1 smoke run.
- [x] Pilot job 021 completed and inspected; 20/20 summaries and checkpoints returned.
- [x] Scale job 022 completed and inspected; 60/60 summaries and checkpoints returned.

## Scalable section-domain banks

Jobs `023` and `024` use the no-global-axis backend with procedural outer encoding, exact-unit-energy section banks,
Binomial local D0, modular BP, and valid-path list recovery.

| Job | Scale | Purpose | Status |
|---|---|---|---|
| `023_sectioned_outer_pilot` | `B=16,n=512`, mainly `K=2` plus selected `K=4` controls | Compare outer graph, section alphabet, exact versus overlapping energy, fixed versus learned banks, and random-sparse graph degree. | Partial: 16/26 summaries and checkpoints returned. Core triadic controls completed; ten broad graph-sweep rows are missing. |
| `024_sectioned_B128_scale` | `B=128,J=16,n=38400`, `K=10,25,50,75` | Establish whether the complete M-free D0/BP/association path executes and learns at the target payload scale. | Complete: 8/8. M-free execution and exact energy pass, but D0 saturates and PUPE is 1.0 throughout. |

These are not directly attributable against `021/022`: they change the message representation, physical codebook class,
prior, association layer, outer constraints, and operating scale together.

Returned `023` conclusions:

- learned exact-energy `J=4` triadic improves mean D0 PUPE from `0.6833` (fixed) to `0.5500` at `K=2`;
- sampled overlapping energy is worse and violates exact deployable energy, reaching approximately `1.29` sampled energy;
- BP is worse than D0 in eight of nine completed configuration aggregates;
- the outer-loss ramp produces pre-clipping gradient norms as large as `10^15`, so BP and fine graph rankings are not
  considered well-optimised;
- the missing graph rows should not be rerun as the same broad sweep.

Returned `024` conclusion: `B=128` runs without a global message axis and preserves unit energy to `4.8e-7`, but its
support logits sit at the lower clamp. Support loss remains approximately `15`, initial gradients are effectively zero,
and all PUPE values are `1.0`. Fix local-evidence/denoiser calibration before any unchanged scale rerun.

## Small-B causal bridge banks

Jobs `025` and `026` restore a matched `B=12,n=256` comparison before interpreting `023/024`.

### `025_sectioned_L1_bridge`

- Six array rows: dense, 25%-sparse global, and four-placement ODMA codebooks, each with two seeds.
- Same `K=9--26` training range, `K=9,17,26,30` evaluation loads, `[-4,0,4,8,12]` dB grid, and 8,000 D0 steps as
  the corresponding job-`022` D0 bank.
- The physical codebook is unchanged. Global Bernoulli D0 and section Bernoulli-compatibility D0 share learned weights
  and observations; the run fails if their logits or estimates differ beyond numerical tolerance.
- A separately trained Binomial section D0 isolates the change from the old support prior to the collision-aware prior.
- Status: complete, 6/6. Compatibility logits/soft/hard outputs agree exactly. Binomial versus Bernoulli has maximum
  absolute PUPE difference below `0.0037`, so neither backend nor prior causes the later structured gap.

### `026_sectioned_Lgt1_bridge`

- Six array rows: fixed and learned exact-energy triadic encoders, plus the fixed identity/no-outer control, with two seeds.
- Every final encoder is frozen before decoder comparison. Learned rows first receive 4,000 projected local-D0 encoder
  steps; both decoder routes then receive 8,000 steps on matched random streams.
- The induced `Phi in R^(256 x 4096)` is materialised only inside this small-`B` certification. Its global D0 result
  measures the physical encoder/model-class effect without scalable association.
- The same observations are decoded by local Binomial D0 plus valid-path association, and then with BP evidence added.
  The recorded gaps therefore localise global-to-local inference/association and the incremental BP effect.
- The identity row demonstrates the unidentifiable no-outer association problem. It has fewer physical sections than the
  redundant triadic construction, so it is a structural necessity control rather than a pure one-factor BP ablation.
- Status: complete, 6/6. Signal equivalence and exact energy pass within `6.6e-7`. Identity remains near PUPE `0.93--0.99`
  at high SNR; learned triadic reaches `0.414` at `K=9` but degrades to `0.947` by `K=30`. BP is worse than local
  association in every high-SNR cell.

Current interpretation: `025` clears the refactor and prior. `026` shows that outer constraints are necessary and useful,
but the present triadic code, locally trained encoder, association route, and marginal BP do not preserve the dense or
sparse-global landscape as local occupancy grows. This is a practical matched-system result, not an optimality theorem,
because the induced-global comparator is D0 rather than exact MAP/ML.

## Sparse-global density frontier

### `027_sparse_density_frontier`

- Status: returned but not strictly complete. All 72 array logs are present; 68 summaries/checkpoints are complete. Four
  full-density seed-2702 rows (`3,4,67,68`) completed evaluation and then failed in posthoc diagnostics because one
  float32 Gaussian draw was exactly zero. The generator invariant is fixed; rerun those four indices before treating the
  strict aggregate as final.
- Scope: retain the job-`022` `B=12,n=256`, load, SNR, unit-energy, D0/D1, and training contract while changing only
  sparse-global integer support `s` over `256,192,128,96,64,48,32,24,16,12,8,6,4,3,2,1`.
- Controls: separately trained dense at `p=1` and four-mask ODMA at `p=0.25`, where `p=s/n`.
- Replication: two codebook/training seeds. Sparse codebooks are nested across `s` within a seed while retaining the same
  marginal random-support/Gaussian-amplitude law at every support size.
- Pairing: codebook/init, training-data, and evaluation-data streams have separate seeds. Within one replication seed,
  all densities, both controls, D0, and D1 use identical held-out messages and noise.
- Precision: eight evaluation batches rather than the four used by jobs `021/022`, giving 64 realisations per
  `(K,E_b/N_0)` cell per seed.
- Diagnostics: exact energy and support repetition, row load/energy balance, 30,000 sampled column pairs, and 256 sampled
  active sets. No `M x M` Gram matrix is formed.
- Primary output: high-SNR mean PUPE versus nonzero fraction on a reversed log axis. Zero is excluded; `s=1`, or
  `p=1/256`, is the physical endpoint.

Audit of the 68 strict artifacts:

- all contain the expected 80 D0 or 20 D1 epochs, 20 learned and 20 matched-filter evaluation cells, finite values, and
  checkpoints;
- all 34 returned D0/D1 pairs have bit-identical encoder states and identical matched-filter evaluation streams;
- nested supports and shared signs are exact within both seeds over every returned density;
- maximum codeword-energy deviation is `9.54e-7`;
- the incomplete merger and plots are reproducible only with `--allow-incomplete` until the four rows are rerun.

High-SNR mean PUPE (8/12 dB, all four loads) shows:

| Family/support | Nonzero fraction | D0 | D1 | Evidence |
|---|---:|---:|---:|---|
| dense | 1 | 0.2935 | 0.2239 | two-seed context; second seed from 4-decimal completed logs |
| sparse global, `s=256` | 1 | 0.2979 | 0.2226 | two-seed context; second seed from 4-decimal completed logs |
| sparse global, `s=64` | 1/4 | 0.2932 | 0.2256 | two complete summaries |
| sparse global, `s=48` | 3/16 | 0.2975 | 0.2188 | two complete summaries |
| sparse global, `s=32` | 1/8 | 0.3051 | 0.2276 | two complete summaries |
| sparse global, `s=16` | 1/16 | 0.3312 | 0.2323 | two complete summaries |
| sparse global, `s=8` | 1/32 | 0.3984 | 0.2813 | two complete summaries |
| sparse global, `s=4` | 1/64 | 0.5171 | 0.3994 | two complete summaries |
| ODMA, four masks | 1/4 | 0.5363 | 0.4884 | two complete summaries |

The useful region is broad, not a single sharp support. D0 is near-flat through about `s=48`; D1 remains near dense at
`s=16`, but its two-seed errors are larger and its 20-epoch training curves are still descending. At the same `p=1/4`,
random sparse-global supports beat ODMA by `0.243` (D0) and `0.263` (D1). Even `s=4` nominally beats ODMA. This extends
the original result: density is not enough to explain the ODMA penalty; support diversity/geometry matters strongly.

The degradation begins well before the raw mask count is exhausted. Through `s=4`, both seeds retain 4,096 distinct
support masks and signed patterns. From `s=32` to `s=8`, sampled correlation q99.9 rises from `0.256` to `0.479`, while
the `K=30` occupied-row fraction falls from `0.982` to `0.613`. At `s=1`, only 256 supports/512 real signed directions
exist, signed-pattern repetition is about 87.5%, correlation q99.9 is 1, and PUPE collapses. A procedural generator must
therefore target recovery geometry and searchability, not just enough combinatorial labels.

Do not repeat all 16 supports over all old geometries. After repairing the four controls, confirm `s=8,16,32,64` with
more replication and then at one larger explicit payload (preferably `B=14,n=256`). This distinguishes the transition
and its scaling before committing to a procedural support family.
