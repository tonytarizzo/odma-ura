# 032 — Published receivers versus the explicit framework

**Purpose:** test the motivation for compressing the explicit encoder before adding more generator restrictions.
An explicit codebook with a trained receiver is not assumed optimal. A poor small-B adaptation is not evidence that
the published large-B scheme is poor.

## Current decision — 30 September 2026

The **80/80 pilot and 24/36 native rows** returned and passed provenance/grid/metric checks. The remaining 12 native
rows are all dynamic CS, with no completed SNR cell in the pulled snapshot. No main-comparison results have returned.
The pilot selection is ready, but **run `032_checks.sh` next, not the 192-row comparison yet**.

Native mean-PUPE 5% crossing brackets are ODMA K50 `(0,0.25]`, K100 `(0.5,0.75]`; CCS K50 `(2,2.25]`, K100 `(2.5,3]` dB.
These are grid brackets, not confidence intervals. K50 agrees reasonably with the prior paper readings; K100 needs review.
At ODMA K100,0.5 dB, 66/128 frames were still recovering messages when the ten-round cap stopped them. Two selected
difficult frames improved from 51%/48% PUPE to 1%/2% when allowed to finish at round 18. Eight unselected replay frames
improved from mean 3.50% to 2.63%. These are diagnostic replays, not a corrected aggregate curve.

The separate **20-row checks array** leaves the original three manifests and completed outputs unchanged:

| Rows | Check | Budget and purpose |
|---|---|---|
| 1–4, 9–12 | ODMA K100: cap30 at 0.25/0.5/0.75 dB; cap60 at 0.5 dB | Two original seeds, 64 frames; isolate truncation and check cap30 |
| 5–8, 13–16 | CCS K100,2.5 dB: AMP40 with SIC fractions 0.5/0.7/0.9; AMP80 with 0.7 | Two original seeds, 16 frames; targeted receiver-budget diagnostic |
| 17–20 | Dynamic CS K50/100,1.5/2.5 dB, double-precision cached banks | Four frames each; measure native runtime/behaviour before a large rerun |

ODMA/CCS replays preserve the original batch size and message/noise sequence. Four-frame dynamic checks are runtime
diagnostics, not a reproduction curve or a paired subset of the original eight-frame batches. Analyze variants separately
with `analyse_checks.py`, not the ordinary seed-pooling merger. Pilot scores are tuning data; small-B error floors do not
yet establish explicit-codebook superiority. The matched D0/D1 stage is still needed to separate receiver and encoder losses.

### Dynamic-CS cost and the bounded optimization

The author's thesis, Chapter 9, p.132, identifies matrix multiplication as the dominant receiver cost and gives
`O(2^Bp N1 + max_l 2^(Bl/S) Nl Ka)` (iteration/slot factors suppressed). It explicitly states higher complexity than
its ODMA comparator; it does not report a runtime that validates our 60-hour jobs.
Our native banks are 3000×262144 and 9000×32768. Previously float32 caches were converted repeatedly for float64
matrix products. The checks opt into float64 caches (8.65 GB combined; the 8 GB limit is **per bank**) and construct
them in blocks to bound temporary memory. Matrix entries, equations, thresholds, precision of AMP state and candidate
policy are unchanged. A laptop 2048×8192,50-column forward/transpose microbenchmark improved from median 0.104 s
to 0.028 s (3.7×, five repetitions), with identical outputs. This is not a native end-to-end HPC speedup claim.

Only the new checks array enables atomic per-frame checkpoints and progress logs. Resubmitting its same row resumes
under identical source/configuration, or skips a completed row. Each cell's metrics are rebuilt without duplicate frames.
Old empty native outputs cannot be resumed retrospectively. Training resume is not added. Keep code unchanged during
active jobs; source fingerprints intentionally reject mixed implementations.

Verification: 32 baseline/bound/manifest/resume tests pass; all three follow-up families pass small-B row smoke checks.
The 28-path B6 smoke suite completes and all 24 learned paths reduce validation loss. These are execution checks, not
native-scale validation of the new settings. Native dynamic-CS full memory/runtime checks remain the purpose of rows 17–20.

## Comparisons

For each of ODMA–polar, dynamic CS, and CCS-AMP:

| Test | Encoder | Receiver | What the comparison isolates |
|---|---|---|---|
| 1 | Published construction, fixed | Its complete native chain | Achieved end-to-end baseline |
| 2 | Exactly the same waveforms | D0 / D1 | Receiver versus codebook limitations |
| 3 | Dense / independent sparse global, fixed | D0 / D1 | Geometry under the same receiver |
| 4 | Dense / independent sparse global, jointly learned | D0 / D1 | Gains available from encoder adaptation |

The reference rows are shared across papers, not rerun three times. All D0–D4 have both **fixed-encoder** and
**joint encoder/receiver** dense/sparse rows. In either mode the receiver is trained. Sparse supports stay fixed;
joint training learns the nonzero values, with exact unit-column projection after every update.
D3/D4 receive 1,024 non-oracle matched-filter candidates, never injected truth. This still scans all `2^B` messages.
Report recall, PUPE and total proposal/refinement runtime; D2 is the full-alphabet reference, not D3/D4.

## Three phases

- **Pilot: 80 rows.** `B=12/14,n=256`, seed 3299; 64 frames per cell at `K=7,15,26`, physical `Eb/N0=0,4,8 dB`.
  ODMA varies prefix 4/6/8, polar length 32/64, and paired CRC/list budgets 8/8, 12/32, 16/128. This avoids assuming
  that the native 16-bit CRC is best for a 12-bit payload. Dynamic CS varies prefix 6/8/10 and first-slot length 85/128.
  CCS varies 20/40 AMP iterations, SIC cancellation fractions 0.5/0.8, and original/non-DC Hadamard embedding.
  Both dense CCS and an exact-energy block-diagonal adaptation are included. Select one configuration per family/B
  by mean pilot PUPE; the held-out stage uses different construction/data seeds.
- **Comparison: 192 rows.** `B=12/14,n=256`, seeds 3201–3203; `K=7,15,22,26`; physical `Eb/N0=-4,-2,0,2,4,6,8,10 dB`;
  256 frames per cell, separately for distinct and iid messages. Sparse support size is 32 (12.5% density).
  Training uses distinct messages, 8 layers, batches of 8, 100 batches/epoch, at most 200 epochs, patience 10,
  fixed validation and restoration of the best encoder/decoder pair, including epoch zero.
- **Native paper alignment: 36 rows.** Two seeds, `K=50/100`, seven SNRs per family, 64 frames per seed/SNR
  (128 per plotted point). Each task holds at most three SNRs. These are real native decodes, not training jobs.
  Dynamic CS's dense local matrix AMP is costly and caches several GB; all native jobs request 64 GB and 72 hours.
  Treat this as a first alignment check, not a verified reproduction or a high-load sweep. Large-B D0–D4 are not run.

| Native scheme | Paper dimensions/settings | Physical Eb/N0 grid (dB) | Paper 5% threshold at K=50 / 100 |
|---|---|---|---|
| ODMA–polar | B=100, n=30000; prefix 12/13; polar 512; CRC16; list128; one power level | −0.25, 0, 0.25, 0.5, 0.75, 1, 1.5 | ≈0.25 / 0.4 |
| Dynamic CS | B=100, n=30000; B1=15; 3×30 data bits; S=2; CRC5; lengths 3000+3×9000 | 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5 | ≈1.4 / 1.55 |
| CCS-AMP/BP | B=128, n=38400; 16 sections of 16 bits; original author embedding; two passes | 1, 1.5, 1.75, 2, 2.25, 2.5, 3 | ≈2.1 / 2.4 |

Thresholds are manual visual readings from ODMA Fig. 3, dynamic-CS thesis Fig. 9.3, and CCS Fig. 8, with a ±0.15 dB
reading guide, **not author-tabulated numbers or acceptance tolerances**. The merger plots them beside measured
PUPE and reports the tested-grid crossing of 5%; a crossing bracket is not a statistical confidence interval.

## Fairness and interpretation

- Real, synchronous, unfaded GMAC; all receivers know `K`, the code construction and the physical noise variance.
  Every parity, CRC, order/header bit and unused/padded coordinate is included inside the same total `n`.
- Physical `sigma²=1/(2 B 10^(Eb/N0/10))` for nominal energy one. The old real-valued framework's dB coordinate
  is **3.0103 dB higher for the same noise**; do not directly overlay its old numerical x-values.
- ODMA with one power level, single-stream dynamic CS, block-diagonal CCS and explicit references have unit
  energy per message. Dense CCS and multi-stream dynamic CS have **nominal**, message-dependent energy. We do not
  normalize their columns and silently break their native receiver model. Dense CCS has a separately labelled
  comparison panel without a peak-power bound overlay. Energy audits are saved.
- Both duplicate-tolerant PUPE and Polyanskiy's collision-as-error PUPE are saved. For `B=14,K=26`, the iid per-user
  duplicate probability is about 0.15%, and the frame probability about 2%. **Fragment/header** collisions can be
  much larger. In particular, small-B CCS has only 64/128 states per fragment; its local Bernoulli/list approximations
  are stressed even when full-message collisions are rare. Distinct-message training matches D2–D4's Bernoulli model;
  iid evaluation is a labelled stress test, not a change to the target metric.
- No oracle-support receiver is used. Candidate misses count against all transmitted users. Native CRC failures,
  false positives, SIC errors and search caps are recorded, not corrected using truth.
- Seed intervals are Student-t intervals over independent run means (three seeds in the main comparison); frame
  standard errors are also retained. With few seeds these intervals are only indicative. Zero observed errors
  and an overlapping interval are not proofs of optimality or non-inferiority.

## Audit findings and remaining decisions

- **Proposal bottleneck corrected, not solved.** At B14,n256,K26,10 dB, a 128-frame diagnostic found only
  82–83% recall with 256 matched-filter candidates; 1,024 raised it to 94%. The new default is 1,024, at higher
  quadratic D4 cost. D3/D4 still have a proposal-imposed PUPE floor; their known-K projection also assumes all
  active messages are in the list. Recall plots expose this mismatch.
- **Miss-aware loss.** Previously the candidate loss normalized over retained true messages only. Now D3/D4's
  loss/validation uses the full target: rejected messages have zero predicted count and clipped evidence −30.
  This prevents early stopping from favouring dropped difficult messages. Top-C membership remains discrete:
  gradients flow through the transmitted waveform and retained dictionary, not through membership changes.
  No extra differentiable proposer/ranking objective is silently added; these are not globally optimized searches.
- **Training budget is not convergence evidence.** 200/10 is a ceiling/stopping policy, not a guarantee. Keep loss,
  PUPE, best epoch and hit-cap diagnostics. Eight layers, learning rates, mixed-SNR/load training and loss choices
  remain specified hyperparameters; they do not certify optimal decoding. Historical jobs retain their old budgets.
- **Paper settings are only partly specified.** Native dimensions and encoder chains match the cited sources, but
  CRC polynomials, some detector thresholds and CCS's empirical SIC delta are not fully specified. Our choices are
  saved in each result. Check native alignment before interpreting a small-B win as beating a published scheme;
  a mismatch calls for receiver/tuning investigation, not a conclusion that its codebook is inferior.
- **Dynamic-CS stopping fixed.** Removing false header candidates changes the modulated AMP matrix even when
  no message passes CRC. The receiver now reruns after that change instead of prematurely stopping; a focused
  regression verifies this branch. It still stops when neither cancellation nor FA removal changes the problem.
- **Pilot selection optimizes native PUPE.** It does not independently optimize each published encoder for D0/D1.
  Small-B header/fragment alphabets and relative CRC overhead differ substantially from native dimensions.
- **Common matrix receiver, not paper-aware optimal inference.** Materialization preserves every waveform, but
  D0/D1 use the same identity-factor matrix adapter for all families; their inputs do not expose the native polar
  code or outer graph. Native receivers do exploit that structure. The comparison must not label D0/D1 as ML/MAP.
- **Power and uncertainty matter.** Dense CCS and native multi-stream dynamic CS use nominal energy, not our exact
  per-message sphere constraint. Native runs validate those original models; they are not strict peak-power controls.
  Three small-B/two native construction seeds give indicative intervals, not a proof of equivalence.

Interpretation order: native alignment first; then native versus D0/D1 on identical waveforms; then fixed/joint
explicit geometry under matched receivers. This can establish a useful empirical motivation, not prove that the
explicit family contains a reachable optimum or that a small-B ranking persists at B≈100.

## Source fidelity

- [ODMA–polar](https://doi.org/10.1109/LWC.2024.3359270): l1 pattern ranking, CRC-aided polar SCL, Gaussian TIN and SIC.
  Independent implementation using the standardized polar reliability order, not the full NR transport chain.
  CRC polynomial, detector slack and TIN calibration are exposed choices; the paper omits their exact values.
  Equal-power small-B runs are not the paper's high-load unequal-power construction.
- [Dynamic CS](https://doi.org/10.1109/LCOMM.2024.3403501): first-slot AMP, header-modulated matrix AMP, false-alarm
  rejection, CRC assembly and SIC, following the author's thesis Chapter 9. Small-B uses one stream; the native
  profile uses two, with first-slot LLR threshold 6 (a stated choice within the source's illustrated range, not a
  recovered optimum). The receiver models distinct substream indices, not their multiplicities. Thresholds, CRC
  polynomial and AMP budgets are stated choices, not recovered author settings.
- [CCS-AMP/BP](https://arxiv.org/abs/2010.04364): pinned author AMP, modular outer graph/BP and list extraction,
  plus the paper's two-pass SIC. The empirical delta schedule was not supplied. Logistic evaluation and FHT are
  numerically stabilized/vectorized; the transform is checked against the original encoder to machine precision.
  At B=12/14 the graph is a single triad with two information fragments, not the native 16-section B=128 graph.
  The original transform shares a constant column across sections and can oscillate on zero fragments; its
  existing non-DC embedding option is a separately labelled pilot adaptation, not an invisible repair.

None is labelled a verified reproduction of a published performance curve. That requires the native-scale checks,
then sufficient frames and matching the paper's settings/curve, not merely passing smoke tests.

## Local evidence (26–27 September 2026)

- Algebra/receiver and manifest/merger regression suites pass. Original and non-DC CCS generated signals match the
  pinned author's encoder at B=8/12/14 within `1.2e-16`. Tiny polar full-list decoding matches exhaustive ML ranking.
- All 80 production pilot configurations execute at B=12/14 in one-frame checks. All five learned decoders also
  pass a B=14 one-step training/evaluation check; these short checks do not measure convergence or final performance.
- The 22-path B=6,n=64 smoke and mini suites completed. All 18 learned paths lowered deterministic validation loss.
  Dense fixed D0: `1.791 -> 0.205`; dense fixed D1: `1.792 -> 0.106`; sparse joint D1: `1.689 -> 0.163`.
  D2/D3 already initialize well here and improve loss only slightly; D4 improves loss, not consistently PUPE.
  These losses are not comparable across the D0/D1 and D2–D4 objectives. Some D0 runs still reach the epoch cap.
- Native spot checks recover two noisy B=100 ODMA messages and two B=128 CCS messages at 8 dB. A B=100 dynamic-CS
  noiseless single-user check also passes (two AMP iterations, uncached banks, about 64 seconds). This is execution
  evidence only; the native manifest's K=50/100 and full budgets remain HPC validation work.
- Mini results are in `results_mini/`; loss/PUPE plots were rendered and inspected. No generated results are versioned.
- After the audit, all six joint D2/D3/D4 mini cases reduced validation loss with the 200/10 policy, stopping after
  21–42 epochs. Dense joint D4: `0.09245 -> 0.08377`; sparse joint D4: `0.12756 -> 0.12301`.
  PUPE did not improve consistently. `results_joint_mini/` is a separate execution/learning check, not a result ranking.
- The revised complete smoke suite passes all 28 paths (24 learned), and 25 baseline/bound/manifest regressions pass.
  A B14,n256 joint D4 forward/backward check with 1,024 candidates and two layers has finite encoder gradients;
  full eight-layer/batch-eight resource use is left to the 64-GB HPC jobs, not claimed laptop-tested.

## Bounds

The survey's Gaussian achievability reference and Polyanskiy's bound are the **same reference**, not independent curves.
Plots show **one** reference: all Gallager `p_t` terms plus `q_1`, following the original numerical recipe. The latter
integrates the shared-noise minimum over users; it is not an independent-user/normal approximation. This remains an
achievable-error upper bound, not a converse: `optimal PUPE <= bound` does not imply `decoder PUPE >= bound`.
Prepared analysis covers B6/12/14/100/128. At B6,n64,K1,0 dB the bound is about 0.930, while a 20,000-frame single-user
exact-ML diagnostic achieved 0.170; small-B looseness remains real. Paper Fig.1 visual alignment is within 0.084 dB;
grid/quadrature refinement changed tested thresholds by at most 0.060 dB. These checks are numerical, not a new theorem.
At B12,K22/26 iid the collision correction alone exceeds 5%, so this bound cannot certify that target.
Legacy `src/ura_bound.py` is unchanged and must not be relabelled as rigorous. The unused weak Fano helper was removed.
To regenerate in an empty output directory: `uv run python -m benchmarks.ura_bound_analysis --out results/032_bounds`.

## Commands

**Next submission**, after updating an idle/isolated HPC checkout (the CCS author dependency already exists):

```bash
git pull --ff-only origin main
module load miniforge/3
uv sync --python python
qsub jobs/032_published_baselines/032_checks.sh
```

Inspect returned checks, including partial sets:

```bash
uv run python jobs/032_published_baselines/analyse_checks.py --allow-incomplete
```

The comparison command below is **deferred until these checks are reviewed and receiver budgets finalized**.
Do not resubmit the original pilot/native arrays merely to run the checks. For a fresh installation only, prepare
the environment and the unvendored author dependency once:

```bash
uv sync
git clone https://github.com/vamsi128/CCS-AMP-Code.git .cache/CCS-AMP-Code
git -C .cache/CCS-AMP-Code checkout 92080d85408d5d19a123d1d61ba76ec6f15451a5
qsub jobs/032_published_baselines/032_pilot.sh
```

If that checkout already exists, skip the clone and verify its commit. After checks are reviewed, generate selection
from the already completed pilot and submit the main comparison:

```bash
uv run python -m benchmarks.ura_merge --manifest jobs/032_published_baselines/pilot.jsonl --results jobs/032_published_baselines/results/pilot --select-pilot
qsub jobs/032_published_baselines/032_comparison.sh
```

Independent native paper-alignment validation (recommended before making claims against the papers):

```bash
qsub jobs/032_published_baselines/032_native.sh
uv run python -m benchmarks.ura_merge --manifest jobs/032_published_baselines/native.jsonl --results jobs/032_published_baselines/results/native
```

Merge/plot the completed comparison (use `--allow-incomplete` only for an explicitly partial analysis):

```bash
uv run python -m benchmarks.ura_merge --manifest jobs/032_published_baselines/comparison.jsonl --results jobs/032_published_baselines/results/comparison
```

Local checks: `bash jobs/032_published_baselines/local_smoke.sh` and `bash jobs/032_published_baselines/local_mini.sh`.
Original-phase outputs must be empty/new; the runner refuses to overwrite or combine old results. The checks phase
alone supports resumable native frames: resubmit just the affected index, e.g. `qsub -J 17 jobs/032_published_baselines/032_checks.sh`.
Manifests are generated by `build_manifest.py`.
The merger checks declared parameters and default budgets, source versions, initial matrices and shared pilot
selection provenance; an under-budget or differently configured run cannot count as a completed comparison row.
The first 156 comparison row indices are preserved; joint D2–D4 rows are appended at 157–192. Budgets and candidate
losses changed, so do not mix previous comparison outputs with this revision. Native rows replace the old eight-frame
smoke manifest. Do not pull/edit source files during running jobs; the runner rejects mid-run source changes.
