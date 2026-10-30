# Experiment bank

Updated **30 October 2026**. This document maps research questions and algebra to executable scripts,
fixture contracts and retained results. [CURRENT_STATE.md](CURRENT_STATE.md) owns priorities;
Reports 7, 9, 10 and 11 own the mathematical derivations and their limitations.
Job manifests and the commands below own HPC row definitions. Historical exact audits and matched performance evidence below retain their original scope. Approximation
preparation is complete for the retained paths. All thirteen families now have shared-harness implementations;
pre-reset prototype scripts remain parked. Jobs035–037 have returned substantial results; the latest local audit below distinguishes complete,
partial, absent and source-guard-failed runs. Local availability does not establish live scheduler status.

## Reading the evidence

- **Algebra checks** verify finite identities/inequalities and exact endpoints, not an approximation's useful regime.
- **Structural audits** measure ranks, states or table sizes, not end-to-end runtime or physical PUPE.
- **Matched performance banks** evaluate native/learned receivers under their recorded contracts. Historical IID
  membership scores need the [metric correction](PUPE_METRIC.md); distinct matched results remain interpretable
  where evaluated support/budget is unchanged.

Truth and exact posteriors supply evaluation labels only unless explicitly marked **oracle/reference-only**.
Generated JSON stays outside git. Prior temporary outputs may no longer exist; commands regenerate the panels.

## Shared exact-reference fixtures

| Label | Fixed construction and source/observation contract |
|---|---|
| Tiny polar | B4=Bp2+Bc2, P4/V4/M16, Nc8/CRC4/n12, placement 3217, deterministic balanced energies. K3, IID and distinct. Exact count states: 816 IID / 560 distinct. |
| Tiny paired panel | Tiny polar; sigma²=.05,.3,1; observation seeds 0,1,2,3 with RNG 9100+seed. Truth and standardized noise are paired across variances: 24 observations. |
| Research banks | B12/B14, Bc6/V64/Nc32/CRC8/n256, P64/P256, seeds 3201–3203. Structural certificates/censuses only in the exact study; no full-bank exact posterior. |

Coverage alone does not establish balance feasibility: the prescribed solver must converge. No silent
support repair, fallback energies or decoder-dependent bank selection is permitted.

## Sum geometry, repair and affine inference: Reports 7–11

The [sum-geometry check](../tests/polar_sum_geometry_checks.py) follows the six-bit first-alias finding.
It is a finite mathematical/geometry audit, with no production decoder changes or PUPE simulations.

```bash
uv run python -m tests.polar_sum_geometry_checks \
  --output results/polar_sum_geometry_2026-10-29/checks.json
```

The output is generated and ignored by git. The script asserts the following independently of the
older Report-7 test; integer equalities/cover checks are distinct from floating distance checks.

| Question | Certificate / observation | Limit |
|---|---|---|
| What causes the first aliases? | Full 766,480-multiset K4 census equals the actual affine-class members: 392 pairs + 12 four-member classes; 52/651 planes deficient, 58 missing character flats, 464 pairwise equalities. | Complete only for this six-bit fixture and K4; K3 separately exhausts 45,760 states. |
| Is the threshold universal? | A four-bit, ten-row linear sign code has 136 unique pairs but a triple alias. CRC0 has 32 complementary pair aliases. No quadratic-complete character basis exists in the selected CRC8 generator. | General linear-code counterexample, not a second polar-family theorem. |
| How many new characters are needed? | All 1,623,160 six-subsets hit at most 55/58 flats; seven-row witness 8/13/19/52/60/61/63 hits all. Best three-row coverage is 40/58, achieved by 8/20/61. | Preserving old distinct rows avoids new aliases; deleting them needs another census. Cover discovery is exponential. |
| What does the three-row replacement change? | 144 ambiguous pairs/288 multisets remain; minimum Hamming distance stays 12. Known-pattern K2 minimum squared distances improve from 1/.801428 to 1.5/1.271657, uniform/balanced. | Same B12/seed3201/pattern0 placements and energies; receiver compatibility and full-system PUPE untested. |
| Does balance improve every noisy comparison? | All 2,162,160 K2 comparisons on that reference pattern: 982,704 distances increase, 1,179,456 decrease; minimum 1 → .801428. Direct Gaussian and grouped-character likelihoods agree. | A conditional counterexample, not a bank-wide distance or risk ranking. |
| Can overlap change the conclusion? | A distinct-support four-pattern rectangle aliases under uniform balance; another balanced allocation separates it. | Same-pattern positive-energy kernel invariance does not extend to mixed patterns. |
| Can inference keep identity without a full alphabet? | Affine-orbit signal equality, exact tails, slot-selection/factorial balance and a 100-bit nullspace control; sparse-Walsh bin identity with repeated counts and noise variance. | Local operations only; no discovery/mixing or complete scalable decoder guarantee. |
| When may near aliases be merged? | Twenty finite Gaussian groups check the conditional likelihood-range/TV bound. | Group discovery, radius certificate, group evidence and global count allocation remain separate costs. |

The character-distance identity's maximum discrepancy is below 8e-15. Runtime is a local diagnostic
cost, recorded in the JSON (about six seconds in the initial run), not a decoder timing claim. The
full pair-distance calculation materializes a 2080-by-2080 array; the four-user census materializes
all 766,480 states. Report 7 owns exact classification/source frequency, Report 8 owns distances and
modified signatures, Report 9 owns near-alias error, and Report 11 owns the local operation and costs.

## Common-D2 encoder comparison: Job038

[Manifest](../jobs/038_polar_signatures/manifest.jsonl), [runner](../jobs/038_polar_signatures/run_row.py),
[signature construction](../jobs/038_polar_signatures/experiment.py),
[archived controls](../jobs/038_polar_signatures/comparators.py), [merger](../jobs/038_polar_signatures/merge.py).
**30 October extension:** the primary comparison now uses **D2 for every encoder**, including the archived
learned dense/sparse and learned-energy controls. It supersedes the earlier 24-row, two-receiver plan.
Shared Job035–037 runtime sources and their compatibility contracts are unchanged.

| Encoder ID | Construction / training | Resources |
|---|---|---|
| `polar_reference_uniform` | Original CRC/polar characters, uniform energy; frozen encoder | 32 occupied coordinates |
| `polar_reference_balanced` | Original characters, deterministic KL-balanced energies | Same 32 positions |
| `polar_repair3_uniform` | Replace duplicate rows 17/21/30 by character keys 8/20/61 | Same positions and uniform energies |
| `polar_repair3_balanced` | Same repair, archived reference balanced energies | Same positions and energies |
| `dense_fixed`, `dense_joint` | Job032 fixed Gaussian / jointly learned dense encoder and D2 | 256 occupied coordinates |
| `sparse_fixed`, `sparse_joint` | Job032 fixed / jointly learned sparse encoder and D2 | 32 occupied coordinates |
| `polar_energy_joint` | Job034 learned positive symbol energies and D2 | Original characters and 32 positions |

These are stable experiment labels, not new framework families. Reference/repair contrasts isolate the
payload signature. Dense comparisons match channel length, B, K, power and D2, but **not occupied-support
size**; report that resource advantage for sparse constructions. CRC and placement are not optimized.
The seven-parity extension uses 39 symbols and remains outside this matched-resource comparison.
Native CRC/SCL is incompatible with the repair; no custom decoder is added. Existing DEC-26 remains
available via the one-frame `--probe`, outside the primary common-D2 ranking.

All nine arms use n256, B12/B14 (six payload bits), known K/noise, real AWGN with
`sigma²=1/(2 B 10^(EbN0/10))`, IID full messages and multiset count-PUPE. D2 has eight layers/16 learned
scalars and an explicit M-axis. Training remains distinct-message, K7/15/22/26, uniform SNR in [-4,10],
200-epoch cap, patience10, 100 batches of8, 16 validation batches, lr .002. Historical joint arms also
optimized encoder parameters at lr .001: this is a comparison of trained systems under a common receiver
architecture/protocol, not equal total optimization work or an optimal-decoder encoder ranking.
Twelve original D2 calibrations and thirty historical encoder+D2 checkpoints are reused; only the twelve
repaired calibrations are new. Report setup, original training and inference costs separately.

The five archived families are **reevaluated**, not merged from old IID membership metrics. Their stored
float32 matrices are promoted and explicitly column-normalized in float64 to satisfy the current strict
unit-power contract. The original error must be below 1e-5; actual rescaling and checkpoint/summary hashes
are recorded. D2 parameters are reused and its physical Gram statistics recomputed. This precision-level
projection is declared, not an exact historical replay. Learned-energy columns are first materialized
using their original float32 computation. Original archives remain untouched.

**54 rows = 2 B × 3 seeds × 9 encoders**, at K7/26, Eb/N0 4/10 dB, 128 frames/cell:
**27,648 D2-frame evaluations**. Rows 1–24 are the four polar arms; rows 25–54 reevaluate archived controls.
Truth and standardized noise are exactly paired across arms; observations differ with the encoder.
The merger requires all nine arms for selected B/seeds and reports seed-paired differences against
balanced, learned-energy, fixed dense and learned dense. These old seeds are exploratory, not held-out.
No broad grid or extra encoder search is justified before this comparison.

The structured arms additionally retain 3,072 separate **oracle** pair diagnostics (first 32 frames/cell):
outside users and the first two slots' patterns are revealed, all 64² ordered payload pairs are scored
with physical likelihood, and count tails allocate two copies. Same/mixed-pattern strata remain separate.
This omits other pattern assignments; O(nV²) time and O(V²+nV) memory include normalization. It is not a
complete receiver or a full-system Bayes result. Dense controls have no corresponding pattern oracle.

```bash
uv run python -m unittest tests.polar_signature_job_test tests.polar_energy_geometry_test tests.polar_mapping_test
uv run python jobs/038_polar_signatures/build_manifest.py
uv run python jobs/038_polar_signatures/run_row.py --index 4 --smoke --out-root jobs/038_polar_signatures/results_local_smoke_new
uv run python jobs/038_polar_signatures/run_row.py --index 26 --smoke --out-root jobs/038_polar_signatures/results_local_smoke_new
```

Production remains an **unsubmitted** HPC campaign. Submit from a stable checkout containing these changes,
with completed Job032/034/035 archives available. Do not update the checkout used by active Job037 runs.
Paths can point to existing archives in another checkout:

```bash
module load miniforge/3
uv sync --frozen
qsub -v "POLAR_REFERENCE_RESULTS=/absolute/path/to/035/results_all_decoders,POLAR_EXPLICIT_RESULTS=/absolute/path/to/032/results/comparison,POLAR_MAPPING_RESULTS=/absolute/path/to/034/results" jobs/038_polar_signatures/038_polar_signatures.sh
```

If the default local-relative archive paths exist, plain `qsub jobs/038_polar_signatures/038_polar_signatures.sh`
is sufficient. The script requests two CPUs, 16GB, 24h per row; **no concurrency cap**, no GPU. Scheduler/site
limits determine concurrency; all 54 simultaneous requests would total 108 CPUs/864GB. Archived rows do no
training. Historical reference calibrations suggest about 7 job-hours for twelve new calibrations, with
substantial convergence/hardware uncertainty; reevaluation is much cheaper. The 24h value is a per-row
limit, not a campaign runtime forecast. Matching completed outputs are skipped; preserve partial outputs
and use a new output root for deliberate retries. Generated outputs remain ignored.

```bash
uv run python jobs/038_polar_signatures/merge.py --out jobs/038_polar_signatures/results_analysis
```

Use `--B 12 --seeds 3201` only for a complete nine-arm smaller cohort. The four-arm interaction and all-nine
D2 figures are separate. Local validation checks physical repair, exact factor geometry, native-SCL rejection,
independent pair posteriors, source pairing, archived normalization/checkpoint contract, manifest completeness
and submission settings. The existing 20 focused tests pass. The first learned-dense smoke exposed the
stricter unit-power tolerance; explicit recorded normalization fixes that real precision mismatch.

**Completed locally, 30 October:** all 42 reusable control rows (seven encoders × two B × three seeds)
now have 512 frames each in `jobs/038_polar_signatures/results_signatures/`: **21,504 evaluations**.
`validation.json` records independent power, physical-observation, count-metric and truth/noise pairing
checks; all 12 historical reference replays preserve their count decisions. Largest archived column
rescaling is 4.915e-7. These are new current-metric control measurements, not repair performance.
The nine-arm B12/seed3201 smoke also exercises construction, pairing, merger and plots; its repaired
training is one epoch and its results must not be ranked. Evidence is under
`results_nine_arm_smoke_2026-10-30/`. The 20 focused tests and Reports 8/10/11 compilation pass.

**Remaining HPC work:** twelve repair calibrations/evaluations. If keeping the local archived controls,
submit only the structured portion after transferring this code to a stable HPC checkout:

```bash
qsub -J 1-24 jobs/038_polar_signatures/038_polar_signatures.sh
```

This includes twelve inexpensive reference replays; rows 25–54 need not be submitted again. Supply the
reference-archive environment path above if it is in another checkout. Alternatively copy completed
`results_signatures/` to HPC and submit the full script: matching rows skip, leaving only the twelve
repairs. Never overwrite incompatible/partial outputs; original archives remain untouched. No HPC
submission has been made here. Pull result directories, rather than old source files, back into this
updated local checkout before the all-nine merger.

**Decision rule:** retain the reference unless the repair has a useful paired gain at matched resources.
A win over balanced does not imply a win over learned dense; inspect each contrast and codebook seed.
Confirm a promising result on held-out seeds and the full load/SNR grid. If no gain appears, record the
failed inference from conditional distance to system performance and close this finite repair trial.

## Tractability algebra checks: Report 11

[Report](reports/11_tractable_polar_decoding.tex) and
[small independent checks](../tests/polar_tractability_checks.py). These are finite identity checks,
not a new decoder benchmark. They verify independent-parity normalization/envelopes, rejection correction,
mixture detailed balance, conditional list-mass bounds, SCL path-metric constants and sparse count decisions.
A dependent-parity triangle in the actual Bc6/Nc32/CRC8 generator demonstrates exponentially poor acceptance
for a realizable family of conditional scores; it does not estimate typical acceptance or PUPE.

The [refinement checks](../tests/polar_tractability_refinement.py) verify explored-set
envelope repair, direct complement sampling, reliable-basis flip tails and failure/repair controls.
The [residual-reuse checks](../tests/polar_tractability_reuse.py) verify transferred conditional
coverage and correction on 40 tiny balanced polar context pairs. These test algebra and conditional kernels,
not practical discovery, mixing or payload scaling at fixed PUPE.

The [pair checks](../tests/polar_tractability_pairs.py) verify the physical/XOR likelihood identity and
its code-consistency factor on 20 tiny same-pattern fixtures, four-symbol local kernels and pair-MH
detailed balance on 64 ordered states. A guide-ranking reversal shows why independent sign
marginalization is only an approximation. Larger generator rank samples are structural controls,
not exhaustive identifiability certificates or noisy recovery evidence. The separate implementation checks
below now test a joint SCL decoder.

The [structural checks](../tests/polar_tractability_structural_checks.py) cover mixed-pattern parity conditioning:
12 three-user physical fixtures, four covers and 512 tuples per fixture, a same-pattern duplicate control,
normalized envelope correction and a failed greedy rank-selection example. Maximum identity discrepancy is
5.33e-15. A B12/seed3201 census over 2,016 distinct pattern pairs gives 31,941 one-sided sectors versus
1,093,347 two-sided table entries. These count representations; sector construction and acceptance still cost work.

```bash
uv run python -m tests.polar_tractability_checks
uv run python -m tests.polar_tractability_refinement
uv run python -m tests.polar_tractability_reuse
uv run python -m tests.polar_tractability_pairs
uv run python -m tests.polar_tractability_structural_checks
```

## Procedural decoder implementations: DEC-25–34

[Research log and declared variants](logs/POLAR_TRACTABILITY_LOG.md),
[procedural decoder code](../framework/polar_tractable.py),
[joint/XOR pair proposals](../framework/polar_pair_proposals.py).
DEC-25/26 are SCL/basis multiset beams; DEC-27/28/29 use independent/repaired/cached parity MH;
DEC-30/31 add joint/XOR same-pattern pair moves; DEC-32 uses SCL-list MH.
DEC-33 uses an exchangeable parity-product/beam importance proposal, with a global envelope and an
expected excess-PUPE bound. DEC-34 unions SCL and basis candidates before beam pruning.
All are IID-only and nonlearned. Their procedural interface returns sparse full-message count dictionaries;
the shared benchmark uses a dense output adapter and still constructs its explicit small-B encoder.
The new mixed-pattern parity-cover algebra is not yet one of these implementations.

**Validation, 29 October:** 66 focused tests pass, including 25 new tests covering normalized proposals,
exact repaired/complement draw laws, physical scores, single/cached/pair detailed balance, exhaustive tiny
REF endpoints, count readout and procedural/dense agreement. B100/P4/b98/N128/CRC16/n256 execution checks
run all eight variants at K1/2 without an alphabet array; this is not a multiuser B100 performance result.

**Local smoke evidence:** `results/tractable_smoke_2026-10-29/` stores 3,456 paired B4 evaluations
(32 frames/cell, K1/2/4, -4/4/10 dB), REF/D2/ODMA controls, both bound overlays and posterior-regret diagnostics.
`results/tractable_combined_B12_2026-10-29/` merges the mini and stress runs: 240 evaluations, four frames/cell,
K2/7/22, 4/10 dB, DEC14/ODMA controls. All records retain physical inputs, outputs, timings and hashes.
At K22/4 dB DEC26 matches DEC14 at PUPE .01136, versus DEC25 .52273; the tiny budget and four-frame
sample cannot establish a ranking. Empty CRC-valid discovery lists and weak SCL initialization are exposed
by the diagnostics. The log records each budget, the full table and the needed matched-initializer control.
The new algorithms improve with SNR in these checks; they are not uniformly competitive at the hard load.
Those initial runs predate the matched-initializer study below.

```bash
uv run python -m unittest tests.polar_tractable_test tests.polar_tractable_execution_test tests.polar_tractable_harness_test tests.polar_pair_proposals_test tests.polar_tractable_importance_test tests.polar_tractable_jobs_test
uv run python -m tests.polar_decoder_benchmark --profile smoke --decoders REF DEC-25 DEC-26 DEC-27 DEC-28 DEC-29 DEC-30 DEC-31 DEC-32 ODMA --loads 1 2 --frames 16 --ebn0 -4 4 --no-bounds --out-dir results/tractable_local_smoke
```

Use a fresh output directory. Saved `config.json` files reproduce the recorded budgets through `--config`;
the procedural B100 fixture is in `tests/polar_tractable_execution_test.py` because the common explicit
benchmark deliberately rejects B>20. Generated results and plots remain outside git.

**Matched CPU campaign:** [Job037](../jobs/037_polar_tractable/README.md) queues DEC-25–34 in 60 uncapped
jobs on exactly the balanced Job-035/036 physical grid, with audited historical D2/ODMA and selected
explicit-decoder reuse. The chains share width-32 basis initialization. No new HPC jobs have been submitted.
`results/tractable_matched_smoke_2026-10-29/` contains 1,872 paired B4 evaluations with REF/DEC14/ODMA,
both coding bounds and exact conditional-risk diagnostics. `results/tractable_historical_reuse_2026-10-29/`
verifies the real B12/seed3201 D2/ODMA/DEC14/DEC24 records merge with unchanged inputs and budgets.
Cost probes at B12/B14, K26/4dB use the actual Job037 budgets; their single-frame timings are planning
estimates, not PUPE evidence. Detailed commands, uncertainties and next gates are in the job README and log.
The current suite passes **81 tests**. `results/tractable_matched_B12_2026-10-29/` contains 96 paired
K22 evaluations at 4/10dB, four frames/cell and Job037 budgets. At 4dB DEC25 misses one copy; the other
nine tractable variants and DEC14 miss none. The chains start at zero-error configurations and usually
stay there, so this is not a correction/mixing success claim. `results/tractable_importance_bound_smoke_2026-10-29/`
reruns REF/DEC33 after tightening the K1 bound; all 144 DEC33 decisions/input hashes are unchanged.
Its K1 expected-regret bounds average about 0.48–0.50; all tested multiuser bounds remain 1 (vacuous).

## Returned decoder audit: 30 October 2026

```bash
uv run python jobs/037_polar_tractable/audit_results.py --out jobs/037_polar_tractable/results_analysis_NEW_DATE
# Replot an existing snapshot without rereading all frames:
uv run python jobs/037_polar_tractable/audit_results.py --plot-only --out jobs/037_polar_tractable/results_analysis_2026-10-30_verified
```

The retained local snapshot is `jobs/037_polar_tractable/results_analysis_2026-10-30_verified/`:
`audit.json`, `inventory.json`, `cells.json`, `aggregates.json`, `contrasts.json`, `log_failures.json`
and decoder-selection/cost PNGs. These are generated and ignored. The audit recomputes1,130,406 frame
metrics, checks inputs and provenance, and separates complete matched cohorts from uncertified cells.
The analysis script hash identifies the audit implementation; later plot-only additions do not change
that recorded provenance. Detailed decisions and limitations are in the existing
[tractability log](logs/POLAR_TRACTABILITY_LOG.md#30-october-returned-result-audit-pruning-decisions-and-common-d2-encoder-test).

Job036:36 complete,27 partial,one source-guard failure,17 absent from the original81 rows.
Job037:42 complete,18 partial of60. All 42 chosen balanced035 controls are complete, using certified
DEC-05 retries where available. These counts describe local artifacts, not scheduler status. Refresh
returned files before treating the missing036 results as unrun or resubmitting anything.

The replicated result is the explicit multiset beam14's B14 gain over D2: K26 at4/10dB gives
6.030/.341% versus40.400/8.569% count-PUPE. Procedural26 gives31.510/4.778%; single-slot correction32
improves that to30.198/3.135%. SCL-only discovery25 is weak; cached29, common-product IS DEC-33 and union34
have not earned further broad sweeps at the tested budgets. The33/34 closure evidence is B12-complete;
B14 remains incomplete. Pair31 is inferior to single32 in the highlighted hard cells.27/28 at B14 and
30 remain open. Old035/036 examples below are historical snapshots, not the latest coverage or ranking.

The next test is joint-state retention and conditional correction at matched cost, with14 and26→32
as the two retained references. No additional decoder array was created or submitted in this pass.
Existing running jobs should finish; no automatic cancellation or blanket resubmission is authorized
by these analysis decisions. Missing-copy and positive-tail capacity diagnostics localize information
loss but are not posterior-error certificates or causal proofs of the specific pruning failure.

## Active combination campaign: 036

[Commands, decoder matrix and posterior accounting](../jobs/036_polar_combinations/README.md);
[combination implementation](../framework/polar_decoder_combinations.py);
[numerical checks](../tests/polar_decoder_combinations_test.py);
[job/reuse checks](../tests/polar_combination_jobs_test.py).

DEC-14–24 cover multiset beam search, guide/product/beam proposal mixtures, terminal single-message
Gibbs and guided pair Metropolis, and a two-proposal-by-three-refinement SMC grid. All are nonlearned,
IID-only and explicit-alphabet. Budgets and physical correction are recorded in the common harness.
The fixed balanced B12/14 setup, seeds 3201–3203, K/SNR grids and 256 paired frames/cell match Job035.
The original manifest has 81 rows after splitting heavy B14 IS loads and 540,672 new evaluations,
at two CPUs/16 GB each, with no concurrency cap. The reduced 24 October plan retains 34 rows/237,568
evaluations. D2/ODMA are reused; three missing B14 DEC-05 controls are being retried separately. Controls are
joined only after pinned source, model, configuration, matrix, observation and seed-coverage checks;
original per-run provenance remains in the aggregate. This is exploratory reuse, not a fresh-seed test.

26 October snapshot: **19 complete, four partial** Job036 rows, with 147,456 complete and 12,288
provisional evaluations. All DEC-14–24 have B12 seed3201; DEC-14–17 also have seed3202, DEC-14 seed3203,
and DEC-14 its first B14 seed3201. B14 DEC-15/16 seed3201 K7/15 also completed. The audit verifies
524,288 input/count/PUPE records and
unchanged historical hashes. No Job036 tracebacks or missing selected inputs. Current-source revisions
743e81c/c19c377 have identical experiment hashes and pass the historical shared-definition audit.
B12 beam K26/4 dB PUPE .003105 versus D2 .025190 on three seeds. B14 beam gives .061298 versus
D2 .403996 and SMC .268930 on one seed. Two-seed B12 DEC-17 gives .002028 versus beam .003080,
but costs about 43 times more; pair-refined SMC is promising on one seed, not yet a replicated gain.
New partial B14/K26/4 dB DEC-16 gives .059645 versus beam .061298, at about 81 times the recorded cost;
its paired frame interval includes zero. No B14 terminal-refinement result is returned. Full current
paired tables, provenance inventory and decision boundaries:
`jobs/036_polar_combinations/results_combinations/analysis_2026-10-26/observations.md`.
The 25 October analysis retains the unchanged completed B12 comparisons and four plots.

Verification: 77 regression tests, 8,064 tiny paired smoke evaluations including REF, and production-size
CPU timing probes. The tiny smoke decreases with SNR for every new variant; it validates mechanics,
while returned HPC evidence establishes the measured comparisons above. Generated smoke outputs live under
`jobs/036_polar_combinations/results_local_*`; generated reports/plots are not versioned.

## Original decoder implementation harness: 035

26 October snapshot: **160/180 completed children, 76/90 complete bundles**. Their summary/frame hashes
match the previously audited 1,310,720 records. Fourteen children finished evaluation but failed the final
source-change guard (rows35,50,52,56,58,65,66,67,71,73,80,82,86,88); six are absent locally.
Preserve failed-source data separately. Three targeted balanced B14 DEC-05 retries are partial, with
6,656/6,912/6,912 frames and exact count/PUPE reproduction of the matching failed originals so far. No retry
is yet certified complete. Current main supports mapping filtering and a fresh retry root; the audited
Job036 merger accepts these only after completion with matching current source hashes.
Balanced B14 SMC has all three seeds: K26/4 and 10 dB PUPE .26237/.01838 versus D2 .40400/.08569.
B12 balanced guided IS also has all three seeds: K26/4 dB .00601 versus D2 .02519, and it retains the
lowest observed high-SNR errors among those original controls. The 26 October audit/results are above;
the preceding completed-run analysis remains in `jobs/035_polar_decoders/results_all_decoders/analysis_2026-10-22_update/`.

[Commands and contract](../jobs/035_polar_decoders/README.md);
[driver](../tests/polar_decoder_benchmark.py); [implementation](../benchmarks/polar_decoders.py);
[decoder bank](../framework/polar_decoders.py); [regression checks](../tests/polar_decoder_bank_test.py).

REF (exact count posterior), all thirteen Report-10 families, D2 and native/adapted ODMA are executable.
These are explicit-alphabet finite variants with recorded budgets, not all possible combinations.
DEC-02/03/10/11/13 are IID-only. DEC-08/10/12 restrict candidates and report support uncertainty;
signed intervals, chain mixing and sampling concentration may remain uncertified at the chosen budgets.
Matched frames, multiset PUPE, count outputs, seed/config/source fingerprints, physical noise,
uncertainty and setup/training/inference time are recorded centrally. The old no-implementation
constraint is superseded by the user's request.

The smoke contract is B4/n12/Bp2/Nc8/CRC4, placement3217, balanced or uniform energies,
K1/2/3/4, Eb/N0=-4/4, 128 frames/cell. D2 has eight layers with a shortened 30-epoch cap;
all new algebraic receivers have zero learned parameters. Tiny REF enables posterior-tail and conditional
regret measurements after decoding. These panels validate operation and reveal approximation behavior,
not performance at the research bank or the paper's native operating point.

The two displayed coding references are the existing p_t+q_1 achievability upper bound and the restored
historical genie/list-Fano lower bound, whose list-miss lower bound also lower-bounds multiset PUPE.
Neither is the optimal error of the fixed encoder. The same-encoder exact Bayes reference is more specific.

The complete 90-job CPU array uses B12/14, balanced/uniform mappings, seeds3201–3203, all thirteen
new families and D2/ODMA. Each job runs both mappings and all four loads sequentially;
D2 still trains once per encoder/seed. Four concurrent jobs at two CPUs each limit the array to eight
CPUs; the same 1,474,560 evaluation workload is preserved. Nothing is queued. All practical receivers
completed local timing probes at B12/14 K7/26;
the expensive independent-guide and field paths justify CPU HPC for the full grid. REF's combinatorial
cap rejects research-load enumeration before allocation. The job README and experiment records own current
smoke evidence, timings, limitations and commands.

## Exact foundations

Report references below use section titles, which remain meaningful if pagination changes. Helpers live in
tests/ because they are finite mathematical references, not production receivers.

| ID / question and corresponding Report 7 algebra | Executable checks / helper | Contract and retained conclusion |
|---|---|---|
| E1 — What posterior and decision are correct? Common posterior, optimal decision and conditional polar interface | [polar_posterior_test.py](../tests/polar_posterior_test.py), [pupe_metrics_test.py](../tests/pupe_metrics_test.py) | Tiny exact count/slot equivalence, factorial priors, count-tail Bayes action and polar conditional evidence. Some additional legacy checks lie outside the current exact-report scope. Selected-code pair certificate is separate. |
| E2 — Can the finite exact decoder be completed? “Complete the exact residual-likelihood decoder”; inside/outside prefixes; polar algebra | [polar_exact_inference_test.py](../tests/polar_exact_inference_test.py) | Both sources, unknown patterns, every count law and minimum-PUPE action agree with exhaustive inference. Equal-signal identities retained; aggregate-XOR shortcut fails. |
| E3 — Are other exact representations equivalent? Fourier, joint elimination and connected-group sections | [polar_alternative_inference_test.py](../tests/polar_alternative_inference_test.py) | Smaller finite fixtures verify marked Fourier sums, omitted-mode bounds, elimination orders and all connected orders. Low-order signed failure is retained. |
| E4 — What information may be forgotten? Sufficient states, completion geometry, connected bounds and identifiability | [polar_exact_reductions_test.py](../tests/polar_exact_reductions_test.py) | Exact research-bank span and selected-code alias certificates; finite support/error bounds. Three-user same-pattern injectivity ends at four users. Conditional noiseless risks are not full-system floors. |
| E5 — Do reductions construct smaller exact calculations? Compatible schedules, mixed-state lower bounds and decision contrasts | [polar_exact_computation.py](../tests/polar_exact_computation.py), [checks](../tests/polar_exact_computation_test.py) | All 24 tiny pattern orders; 24-observation panel × 6 bound/certificate variants = 144 searches. Research-bank private-coordinate certificate gives >=2^48 distinguishable rows at the specified cut. |
| E6 — Can bounds/operators reuse work? Overlap parity, XOR convolution, support reuse, adverse prior masses and rooted trees | [polar_exact_reuse.py](../tests/polar_exact_reuse.py), [checks](../tests/polar_exact_reuse_test.py) | Six-bank pair census; 120 paired search comparisons; exact K2 counts and weighted tree identities. Maps/setup and event/frontier work are charged separately. |
| E7 — What happens at the first cycle? “Three users: close the triangle exactly” | [polar_three_user.py](../tests/polar_three_user.py), [checks](../tests/polar_three_user_test.py) | Complete K3 counts on 24 observations; marked joint-parity checks; restricted 56-triple census per B12 bank. Edge deletion is a counterfactual diagnostic. K4 trace shortcut fails. |
| E8 — Which exact observations combine? “Exact reference changes, invariants and decision contrasts” | [polar_synthesis_test.py](../tests/polar_synthesis_test.py) | Finite exact recentering/cross-ratios, reweighting contrasts, alias conditionals and slot symmetry. Quantitative approximation-error resources now belong to Report 9. No new checks or panel. |
| E9 — Does energy balancing explain the chosen construction? Report 8 | [benchmarks/polar_energy_geometry.py](../benchmarks/polar_energy_geometry.py), [checks](../tests/polar_energy_geometry_test.py) | Exact moments/defect decomposition, balancing and failure conditions. Read-only returned-checkpoint audit; no new PUPE/training result. |

Retained numerical conclusions from E5–E7:

| Calculation | Measured result and limit |
|---|---|
| E5 dual/contrast, evaluated count hypotheses | IID means 3.22/23.62/43.11%; distinct .98/22.46/51.03%, in increasing sigma² order. Thousands of support scores are additional; no runtime speedup is established. |
| E6 eight cached directions + adverse-prior contrasts | 128 root message scores on M16; evaluated-hypothesis group means 11.47–61.47%. Suffix/box/event/frontier work remains. |
| E5/E6 two-block connected certificates | 0/24 polar observations certified, even after improved tree weighting. A separate constructed high-noise K4 example certifies at order two. |
| E6 pair representation | 6.83–7.24x fewer intersecting-pair table entries than symmetry-reduced dense tables; maps/setup additional. B12->B14 parity entries grow about 16–17x. |
| E7 K3 full count laws | Maximum absolute reference discrepancy 1.56e-15; equal Bayes gain. M pair applications remain. |

Reproduce finite audits separately; these commands can be substantially more expensive than their unit checks:

```sh
uv run python -m tests.polar_exact_computation --output /tmp/polar-exact-computation.json
uv run python -m tests.polar_exact_reuse --output /tmp/polar-exact-reuse.json
uv run python -m tests.polar_three_user --output /tmp/polar-three-user.json
uv run python -m benchmarks.polar_energy_geometry
```

The 14 October scaling audit additionally checked scalar-reverse count readout, cheaper Fourier markers,
inverse connected assembly and shared C2/C3 extraction on a five-word fixture, both sources and K<=3.
Those were one-off algebra cross-checks, not separate permanent experiment drivers or optimized implementations.
Their cost consequences are retained in Report 7.

## Parked approximation scripts

These pre-reset scripts are preserved for possible later systematic work. They are not active experiments,
accepted assumptions or a validated approximation bank. The previous A1–A3 numerical comparisons and
combined test-run narrative have been removed. Current assumptions/derivations belong to
[Report 9](reports/09_polar_bayesian_approximations.pdf).

| Existing driver / checks | Historical setup and aim |
|---|---|
| [polar_approximation_audit.py](../tests/polar_approximation_audit.py), [checks](../tests/polar_approximation_audit_test.py) | B4/P4/V4/Nc8/CRC4/n12/K3, placement3217, both sources, sigma²=.05/.3/1, seeds0–3. Separate proposal restriction from conditional forest/sampling error on 24 observations. |
| [polar_revisable_sampling.py](../tests/polar_revisable_sampling.py), [checks](../tests/polar_revisable_sampling_test.py) | B6/P4/V16/Nc16/CRC4/n24/K3, placements3218/3219, both sources, sigma²=.05/.3, seeds0/1. Revisable full-space proposal/sampling study on 16 observations. |
| [polar_branch_audit.py](../tests/polar_branch_audit.py), [checks](../tests/polar_branch_audit_test.py) | B4/M16, placements3217/3219, K3/K4, both sources, sigma²=.05/.3/1, seeds0/1. Explicit-alphabet inference comparison on 48 observations. |
| [polar_assumptions_test.py](../tests/polar_assumptions_test.py) | Finite algebra helpers for former approximation formulations; retained without adopting their assumptions. |

The larger-payload bank rejected placement3217 for uncovered resource13; the B4 two-bank study rejected
placement3218 for incomplete coverage. These historical setup qualifications remain; no support repair is implied.
No command in this section is a request to rerun these studies during the algebra-only stage.

## Matched performance context: jobs 032–034

Recorded audits: 032/033 on 6 October, 034 on 9 October, geometry follow-up on 10 October. These results
motivated the fixed-polar decoder study; their training/native-receiver work is separate from E1–E9/A1–A3.

| Bank | Recorded completion | Purpose / principal conclusion |
|---|---|---|
| [032](../jobs/032_published_baselines/README.md) | 161/174 main rows; D4 11/24 | Native published-method adaptations versus explicit D0–D4; all other main groups complete. Incomplete D4 excluded from aggregate claims. |
| [033](../jobs/033_matched_d2/README.md) | 66/66; all 42 references reused | D2 on unchanged published codebooks; most native ODMA versus dense/D2 loss is receiver-side. |
| [034](../jobs/034_polar_mapping/README.md) | 12/12; all 36 controls present | Learned polar energies approach dense/D2; shared mixer worsens these runs and is retired. Checkpoint/energy audits passed. |

Shared production contract: real Gaussian MAC, B12/14/n256, K7/15/22/26, Eb/N0=-4:2:10 dB,
three seeds and 256 frames/cell/seed. Train distinct; evaluate distinct/IID separately. Known K/noise/codebook,
eight D2 layers, 200-epoch cap, patience 10 and best-checkpoint restoration. Fixed means encoder frozen,
decoder trained. Preserve exact-unit versus nominal/message-dependent energy conventions.

| Full-grid distinct mean PUPE | B12 | B14 |
|---|---:|---:|
| Fixed dense / D2 | .17290 | .28526 |
| Fixed polar / D2 | .18346 | .29620 |
| Learned polar energies / D2 | .17116 | .28776 |
| Shared mixer / D2, retired | .18735 | .30270 |

B14/K26/10 dB: native polar .55874; same polar/D2 .13712; learned energies/D2 .10617; dense/D2 .09831.
Close means are not formal equivalence. Balanced energies have fresh smaller-grid diagnostics rather than
matched full-grid PUPE evidence. Frame defect alone cannot rank performance: balanced/learned B14 defects
.00657/.00840 are below dense .01562, despite dense's lower measured PUPE. Report 8 retains that derivation.

D3/D4 use 1,024 non-oracle candidates obtained by scanning the full alphabet. Proposal misses count; refinement
cannot recover omitted words. Dynamic-CS native results remain unvalidated and excluded. Small-B ODMA/CCS
adaptations are not paper-scale reproductions. ODMA cap30 fixes premature SIC truncation; cap60 agrees in its
recorded native control. Bounds retain their conventions: achievability is an error upper bound, converse a
necessary-condition lower bound; neither certifies encoder/decoder optimality.

Local generated audits, when present: [033 review](../jobs/033_matched_d2/results/analysis/review_2026-10-06/REPORT.md)
and [034 review](../jobs/034_polar_mapping/results/analysis/review_2026-10-09/REPORT.md).
The deleted supervisor-meeting snapshot is no longer a navigation dependency.

## Earlier research retained for reproduction

| Jobs | Recorded completion | Retained conclusion / scope |
|---|---|---|
| 001–020 | See Results 03/04 | Original ODMA/global recovery, oracle controls and early CCS; contracts differ from 032. |
| 021–022 | 20/20 + 60/60 | Explicit factorized encoders/D0/D1; unequal budgets. Implicit multiplication leaves M-sized state. |
| 023–026 | 16/26, 8/8, 6/6, 6/6 | Sectioned execution and L1 equivalence pass; L>1 association/load failures remain. Missing 023 causes unrecoverable; do not resubmit unchanged. |
| [027](../jobs/027_sparse_density_frontier/README.md) | 72/72 | Sparse-density frontier; repaired dense endpoint has changed RNG consumption. |
| [028](../jobs/028_hash_skeleton_B14/README.md) / [029](../jobs/029_joint_encoder_decoder_B14/README.md) | 36/36 / 20/20 | Selected hash competitive; joint comparison also changes budget. Small-B offline selection does not scale automatically. |
| [030](../jobs/030_prototype_amplitude_frontier_B14/README.md) | 64/64 | Whole-vector sharing degrades; no longer the preferred compression route. |
| [031](../jobs/031_coordinate_amplitude_frontier_B14/README.md) | 72/72 | Coordinate J8 approaches unrestricted parent; later matched D2 comparison is in 033. Historical 120-epoch/patience5/noise conventions differ. |

[Results 03](../results/03_results.md), [Results 04](../results/04_results.md), Reports 1–6 and preserved manifests
retain detailed historical evidence, oracle assumptions and failures. None establishes complete B100 decoding.
