# Experiment bank

Updated **22 October 2026**. This document maps research questions and algebra to executable scripts,
fixture contracts and retained results. [CURRENT_STATE.md](CURRENT_STATE.md) owns priorities; the
[exact](POLAR_DECODING_RESEARCH_LOG.md) and [approximation](POLAR_APPROXIMATION_LOG.md) logs own reasoning.
Job READMEs/manifests own HPC row definitions. Historical exact audits and matched performance evidence below retain their original scope. Approximation
preparation is complete for the retained paths. All thirteen families now have shared-harness implementations;
pre-reset prototype scripts remain parked. The user submitted Job035; Job036 is prepared but not submitted by the agent.

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

## Active combination campaign: 036

[Commands, decoder matrix and posterior accounting](../jobs/036_polar_combinations/README.md);
[combination implementation](../framework/polar_decoder_combinations.py);
[numerical checks](../tests/polar_decoder_combinations_test.py);
[job/reuse checks](../tests/polar_combination_jobs_test.py).

DEC-14–24 cover multiset beam search, guide/product/beam proposal mixtures, terminal single-message
Gibbs and guided pair Metropolis, and a two-proposal-by-three-refinement SMC grid. All are nonlearned,
IID-only and explicit-alphabet. Budgets and physical correction are recorded in the common harness.
The fixed balanced B12/14 setup, seeds 3201–3203, K/SNR grids and 256 paired frames/cell match Job035.
There are 81 rows after splitting heavy B14 IS loads, 540,672 new evaluations, and a four-job concurrency
cap at two CPUs/16 GB each. No D2/ODMA or old-control reruns are scheduled. The historical controls are
joined only after pinned source, model, configuration, matrix, observation and seed-coverage checks;
original per-run provenance remains in the aggregate. This is exploratory reuse, not a fresh-seed test.

Verification: 77 regression tests, 8,064 tiny paired smoke evaluations including REF, and production-size
CPU timing probes. The tiny smoke decreases with SNR for every new variant; none is claimed to improve
research-scale PUPE before the HPC results. Generated outputs live under
`jobs/036_polar_combinations/results_local_*`; generated reports/plots are not versioned.

## Original decoder implementation harness: 035

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
cap rejects research-load enumeration before allocation. The job README and design log own current
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
Their cost consequences are retained in the exact log.

## Parked approximation scripts

These pre-reset scripts are preserved for possible later systematic work. They are not active experiments,
accepted assumptions or a validated approximation bank. The previous A1–A3 numerical comparisons and
combined test-run narrative have been removed. Current assumptions/derivations belong to
[Report 9](reports/09_polar_bayesian_approximations.pdf) and the [working log](POLAR_APPROXIMATION_LOG.md).

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
eight D2 layers, 200-epoch cap, patience10 and best-checkpoint restoration. Fixed means encoder frozen,
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
