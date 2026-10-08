# Experiment Ledger

Returned results audited on **6 October 2026**; new batch added 8 October. This indexes evidence,
not a new full result audit or submission request. The [handoff](CURRENT_STATE.md) owns research priorities;
job READMEs/manifests own run contracts, and the [reports](README.md) retain the algebra.

## Next batch: 034 — polar energy and shared mixer

[034](../jobs/034_polar_mapping/README.md) is implemented and locally verified, not yet a production result. Twelve
rows add learned per-pattern symbol energies or one shared orthogonal pre-placement mixer to the selected 033
ODMA–polar construction. Both retain the original polar/CRC/positions, exact unit energy and full-message D2.
Uniform/identity initialization exactly reproduces all six 033 waveforms. Fixed controls are reused, not rerun.

The B12/14,n256 grid, seeds, load/SNR cells, 200-epoch cap, patience 10 and evaluation match 033. This tests whether
extra waveform freedom improves recovery with the same receiver. Neither arm demonstrates scalable decoding.
The B8 mini loss falls from 0.10815 to 0.07585/0.07414 for energy/mixer (fixed control 0.08205); smoke and reduced-budget
B14 execution also pass. These are learning checks only. [Report 7](reports/07_polar_d2_inference.pdf) separately
develops the deferred D2/polar receiver algebra and literature map. No new receiver was silently selected.

## Latest matched comparison: jobs 032–033

| Bank | Returned | Purpose and outcome |
|---|---:|---|
| [032: published baselines](../jobs/032_published_baselines/README.md) | 161/174 main rows; only D4 incomplete | Native ODMA/CCS versus explicit dense/sparse and D0–D4. Strong D2 results motivated same-codebook controls. |
| [033: matched D2](../jobs/033_matched_d2/README.md) | 66/66 | D2 on unchanged published codebooks plus hash/coordinate amplitudes. Most native-ODMA versus dense-D2 loss is a receiver gap. |

All native/D0/D1/D2/D3 groups in 032 are complete; D4 has 11/24 rows. Its remaining rows do not affect 033, which
reuses all 42 native and dense/sparse D2 references. Historical validation: 80/80 pilot rows, 24/36 original native rows,
20/20 follow-up checks. Dynamic CS is excluded because its native implementation remains unvalidated, not because
the published method was shown inferior.

### Shared contract

- Real Gaussian MAC; B=12/14, n=256, K=7/15/22/26, physical Eb/N0=-4:2:10 dB; three seeds, 256 frames/cell/seed.
- Train distinct messages; evaluate distinct/iid separately. Receiver knows K, noise and codebook.
- Eight layers, 200 epochs maximum, patience 10, best-checkpoint restoration. Fixed means encoder frozen, decoder trained.
- 033 keeps published codebooks identical to 032. Hash rows share support and compare unrestricted amplitudes with
  coordinate J=4/8/10, fixed/joint. Discrete maps stay fixed; joint rows learn amplitude entries and D2.
- Exact-unit and nominal/message-dependent energy remain separate. Explicit joint references project columns;
  hash joint rows normalize raw banks in the forward map, so their optimization parameterizations differ.
- D3/D4 use 1,024 non-oracle full-alphabet matched-filter candidates. Misses count, membership is hard, and search
  still enumerates all messages. Original oracle-complete local checks are not production proposer evidence.

### Main conclusions

At B14,K26,10 dB with distinct messages:

| Encoder / receiver | PUPE |
|---|---:|
| ODMA–polar / native | 0.55874 |
| Same ODMA–polar / D2 | 0.13712 |
| Fixed dense / D2 | 0.09831 |
| Fixed random sparse s32 / D2 | 0.13957 |
| Fixed hash coordinate J8 / D2 | 0.13552 |

1. Replacing only the ODMA receiver closes 91.6% of its gap to dense D2 in this cell and 95.5% over the full grid.
   Dense retains an advantage; neither encoder equivalence nor superiority at the papers' native scales is established.
2. Current ODMA–polar has 64/256 patterns at B12/B14 and polar/CRC symbols. It is not the old four-mask control.
3. Joint coordinate J8 has B14 full-grid mean 0.29361 versus 0.29331 for the unrestricted hash parent, with 64x fewer
   amplitude values. Three seeds support promising compression, not formal equivalence or scalable inversion.
4. Some parent/J4 runs are training-sensitive. Post-hoc decoder-only calibration transfer improves tails on unchanged
   waveforms; it is a diagnostic, not a replacement production ranking. Joint encoder learning is mixed.
5. Small-B CCS retains exact multiuser aliases and is not rescued comparably by D2. Native B128 uses a different
   construction; these adaptations do not establish that published CCS is intrinsically poor.
6. D3 exposes a search/refinement trade-off. Dense B14 at this cell gives 0.07943 versus D2's 0.09831; at B12 its
   0.00726 PUPE is already its proposal floor, versus D2's 0.00035. Refinement cannot recover an omitted message.
7. D0's count-to-list readout contributes to some high-SNR tails. Retain/label outputs; re-evaluation is deferred.
   This does not invalidate matched native/D2 comparisons.

### Native checks and bounds

ODMA cap30 removes premature cancellation truncation: native K100,0.5 dB PUPE changes 11.22% to 4.58% on the same
128 frames; cap60 agrees. Cap30 cannot truncate productive small-B recovery at K<=26. CCS alignment remains approximate;
keep the original selected small-B budgets. Exact fidelity caveats and tuning are in the 032 README.

Polyanskiy's p_t+q_1 curve is an **achievable-error upper bound**, not a floor on decoder performance. The converse is
a lower bound combining the available genie-list and non-genie rate-distortion/capacity constraints. Match sampling,
K, energy and channel conventions. At B14,K26, distinct-message 5% references are about 4.68 dB sufficient and 0.07 dB
necessary; none of the principal production D2 means reaches 5% by 10 dB. Neither curve certifies decoder optimality
or applies unchanged to nominal-energy codes.

Evidence: [meeting report and reproducible snapshot](reports/2026-10-07_supervisor_meeting/README.md) and the local
[full 033 audit](../jobs/033_matched_d2/results/analysis/review_2026-10-06/REPORT.md). The latter contains paired tables,
checkpoint checks and calibration diagnostics; it is a local generated artifact, not a versioned report.

## Earlier experiments: evidence, not the current roadmap

| Jobs | Completion | Question and retained conclusion |
|---|---:|---|
| 001–017 | See Results 03 | Initial ODMA/global recovery and oracle controls: much observed loss was practical support search. |
| 018–020 | See Results 04 | Collision-rich oracle checks and early author-code CCS; assumptions differ from 032. |
| 021–022 | 20/20 + 60/60 | Factorised explicit encoders and D0/D1. Broad sparse supports were competitive; four-mask ODMA was not. |
| 023 | 16/26 | Small section pilot. Learning helped some low-load cases, BP usually did not; do not repeat the missing sweep unchanged. |
| 024 | 8/8 | B128 M-free execution and energy passed; saturated local inference gave PUPE 1.0. |
| 025 | 6/6 | L=1 bridge: global/section-compatible outputs agree; Binomial prior changes PUPE by less than 0.0037. |
| 026 | 6/6 | L>1 bridge: correct signal algebra, association deteriorates with load; triadic helps at low load, tested BP worsens results. |
| [027](../jobs/027_sparse_density_frontier/README.md) | 72/72 | Sparse-density frontier at B12,n256; useful sparsity is broad, not arbitrarily close to zero. |
| [028](../jobs/028_hash_skeleton_B14/README.md) | 36/36 | Tables/hashes versus iid support at B14,T16/32. Selected hash is competitive. |
| [029](../jobs/029_joint_encoder_decoder_B14/README.md) | 20/20 | Joint amplitudes and D0/D1: selected hash and iid sparse stay close at matched support. |
| [030](../jobs/030_prototype_amplitude_frontier_B14/README.md) | 64/64 | Whole-vector sharing degrades as J falls. Retained for reproduction, not the preferred route. |
| [031](../jobs/031_coordinate_amplitude_frontier_B14/README.md) | 72/72 | Separate scalar labels: balanced J8 is close to the unrestricted D0/D1 parent; motivates the completed 033 comparison. |

Historical qualifications:

- 021/022 D0/D1 had unequal budgets. Implicit multiplication removed matrix materialization, not M-sized state.
  Full contracts and receiver-knowledge audits remain in Results 04 and Report 3.
- 023 overwrote PBS logs; missing-row causes cannot be recovered. Do not resubmit 024 unchanged. The 026 failure
  does not prove that all sectioned constructions or BP variants fail at larger loads.
- 027's repaired seed-2702 full-density endpoint changed RNG consumption and is not a clean nested paired control.
  Intermediate supports remain paired. D0 is near-flat through s about 48; strict-aggregate D1 at s16 is about 0.0293
  PUPE above dense with a shorter budget. Correlation tails worsen before labels run out; s1 leaves only 512 real
  signed directions for 4,096 messages.
- 028 versus 029 changes budget as well as encoder learning, so it is not a matched causal learning ablation.
  Small-B best-of-128 hash selection enumerates XOR differences; sampled B100 generation does not scale that search.
- 030/031 used 120 epochs maximum and patience five, not the later 032/033 defaults. In 031, fixed D1 J8 versus parent
  is 0.32683 versus 0.32937; joint is 0.32466 versus 0.32174 (8/12-dB aggregate). Different sampling, training and legacy
  noise conventions prevent pooling those curves with 032/033.

Detailed history remains in [Results 03](../results/03_results.md), [Results 04](../results/04_results.md) and Reports 1–6.
Preserved manifests define exact rows/budgets; this index does not replace them. No bank establishes a complete B100
generated-codebook receiver.
