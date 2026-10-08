# ODMA-URA: Current Research State

Updated 8 October 2026. This is the live research handoff; the [experiment ledger](EXPERIMENT_BANK.md) owns job status,
and the [reports](README.md) retain derivations and historical decisions. No new receiver architecture is committed.

## Aim and current decision

Develop tractable multiuser decoding that exploits useful encoder structure without giving up the recovery performance
seen with explicit small-B codebooks. The immediate priority is **receiver inference and its computational cost**, not
another broad search over encoder restrictions. This does not establish that encoding is optimal or unimportant.

Two directions remain open: improve inference on the existing ODMA–polar construction, or retain polar coding while
allowing a broader family of resource-placement/spreading operators. Polar coding and ODMA placement are separate
choices. Job 034 now tests two modest mapping extensions with unchanged D2; receiver selection remains open.

## Implemented next batch: 034

[034](../jobs/034_polar_mapping/README.md) adds 12 job-33-matched runs: learned per-pattern polar-symbol energies and a
shared orthogonal mixer before placement. Both start at exactly the original codebook, enforce unit energy in every
forward call, and train jointly with D2. Fixed polar, native ODMA and dense/sparse controls are reused from 032/033.
Smoke, B8 mini-learning and reduced-budget B14 checks pass; no production result is claimed. The polar frame identity
accelerates exact D2 geometry without changing its recurrence or removing its full-message state.

Investigation 1 is theory first, not an implemented candidate receiver. [Report 7](reports/07_polar_d2_inference.pdf)
derives D2 and polar updates, identifies their overlap, and reviews spreading/MMSE, soft cancellation, soft-output SCL
and learned polar work. Unknown patterns, same-pattern collisions and credible extrinsic uncertainty remain central.

## What the evidence says

Job 033 is complete and supplies the same-codebook D2 controls missing from job 032. Representative production PUPE:

| B=14, n=256, K=26, 10 dB; distinct messages | PUPE |
|---|---:|
| ODMA–polar, native receiver | 0.55874 |
| Same ODMA–polar codebook, trained D2 | 0.13712 |
| Fixed dense, trained D2 | 0.09831 |
| Fixed random sparse, support 32, trained D2 | 0.13957 |
| Fixed hash + coordinate J8 amplitudes, trained D2 | 0.13552 |

Replacing the ODMA receiver closes about 92% of its gap to dense D2 at this point. Dense still has an edge; the result
favours investigating decoding, not declaring the encoders equivalent. Current ODMA–polar uses 64/256 patterns at
B12/B14, not the four-mask ODMA control that performed poorly in earlier sparsity experiments.

Coordinate amplitude banks remain useful compact encoder controls. Joint J8 is close to the unrestricted hash parent
on the full B14 grid with 64x fewer amplitude values. Some production runs were training-sensitive; post-hoc decoder
calibration improves their tails without changing the encoder. Keep those diagnostics separate from production rankings.

Small-B CCS has exact multiuser aliases and approximate native alignment; it is not evidence against CCS at its native
scale. All recent comparisons use known K and noise. Our native implementations/small-B adaptations are not blanket
reproductions of published curves. See the [meeting report](reports/2026-10-07_supervisor_meeting/report.pdf).

## System and computational boundary

Messages w have B bits, so M=2^B. A common codebook Phi has shape n x M and produces y=Phi a+z, where the nonnegative
integer vector a counts transmissions. Binary message labels are distinct from real/complex channel symbols.
Exact-unit-energy and nominal-energy families, and distinct-message and iid evaluation, are labelled separately.

- D0/D1 are historical global-message learned receivers. D0 has a diagnosed count-to-list readout limitation.
- D2 is a known-K Bernoulli effective-channel approximation with learned variance scale and damping: 16 scalars for
  eight layers. It visits all M messages and keeps M probabilities; it is neither exact Bayes nor AMP.
- D3/D4 refine a supplied list. Production proposals still rank all M messages by matched-filter evidence.
  Compact refinement is not scalable search; report proposal misses as well as final PUPE.
- Procedural hash/coordinate amplitudes generate selected B100 codewords with exact energy, not tractable multiuser
  decoding. Section-domain M-free execution likewise did not establish useful recovery.

The hard question is how to represent and update uncertainty without an M-sized state, not merely how to avoid storing Phi.

## Polar coding and placement: an open middle ground

A polar code uses a recursively structured binary transform, fixes selected input bits, and puts information in the
remaining positions. These constraints define valid codewords and support recursive decoding without an exponential
lookup table. They do not specify which physical resources carry the symbols. See [Arikan's paper](https://arxiv.org/abs/0807.3917).

For the current linear CRC/polar encoder, split B=B_p+B_c bits into a pattern index p (2^B_p choices) and payload v:

~~~text
v in GF(2)^B_c -> c=L v in GF(2)^N_c -> x(v)=(-1)^c/sqrt(N_c) -> phi(p,v)=W_p x(v).
L: N_c x B_c binary effective generator; W_p: n x N_c real placement/spreading operator.
~~~

Current ODMA uses W_p=S_p, a coordinate-insertion matrix. An extension could keep L fixed but change W_p through
structured placements, permutations/signs or tractable spreading. W_p^T W_p=I preserves unit energy and within-pattern
distances, but not necessarily cross-pattern distinguishability or multiuser recovery. It is one sufficient energy
condition, not a requirement that every future operator must be an isometry.

An arbitrary learned matrix per complete message would restore the exponential problem. Even compact W_p can make
the effective channel difficult for a polar decoder. Every proposed freedom needs both a forward and an inference budget.
Neither polar invertibility nor a factor graph guarantees tractable noisy multiuser recovery. Pattern collisions must
not be excluded merely because full-message collisions are rare. Only the two constrained mappings above are implemented;
arbitrary learned placement/spreading and a tractable joint receiver are not.

## Questions to resolve next

1. Understand polar encoding/decoding, then derive the joint activity, pattern and coded-payload posterior.
2. Separate exact structural replacements for D2 operations from new inference approximations. Compare revisable
   codeword lists with soft graph inference; D2's exact recurrence need not be preserved.
3. Isolate soft cancellation, uncertainty calibration, hypothesis competition and known-K enforcement before crediting
   learning. Keep physical noise, residual interference and decision calibration distinct; avoid double-counting observations.
4. Compare with existing joint ODMA and soft polar receivers before claiming novelty. Weighted BP and conditioned
   denoisers are established ideas; learn where a measured approximation/calibration benefit warrants it.
5. At small B, measure inference loss on identical codebooks. At larger B, measure runtime, memory, proposal recall
   where applicable and PUPE. Unknown activity and false alarms remain an unresolved requirement.

Starting references: [joint ODMA pattern/data recovery](https://ieeexplore.ieee.org/document/10105449/),
[soft-output SCL](https://arxiv.org/abs/2412.03623), [learned polar graph decoding](https://arxiv.org/abs/1811.09801),
and [ODMA–polar soft inference in MIMO](https://arxiv.org/abs/2608.28085) (different channel/pilot assumptions).
The older [conditioned-inference brief](archive/conditioned_inference_brief.md) remains a hypothesis, not an agreed paper plan.

## Where to look

- [Experiment ledger](EXPERIMENT_BANK.md): dated completion, conclusions and exact run-contract links.
- [Documentation map](README.md): meeting summary, seven technical reports and detailed historical evidence.
- framework/learned_decoders.py and candidate_decoders.py: D2 and bounded-list receiver algebra.
- baselines/polar.py and odma_polar.py: coding, placement and native receiver, already separate modules.
- framework/prototype_amplitudes.py and hash_skeleton.py: retained generated-encoder controls.
- framework/polar_mapping.py: unit-energy polar mappings and the small-B D2 certification adapter.

Job 034 is ready for submission; receiver architecture remains a research decision. Code, manifests and original
outputs are retained for reproduction; historical reports and their proposed next steps are not the current roadmap.
