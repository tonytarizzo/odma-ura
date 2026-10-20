# Why the project moved from encoder search to polar multiuser decoding

18 October 2026. Historical evidence only; no new receiver was evaluated for this note.

The original question was whether an unrestricted, learned sparse codebook could substantially outperform useful
structured communication schemes, and whether that advantage could survive replacing the stored matrix by a generator.
The experiments made that premise less compelling: encoder families often clustered together, while receiver choices
produced much larger differences. Polar now supplies a strong structured reference for studying inference.

## The decisions and the evidence

| Research choice | What the measurements support |
|---|---|
| Pause broad sparsity/hash/amplitude searches | Hash support was competitive, and coordinate J8 retained its parent's performance with 64x fewer amplitude values at B14. Neither supplied a demonstrated scalable multiuser inverse. Polar now offers similarly strong performance and useful algebra for decoding. This is a prioritization decision, not evidence that hashing cannot work. |
| Retire whole-vector amplitude sharing | Earlier Job 030 showed degradation when many messages shared entire amplitude vectors. Coordinate sharing recovered useful compression later. Do not merge these different parameterizations into one claim that all amplitude compression failed. |
| Retain polar structure | With the same D2 receiver, learned-energy polar reaches full-grid mean PUPE 0.17116/0.28776 at B12/B14, versus dense 0.17290/0.28526. This retains strong measured performance while giving an explicit coding structure to exploit. These close means do not prove optimality. |
| Keep mapping/energy treatment simple | Row-energy imbalance explains much of energy learning's benefit. Deterministic balancing gives encouraging smaller-grid results and is the frozen current construction. It has no matched full-grid PUPE ranking yet. |
| Retire the shared mixer | It worsened full-grid means to 0.18735/0.30270, compared with uniform polar 0.18346/0.29620. Its family included identity; the result establishes poor usefulness of this trained arm, not an impossibility theorem. |
| Focus on joint inference | At B14/K26/10 dB, native polar is 0.55874; D2 on the identical codebook is 0.13712. About 92% of the original native-polar versus dense-D2 gap disappears by replacing the receiver alone. Later likelihood refinement halves much of D2's remaining error. |

## A clean way to describe the sparse comparison

At B14, full-grid distinct-message mean PUPE under trained D2 is:

| Encoder | Mean PUPE | Encoder parameters |
|---|---:|---|
| Dense | 0.28526 | Fixed |
| Random sparse T32 | 0.29325 | Learned amplitudes |
| Hash support, stored amplitudes | 0.29331 | Learned amplitudes |
| Hash support, generated coordinate J8 | 0.29361 | Learned amplitudes |
| Polar with uniform energy | 0.29620 | Fixed |
| Polar with shaped energy | 0.28776 | Learned symbol energies |

All use trained full-alphabet D2. Thus the comparison concerns **sparse encoders with a common decoder**, rather than
a separate sparse decoder. Shaped polar is better on these means, but the stronger conclusion is that its performance
is close to dense while keeping coding structure. The sparse/hash learned arms do not show a large enough practical
advantage to justify prioritizing their still-unsolved inversion problem. Different joint-learning parameterizations
and three seeds limit model-class conclusions.

At B14, the polar energy bank and hash J8 amplitude bank each contain 8,192 learned real values (256x32 and 32x256).
Their comparable sizes make the polar result especially useful as a structured reference. Their parameterizations
and inference access differ; this equality alone does not make their model classes or optimization equivalent.

## The ODMA point needs careful wording

Polar and placement are separate: payload bits become a coded sign vector, and a placement map inserts that vector
into the channel resources. Their roles are different, so keeping polar does not logically require retaining the
original placement design forever. However, the retained learned-polar tests kept these placements fixed. They learned
symbol energies; the failed mixer mixed symbols before placement. Neither removed nor learned the placement map.

The very early poor ODMA result involved a different, heavily reused mask design. It cannot establish that the later
ODMA-polar placement structure is intrinsically harmful. The supported statement is: **keep polar, question placement
separately if a diagnosed limitation warrants it**. Removing or learning ODMA is an untested possibility, deferred from
the current fixed-encoder receiver study.

## What decoder research means now

The current model freezes polar/CRC, placements and deterministic balanced energies. It studies the joint posterior,
message multiplicities and the final PUPE decision, then asks which calculations can be approximated with controlled
error and acceptable cost. This stage is algebra/specification work; it has not yet established a scalable receiver.
Polar's efficient isolated-codeword decoder does not by itself solve multiuser interference or message association.

The later paired diagnostic at B14/K26/10 dB uses fresh frames and must remain separate from production rankings:
dense D2 0.09981 -> refined 0.04457; learned-energy polar 0.09250 -> 0.04562; balanced polar 0.08964 -> 0.04344.
Refinement uses observation-only likelihood-improving swaps, with known K. It is non-scalable and does not certify ML
or Bayes-optimal PUPE. Its role is to show substantial unused inference headroom on unchanged encoders.

## Comparison scope and reproduction

Production panels use real AWGN, n256, known K/noise, distinct messages, three seeds, and 256 frames per cell/seed.
The full grid is K=7/15/22/26 and Eb/N0=-4:2:10 dB. Historical small-B native implementations are adaptations, not
paper-scale reproductions. All selected constructions have unit-energy waveforms. Distinct sampling avoids the later
historical IID membership-versus-multiplicity metric issue. Panel C uses 512 fresh frames per seed and the stored
post-hoc diagnostic. No new simulation, training, or approximation ranking was performed for these figures.

`plot_data.json` records the selected numbers and hashes of all 48 production summary sources plus the diagnostic
source. `build_figures.py` verifies completion, exact cell grids, frame-level means and output budgets before plotting.
Run it with `uv run python output/pdf/research_direction/build_figures.py`. The handout builder uses ReportLab and pypdf
from the bundled document runtime. The current research direction is owned by `docs/CURRENT_STATE.md`;
Jobs 033/034 audits and the 10 October gap investigation own the historical conclusions.
