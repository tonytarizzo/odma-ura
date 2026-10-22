# Current state: polar multiuser decoding

Updated **22 October 2026**. This page owns the current research direction. The
[experiment bank](EXPERIMENT_BANK.md) owns script/setup/result mappings; the
[exact log](POLAR_DECODING_RESEARCH_LOG.md), [approximation log](POLAR_APPROXIMATION_LOG.md) and
[decoder design log](POLAR_DECODER_DESIGN_LOG.md) own the reasoning and retained lessons.
[Report 7](reports/07_polar_decoding_from_posterior.pdf)
contains the exact Bayesian foundations, with adjacent [source](reports/07_polar_decoding_from_posterior.tex).
[Report 9](reports/09_polar_bayesian_approximations.pdf), with [source](reports/09_polar_bayesian_approximations.tex),
separately owns algebraically motivated assumptions and their error bounds. Report 8 defines the fixed encoder.
[Report 10](reports/10_polar_decoder_designs.pdf), with [source](reports/10_polar_decoder_designs.tex),
owns the mathematical decoder catalogue; implementation evidence is recorded in the design log.

## Problem and fixed contract

Develop the strongest practical nonlearned receiver for the fixed ODMA–polar/CRC construction, aiming
for a substantial PUPE improvement over D2. Accuracy is the immediate priority; unknown K/other latent
parameters and learned internal approximations for efficiency follow after establishing this foundation. The receiver knows the encoder,
active-user count K and physical noise variance; messages and their placement patterns are unknown.

\[
w=(p,v),\quad \phi(p,v)=S_p\operatorname{diag}(\sqrt{e_p})(-1)^{Lv},\qquad
y=\sum_{k=1}^K\phi(w_k)+z,\quad z\sim\mathcal N(0,\sigma^2I_n).
\]

Here L is the Nc-by-Bc binary CRC/polar generator, S_p inserts Nc symbols into n resources,
P=2^Bp, V=2^Bc and M=PV=2^B. Every column has unit energy. Fix the code, placements and deterministic
KL-balanced energies from [Report 8](reports/08_polar_energy_geometry.pdf). Balancing is a specified
construction, not a certificate of optimal PUPE. Its feasibility/convergence must hold for each fixture.
The selected research banks use B=12/14, Bc=6, Nc=32, CRC8 and n=256.

Start with IID uniform full messages. Distinct-message sampling is a separate, explicitly conditioned control;
same-pattern collisions remain possible under both. The intended loss matches transmitted copies:

\[
\ell(a,\widehat a)=1-\frac1K\sum_w\min(a_w,\widehat a_w),\qquad \sum_w\widehat a_w\leq K.
\]

The Bayes action allocates K copies using posterior tails Pr(a_w >= j | y). MAP, posterior means and
membership-only lists generally give different decisions. The [metric record](PUPE_METRIC.md) explains why
historical IID membership scores cannot be relabeled as multiplicity recovery. Distinct-message matched
comparisons retain their PUPE where the evaluated output support/budget is unchanged.

## Where the research stands

**The broad exact-derivation phase is complete for this contract.** Report 7 derives a complete finite
Bayes decoder and alternative exact evaluations, lossless state reductions, polar pair/three-user contractions,
identifiability limits and decision certificates. The exact log records their computational accounting.
This is a sufficient foundation for approximation design; a general K4 expansion or another exact survey
is not a prerequisite. Targeted exact work remains useful when a proposed approximation needs it.

**No derived exact construction is established as tractable at the intended message length and load.**
Count aggregation leaves roughly M^K/K! hypotheses at fixed K. Walsh/parity operators save real work but
retain explicit alphabet or pattern dependence; small geometric interfaces need not have small state tables.
There are concrete obstructions to simple exact merging, without a universal impossibility theorem.
Compact encoding, memory savings and complete scalable decoding are distinct achievements.

**The broad approximation preparation is complete for the retained paths.**
Report 9 develops shared posterior/decision error bounds and conditional reductions at the exact bottlenecks:
support restriction, continuation quantization, regional/feature/connected interactions, residual moments,
Fourier work, weighted sampling and decision-directed prefix queries. Its completion audit records finite
constructions, source/count handling, error propagation and costs for entries A–T. This is not a claim
that every useful approximation has been discovered or that the stated regimes hold for the selected bank.
Three finite compositions show how the ingredients connect; these remain useful illustrations:

1. **Residual-guided independent importance decoding.** Conditional polar moments guide two child queries
   per message bit. Ordered-slot importance weights recover the exact factorial count prior; sparse count-tail
   readout avoids an explicit full-message array. A whole-configuration prior mixture gives a valid weight
   bound. Pattern-moment work and importance concentration can still be prohibitive.
2. **Deterministic prefix refinement.** Moment, spectral or other certified query providers bound the gain
   available inside unresolved prefixes. Refine competitive cells and output actual full-message counts
   with an excess-loss bound. Useful interval width and visited-prefix count are unproved.
3. **Positive structured contraction.** Regional states and positive features retain total K and message
   identity. Conditioning selected feature/state indices extends the forest calculation to cycles.
   Optional importance or control-variate correction targets the original supported law. Local-state,
   alphabet and joint-separator construction can still be exponential.

The guided sampler also has a corrected block-chain variant: exact K2/K3 conditional updates retain all
outside-user effects in the residual, and a fixed global proposal supplies a conservative finite-chain
bound. The proof does not establish an advantage over independent sampling. Observation-weighted correction
moments can sharpen worst-case errors, provided their own computation is charged. A factorial-moment
hierarchy bounds multiplicity truncation; means alone are justified only with posterior collision control.
These are conditional constructions, not validated favourable regimes or a selected receiver.
The previous assumption menus and exploratory PUPE tables remain retired; their scripts are parked.

The preceding pass added three further deductions. **Exponential tilting** moves a residual closure toward the
queried observation, gives relative-error and positive evidence bounds, and connects to the exact support
dual; nonuniform polar partition/moment access can still be expensive. **Positive Gaussian auxiliary fields**
give an exact conditional-product representation for the IID source, controlled field-domain restriction and
direction removal. Naive Gaussian-field sampling can nevertheless have exponentially large variance, even
at K=2. **Finite moment optimization and prefix-count consistency** tighten tail/gain bounds from information
already computed, retaining the common evidence and total K. Mixed moments can add selected cross-cell
information. None establishes a favourable regime for the current bank.

The completion pass closes their concrete construction dependencies. **Conditional polar character
polynomials** provide finite tilted partition/moment and Fourier queries, including positive weights and
nested error bounds. A positive likelihood-polynomial hierarchy converges without assuming Gaussian
interference, with potentially prohibitive degree/tensor costs. **Field geometry** connects discarded
directions to pairwise dispersion and gives a bounded Gaussian proposal in a stated global-curvature
regime; a finite-ball alternative remains available outside it. **A variational common center** has a
unique optimum within the identical-slot product family, a finite optimization procedure and an
evidence-gap certificate, but exact optimization alone need not yield accurate inference.
**Signed low-rank contractions** can supply marked queries through explicit positive error envelopes,
whose own contraction cost is charged.

Certified child intervals now define a minimax branch proposal with a precise all-prefix concentration
bound. A finite moment counterexample shows that exact low-order moments and count consistency can still
leave tails ambiguous. The optional distinct-source residual comparison uses its correct collision and
source factors; a small prior collision probability is not a posterior target-equivalence guarantee.
The final error account separates support loss, target approximation and numerical uncertainty, and removes
guide bias only when the original target is exactly reweighted. These results complete the mathematical
interfaces without claiming tight bounds, cheap pattern discovery or a scalable receiver.

**Report 10 supplies the decoder-option catalogue; all thirteen families now have finite implementations.** It assembles
one exact reference family and thirteen execution families: direct residual queries, a variational product
law, message-prefix refinement, bounded configuration search, independent importance sampling, corrected
block/global chains, sequential particles, quantized continuation states, positive factor contraction,
connected expansions, Fourier queries, signed low-rank contraction and positive auxiliary fields.
The DEC-01–DEC-13 identifiers are separate from historical D0–D4. Shared providers and corrections are
variations, not an unbounded list of supposedly different receivers.

Each family closes message discovery or explicit traversal, legal inference, count-tail readout,
allocation, stopping and cost/error accounting. New derivations show that an identical-slot product
decoder only needs its highest min(K,M) word probabilities, that auxiliary-field proposals can be corrected
on their joint space without an unavailable marginal tuple density, and that independent prefix/tempered
particle batches supply unbiased unnormalized count queries with finite confidence. These facts do not
establish useful concentration, cheap construction or a preferred architecture. All Report-9 entries
A–T and the major Report-7 exact resources are mapped to their decoder roles.

## Why this encoder and receiver problem

Jobs [033](../jobs/033_matched_d2/README.md) and [034](../jobs/034_polar_mapping/README.md) completed
66/66 and 12/12 rows in their recorded audits. In the B14/K26/10 dB distinct-message cell, native ODMA–polar
PUPE was .55874, the same codebook with D2 gave .13712, learned polar energies/D2 gave .10617 and fixed
dense/D2 gave .09831. These measured comparisons motivate receiver work; dense/D2 is a comparator, not an optimum.
Job035 now adds matched balancing comparisons across receivers; its benefit depends on load, SNR and receiver.
D2 remains an explicit-M effective-channel approximation, with no derived AMP/state-evolution guarantee.

## Agreed approach: algebra within the Bayes decoder route

1. **Locate the exact bottleneck.** Start from an exact residual, convolution, interaction, elimination,
   Fourier, connected or decision-certificate equation. Name the object whose size/work causes the scaling problem.
2. **Use exact discoveries before assuming anything.** Character/parity identities, compatible projections,
   reference changes, marked contractions and support bounds may expose a simpler remaining problem.
3. **Derive a precise assumption and replacement.** State its parameter/regime, retained constraints,
   discarded information and complete normalized calculation. The physical model, source law and Bayes target
   remain fixed; MAP or residual minimization does not replace posterior count-tail inference.
4. **Derive accuracy and full cost together.** Seek error bounds, exact error identities, limiting cases and
   counterexamples. State unproved conditions explicitly, including the cost of obtaining a certificate.
   Include construction, pattern discovery, all count/message association, memory and numerical precision.
5. **Record the algebra clearly.** Decoder-composition questions and deductions enter the decoder design
   log; settled constructions enter Report 10. Report 9 and its log retain approximation algebra, while
   Report 7 keeps the exact foundations. Compare choices through their mathematics and computational
   effect, and distinguish empirical finite-budget prioritization from a theorem about an entire family.

**Current authorization:** after the first returned Job035 results, the user requested a broader round
of combinations, smoke/correctness checks, Job036, and a push to main. DEC-14–24 now combine multiset
beam discovery, product/Gaussian proposals, importance correction, SMC and terminal Gibbs/guided-pair
Metropolis moves. Do not narrow immediately to only two architectures. The detailed comparison matrix
and derivation are in the [036 README](../jobs/036_polar_combinations/README.md). Encoder learning remains
outside scope. The user explicitly requested reuse of matching D2/ODMA results rather than retraining.

**Active harness:** [tests/polar_decoder_benchmark.py](../tests/polar_decoder_benchmark.py) calls the
[decoder bank](../framework/polar_decoders.py) through the shared count/metadata interface.
REF and DEC-01 through DEC-13 are implemented, alongside existing D2 and ODMA controls. Each family
has a declared finite variant, not every possible Report-10 combination. All use explicit alphabets;
DEC-02/03/10/11/13 support IID only. Candidate-restricted paths and vacuous certificates are labelled.
The [035 README](../jobs/035_polar_decoders/README.md) owns commands, exact parameters, output layout,
comparison limits and B12/B14 CPU estimates. Full frames and count outputs, posterior regret on tiny
cases, PUPE uncertainty, setup/training/inference costs and both reference bounds are retained.
Uniform ODMA is unchanged; its balanced-energy version is an explicitly labelled adaptation.

**Current evidence:** the 22 October local snapshot has 142/180 complete Job035 children. B12 guided
sampling is promising, but rankings depend on load/SNR; B14 product and beam methods can outperform
D2 while finite MCMC has a provisional deficit at K22. Some weak variants lose primarily at candidate
or configuration discovery, not at their downstream contraction. This is finite-budget empirical
selection, not a refutation of the underlying exact algebra. All reported physical excess-risk
certificates remain vacuous. Keep product, beam, guided IS, SMC and invariant refinement active;
deprioritize standalone Gaussian/prefix/quantized/regional/connected/Fourier/cut/field variants for now.

**Next step:** queue the prepared **81-row Job036** array: eleven new combinations, B12/14, balanced
encoder, three original seeds, four K values, eight SNRs, 256 frames/cell. Heavy B14 importance runs
are split by load; total new work is 540,672 frame-decoder evaluations. Four concurrent two-CPU jobs
reserve eight CPUs/64 GB per array. D2, ODMA and matching DEC-02/04/05/06/07 controls are reused from
Job035 through a deliberately source-audited merge. Job036 is not submitted by the agent. Use a separate
checkout while Job035 is active, so its source fingerprints remain stable. Retaining original seeds
saves control computation but is not held-out validation; evaluate selected winners on fresh seeds later.

## Deferred scope

Further encoder learning/placement search, the retired shared mixer, hash/amplitude searches and sectioned
association work are outside the current decoder study. Historical code, manifests and Reports 1–6 remain
available. Unknown K, fading and false-alarm-aware output losses require new contracts. No current evidence
establishes dense optimality, paper-scale reproduction or complete B100 decoding.
