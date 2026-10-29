# Current state: polar multiuser decoding

Updated **29 October 2026**. This page owns the current research direction. The
[experiment bank](EXPERIMENT_BANK.md) owns script/setup/result mappings; the reports below own the
derivations, computational accounting and retained lessons.
[Report 7](reports/07_polar_decoding_from_posterior.pdf)
contains the exact Bayesian foundations, with adjacent [source](reports/07_polar_decoding_from_posterior.tex).
[Report 9](reports/09_polar_bayesian_approximations.pdf), with [source](reports/09_polar_bayesian_approximations.tex),
separately owns algebraically motivated assumptions and their error bounds. Report 8 defines the fixed encoder.
[Report 10](reports/10_polar_decoder_designs.pdf), with [source](reports/10_polar_decoder_designs.tex),
owns the mathematical decoder catalogue; implementation evidence is indexed in the experiment bank.
[Report 11](reports/11_tractable_polar_decoding.pdf), with [source](reports/11_tractable_polar_decoding.tex),
owns the separate investigation of tractable approximations for the same encoder and loss.
The [tractability log](POLAR_TRACTABILITY_LOG.md) records exploratory reasoning, checks and implementation decisions.

## Problem and fixed contract

Develop the strongest practical nonlearned receiver for the fixed ODMA–polar/CRC construction, aiming
for a substantial PUPE improvement over D2. The current priority is to reduce payload-dependent computation
while retaining useful accuracy, with any accuracy concessions explicit. Unknown K/other latent parameters
and learned internal approximations remain later work. The receiver knows the encoder,
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
identifiability limits and decision certificates, with their computational accounting.
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
5. **Record the algebra clearly.** Report 10 retains complete decoder constructions, Report 9 the
   approximation algebra and Report 7 the exact foundations. Report 11 develops tractable approximations;
   the experiment bank indexes checks and measured results. Compare choices through their mathematics and computational
   effect, and distinguish empirical finite-budget prioritization from a theorem about an entire family.

**Current authorization, 29 October:** continue algebra-first, nonlearned tractability research using
sub-agents, seeking structural combinations of the exact and approximate observations. Retain a progress
log plus the concise shared-style report. Implement the current tractable routes as new numbered decoders
and run correctness/smoke checks. Do not create HPC job scripts yet. The author-owned paper outline is
outside this task.

**Implementation history:** after the first returned Job035 results, the user requested a broader round
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

**Current evidence, 26 October:** **160/180 original Job035 children** remain certified complete, with
unchanged hashes for their previously audited 1,310,720 records. Fourteen further children have full
evaluations but failed the source-change guard; six are absent locally. The three targeted balanced B14
DEC-05 retries contain 6,656/6,912/6,912 frames (83.3% of the total), all reproducing the old counts exactly;
none has completed yet.
Keep failed originals and fresh retries separate. `c19c377` enables current-main retries with unchanged
experiment hashes: rows50/65/80, balanced filter, fresh output root, audited current-source control reuse.

**Combination evidence:** **19 completed Job036 rows**, four partial. All eleven new variants have B12
seed3201 results; DEC-14–17 also have seed3202, and DEC-14 has seed3203 plus B14 seed3201. The snapshot
now also completes B14 DEC-15/16 seed3201 K7/15. It independently verifies 524,288 records and all selected
input archives; all 177 previously complete runs retain their frame/summary hashes. Both Job036 revisions
(`743e81c`, `c19c377`) share identical experiment hashes and pass the old/new shared-definition audit.

DEC-14's B12 improvement replicates: K26/4 dB three-seed PUPE .003105 versus D2 .025190, old beam
.069611, guided IS .006010 and SMC .014774. Recorded time is .211 s/frame versus D2 .022 and guided IS
8.53. At B14/K26/4 dB, **one matched seed only**, DEC-14 gives .061298 versus D2 .403996 and SMC
.268930, at .440 s/frame versus SMC 12.03. B14 beam replication is pending. The prior 16-frame replay
showed one distinct multiset in the old 32-tuple beam versus 32 in the new beam; deduplication and
multinomial prior restoration changed together, so this is not an isolated ablation.

Mixtures add occasional corrections: on two B12 seeds at K26/4 dB, DEC-17 gives .002028 versus beam
.003080, at about 43 times the recorded cost; at 6 dB it misses one copy versus thirteen. Two-seed
rare-error rankings remain uncertain. Beam mixtures are worse at low SNR, so correction is not a uniform
upgrade. DEC-15/product IS and DEC-20/product SMC do not yet justify default inclusion. Standalone product
still beats beam at B14/K26/0 dB. DEC-24/pair-refined SMC gives .009165 versus original SMC .014724 at
B12/K26/4 dB on seed3201; the paired frame interval includes zero. Keep pair refinement as a focused
control, not a proven gain. Guided IS remains better than beam in the B12 high-SNR residual-error regime.
The new, still-provisional B14/K26/4 dB DEC-16 cell gives .059645 versus beam .061298, at 35.75 versus
.440 s/frame: eleven fewer missed copies, about 81 times the recorded cost. Its paired frame interval
includes zero. DEC-15 gives .240835; DEC-17 has not reached K26. No B14 terminal-refinement result is
returned. All finite physical excess-risk certificates remain vacuous.

Current inventory, matched tables and decision boundaries:
`jobs/036_polar_combinations/results_combinations/analysis_2026-10-26/observations.md`.
The 25 October analysis retains the unchanged completed B12 tables and four plots.

**Pending experiment decisions:** finish the informative parts of Job036 and the three DEC-05 retries. The original array
has 81 rows/540,672 evaluations; the reduced plan discussed on 24 October retains 34 rows/237,568
evaluations. Against that plan, 19 are complete, four partial and 11 not returned locally (not a live
scheduler audit): 62.1% of planned evaluations are certified and 67.2% have returned. Prioritize B14 beam
replication, certified guided-IS controls and hard-load refinement before selecting the final correction
combination. There is enough evidence to begin a focused beam-width/compute study and design cheaper
targeted correction while those results arrive; another broad sweep need not precede that work. Follow
selection with fresh-seed validation. No new decoder or cancellation was made in this analysis.
Keep runtime sources stable while jobs use them. Results reuse original seeds
and assume known K/noise plus an explicit alphabet; no scalable large-payload claim follows.

**Scalability clarification, 26 October:** the user identifies practical payload scaling as the next
essential gate. DEC-01–24 still depend on the full M=2^B alphabet. Multiset beam's main scoring cost is
O(K W n M); IS/SMC replace W by their sample/particle budgets. D2 already uses O(Nc M) factorized
operators per layer, but retains global M-dimensional state. B12/B14 keep Bc=6 and Nc=32, so they do
not test longer polar payloads. Preserve these receivers as small-B accuracy references; prioritize a
polar-aware non-exhaustive proposer, procedural encoding and sparse count inference alongside the
limited beam-budget study. Report 7's conditional LLRs permit polar candidate generation, but SCL
lists are not exact conditional samples or partition functions. A practical design must bound the
number of polar decoding calls and account for candidate misses and actual proposal densities.
Full code-derived costs, paper comparison and proposed transition:
`jobs/036_polar_combinations/results_combinations/analysis_2026-10-26/scaling.md`.

**Tractability investigation, first installment:** Report 11 extends Report 9's character algebra with
a normalized independent-parity proposal and a computable partition upper bound. It permits exact
one-slot rejection correction without an exact polar partition oracle. A realizable dependent-parity
triangle makes its acceptance exponentially small, so this is a conditional route with a concrete
failure case, not an established practical receiver. Retaining small groups of dependent parity factors
is one open repair; SCL-list proposals and bounded candidate inference remain alternatives.

The report also separates discovery, interaction and output costs. Weighted configurations permit
sparse count-tail storage bounded by their total transmitted-copy count, rather than M times K.
Restricting only the output action has an expected-omitted-copy-fraction bound; conditioning the entire
posterior on a candidate set requires a different error account. Existing revisable-sampling prototypes
already demonstrate several proposal/correction interfaces on tiny cases; they do not establish useful
mixing or large-payload performance. The experiment bank indexes tiny algebra checks and their limitations.
Continue by resolving proposal coverage, correction acceptance and repeated residual-context costs
before choosing an implementation. The runtime sources and HPC experiments are unchanged by this stage.

**Second tractability pass, 27 October:** an explored-set repair replaces an envelope by exact physical
weights on selected states and retains a normalized sampler for its complement. A product-coordinate
exclusion trie avoids a hidden rejection cost when most reference mass is already explored. Reliable-basis
flip search supplies one finite candidate rule; existing SCL lists can use the same repair. Correcting one
cell repairs the earlier triangle's acceptance failure, although that cell contains only about one third
of its target mass. This is a sampling improvement, not a claim that the explored list is sufficient.
Repeated conflicts still defeat fixed global flip radius; exact component factorization repairs that
particular control, but structural checks find no such small components automatically in the tested codes.

Separately, score-variation bounds transfer conditional list coverage and envelope correction to nearby
residuals, permitting reuse with an explicit deterioration bound. Repair and reuse can combine through the
same pointwise envelope. Their total preparation, pattern-bank cost and global multiuser mixing remain
unresolved. The report and three tiny algebra scripts now retain these constructions and counterexamples;
there is no new end-to-end PUPE result or production decoder.

**Editorial revision, 27 October:** Report 11 now has adjacent source and PDF directly in `docs/reports/`,
uses the shared report style, and opens with the research aim. The concise rewrite retains the derivations
and counterexamples. Its three algebra checks live in `tests/`; reproduction commands are in the experiment bank.

**Third tractability pass, 27 October:** Report 11 now derives a bounded same-pattern pair proposer using
the established four-symbol polar MAC construction, with explicit list, screening and repeated-context costs.
An alternative XOR-first search exposes a code-consistency factor; a tiny counterexample reverses the
uncorrected guide's preferred difference. Ordered-pair normalization and MH preserve the IID count target.
The fourth algebra script checks these identities and finite balance, not joint SCL performance.
Mixed-pattern discovery, useful list widths and complete-receiver cost remain open. Existing literature
is now identified in the report; deriving an operation here is not itself a novelty claim. Small conditional
benchmarks can proceed alongside further algebra, before selecting a complete scalable receiver.

**Fourth tractability pass and initial implementation, 29 October:** conditioning selected overlap
parities separates a fixed-pattern multiuser block into affine one-user laws. Shared rows across neighbours
are charged once per slot; the sector count is 2^R for total conditioned rank R. A rank budget gives
explicit omitted-interaction and conditional count-loss bounds, or the existing envelopes permit exact
correction. A three-user physical control, a greedy-selection counterexample and a B12 pattern-pair census
verify the algebra. Small useful ranks, inner acceptance and pattern discovery remain unproved; generic
cutset conditioning is established prior art. This operation is not yet a production decoder.

DEC-25–32 now implement SCL/basis multiset beams, independent/repaired/cached parity MH, joint/XOR pair
MH and SCL-list MH. They accept a procedural PolarScenario and return sparse integer-ID counts without
an M axis. The existing paired harness also accepts them through an explicit small-B adapter. Independent
checks establish normalization, physical scores, detailed balance and exhaustive tiny REF endpoints;
bounded B100 execution at K1/2 is not a multiuser performance claim. Smoke outcomes and commands belong
to the experiment bank and tractability log. No HPC scripts were changed.

**Whole-decoder bound and matched campaign, 29 October:** Report 11 now combines a common tilted parity
product with an orbit-corrected multiset beam into DEC-33. Completing the Gaussian square gives a full
configuration envelope. Independent weighted count tails then give an expected excess-PUPE bound;
the common normalization cancels from the decision comparison. The single-user unit-energy factor
tightens this bound. The current multiuser bound is still vacuous in the smoke: it is not a useful PUPE
prediction or a certificate for finite MH chains. DEC-34 unions SCL and basis candidates before pruning.

The bank now has ten procedural variants, DEC-25–34. 81 focused tests pass, including the historical
job guards, new whole-tuple normalization/envelope checks and direct B100 K1/2 execution controls.
Matched basis initialization and larger search budgets remove most of the earlier B12 smoke deficit:
at K22/4dB, four frames, DEC25 misses one copy and the other nine tractable receivers miss none.
The chain initializers already have zero PUPE and most chains do not move, so these outcomes establish
neither an improvement from correction nor good mixing. Hard B14 discovery remains unresolved.

[Job037](../jobs/037_polar_tractable/README.md) is configured locally: 60 CPU jobs, cap4, exactly the
balanced Job-035/036 physical grid, with pinned historical D2/ODMA and selected explicit controls.
Real historical B12/seed3201 controls have been merged successfully through the strict new audit.
New HPC jobs have not been submitted. The experiment bank and tractability log retain commands,
budget differences, timing extrapolations and next research gates. Report 11 remains in the flat report
directory with shared styling; its source was compiled in the existing editor, without a separate PDF export.

## Deferred scope

Further encoder learning/placement search, the retired shared mixer, hash/amplitude searches and sectioned
association work are outside the current decoder study. Historical code, manifests and Reports 1–6 remain
available. Unknown K, fading and false-alarm-aware output losses require new contracts. No current evidence
establishes dense optimality, paper-scale reproduction or useful multiuser PUPE at B100.
