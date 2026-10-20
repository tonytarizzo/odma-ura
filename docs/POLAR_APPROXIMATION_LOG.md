# Polar decoding research log: approximation algebra

**16 October 2026 — research reset.** This is the working record for the separate
[Bayesian approximation report](reports/09_polar_bayesian_approximations.pdf)
([source](reports/09_polar_bayesian_approximations.tex)). The
[current state](CURRENT_STATE.md) owns the agreed direction. The
[exact report](reports/07_polar_decoding_from_posterior.pdf) and [exact log](POLAR_DECODING_RESEARCH_LOG.md)
retain the mathematical resources already developed.

## Agreed scope

We are following the Bayes decoder route. Counts, residual recursion, positive convolution, exact elimination,
polar pair/three-user contractions, Fourier sums, connected expansions and certified search are different
calculations of the same target. Their expensive operations identify where an assumption could help.
An assumption must be algebraically motivated and state both its computational effect and the accuracy
claim or unresolved condition. The source law, known K, fixed balanced encoder and count-tail Bayes loss remain fixed.

No decoder implementation, new tests, tuning, simulations or PUPE comparisons are part of this stage.
No assumption is preferred because a previous tiny-panel result was favourable. Existing exact discoveries
may be reused when they help a derivation, without requiring every one to appear in every construction.

## What was cleared

The previous approximation catalogue, forest/beam/sampling/feature recipes, branch rankings, exploratory
numerical tables and associated recommendations were removed from the active log and experiment bank.
The report's earlier candidate/BP/Gaussian/mean-field/MAP routes were removed from the exact narrative.
Their existence does not define the new search. Scripts and existing outputs remain unchanged and are
listed briefly as parked work in the [experiment bank](EXPERIMENT_BANK.md#parked-approximation-scripts).
Useful ideas can be derived again in the appropriate exact context during the new investigation.

## Foundation now in Report 9

The initial report contains the fixed posterior/decision, exact-formulation bottleneck map and a common
record for each future assumption: exact origin, algebraic motivation, replacement, accuracy, full cost
and dependencies. It also retains reusable mathematical error statements:

- A positive log-weight-ratio range Omega gives TV <= tanh(Omega/4).
- Restriction with omitted posterior mass eta and conditional TV error tau gives total TV <= eta+(1-eta)tau.
- Uniform count-tail error e gives excess posterior Bayes loss <= min(1,2e), for an action maximizing the
  approximate feasible gain. A sufficient selected/unselected margin can certify the exact action.
- Directional future-sum widths quantify continuation log-ratio errors after exact scalar absorption.
- Signed contractions require separate numerator and positive-normalizer control; they are not automatically
  approximate probability laws.

These are shared bounds and identities, not assumptions that the polar bank satisfies a favourable regime.
Their practical usefulness and the cost of their controlling quantities remain questions for each derivation.

## Investigation entries

### 16 October: first bottleneck derivations

The existing foundation survives a direct algebra review. It is extended with three distinct replacements,
without choosing a decoder or importing the former empirical rankings:

1. Restrict a positive count search using disjoint omitted-subtree mass bounds. Separate individual bad
   likelihood from total omitted evidence; retain the cost of discovering and bounding those subtrees.
2. Quantize continuation rows after absorbing their exact scalar differences. Derive both a covering-number
   bound and a sequential recurrence whose likelihood errors telescope. A bound at one cut alone does not
   justify a full recursive decoder. Preserve the original count paths and their readout through the merged graph.
3. Remove residual interactions between disjoint message regions while retaining exact source weights,
   cardinality and local count configurations. Work out the contraction cost and the error of omitted edges.
   Region-dependent centering needs care: unknown occupancies can reintroduce the coupling being removed.

The common geometric quantity is the four-point log interaction, invariant under unary rescaling. The new
derivation bounds how accurate a single separable factor can be uniformly on a feasible rectangle; it does
not give a lower bound on posterior or Bayes loss. Nonuniform positive error envelopes propagate to count-tail
and decision bounds, so one worst-case configuration need not automatically control every accuracy statement.

#### Settled algebra now in Report 9: invariant, Entries A–C and composition

- **A, restriction:** for disjoint omitted subtrees with total evidence upper bound U_D and retained evidence
  lower bound L_E, omitted posterior mass is at most U_D/(L_E+U_D). A likelihood gap needs to beat the
  logarithm of the total competing multiplicity; poor individual explanations are insufficient grounds to prune.
  Cost is parameterized by visited nodes J and retained count states F, including discovery and bound construction.
- **B, continuation quantization:** midpoint scalar absorption bounds each row's log error by its directional
  future width divided by 2 sigma^2. A coordinate cover gives a product bound on R, exposing dependence on
  future dimension, noise and accuracy. A full sequential count recurrence is derived: its scalar corrections
  telescope, so errors add along paths without reusing the Gaussian observation. No original legal count path
  is removed. With R representatives per cut/occupancy it has O(M K^2 R) transitions; a reverse marked pass
  returns all count probabilities. This remains explicit in M and can require large R. Cover ranges must include
  approximate incoming states, including quantization drift, not just original exact sums.
- **C, regional interactions:** disjoint message regions retain exact local count priors and total K. With a
  single common center, deleting cross-region factors gives absolute log error bounded by the sum of
  gamma_rq k_r k_q over omitted edges. A forest surrogate has an explicit polynomial count-message recurrence,
  with work O(K sum_edges D_r D_q + K^2 sum_r (1+degree_r) D_r). Constructing local states, edge tables,
  omitted-edge certificates and the alphabet are separate costs. Local count caps are support restriction A.
- **Composition:** common-support log errors add only in a compatible representation. If E_* bounds their
  total and eta bounds omitted posterior mass, excess posterior Bayes loss is at most
  min(1, 2[eta+(1-eta)tanh(E_*/2)]). Marked positive lower/upper sums can instead certify individual tails or
  exchange signs. Their own contraction cost must be counted; an error envelope can restore dense coupling.

#### Corrections and limitations discovered while deriving

1. Independent regional centers are not free unary improvements: the shifted observation becomes
   y minus sum_r k_r c_r, depending jointly on unknown occupancies. Exact sector conditioning can introduce
   binomial(G+K-1,K) sectors. Retaining the occupancy factors is another option, with its own separator cost.
2. A small cover at one cut is not a full recursive complexity guarantee. A uniform total error budget over
   M cuts may require local tolerance proportional to 1/M, making the cover much larger.
3. A large four-point interaction obstructs uniformly accurate rank-one replacement on that rectangle,
   not every low-rank approximation and not necessarily the Bayes decision. Low-mass and decision-specific
   cancellations remain possible; hard source constraints and global cardinality identities are retained.
4. Claims are conditional algebra in real arithmetic. No favourable parameter regime is established for
   the research bank, and no polynomial-in-B,K receiver has been obtained. No decoder code or numerical
   experiment was added or run; only the report is compiled and visually checked.

#### Next distinct questions, without selecting a winner

Investigate positive multi-feature boundary representations beyond row grouping; marked Fourier and
connected remainders; then residual-distribution closures and finite sampling error. The three present
entries also need affordable prefix discovery, useful covering dimension, cheap directional bounds and
sparse interactions with controlled aggregate omission. These are unresolved conditions, not a plan to
start implementation. All earlier empirical branch rankings remain parked.

### 17 October: deeper replacements, without an assumption atlas

The next pass starts from the remaining costly objects, keeping the algebra-only scope. Working questions:

- Can the sequential cover avoid charging an approximation at all M positions when a legal path has at
  most K occupied words? Investigate a fixed grid within each occupancy and identity transitions for zero counts.
- Can an exponential cross factor use several positive separable features, with an explicit rank/error
  relationship, instead of either keeping the whole table or deleting the interaction?
- Can connected-user truncation be made into a positive auxiliary-variable restriction, so omitted cluster
  size has a probability meaning? Check the price of making every expansion term nonnegative.
- Can a residual moment closure control the smoothed likelihood and all excluded-word count queries?
  Matching a covariance alone is not a posterior-error guarantee; normalization and the IID source matter.

These are derivation questions, not selected receiver branches. No implementation or numerical receiver
experiment is part of this pass. Settled deductions and failed simplifications will be recorded below.

#### Settled deductions added to Report 9

1. **B can avoid error at absent words.** Use a grid fixed within each occupancy and make count-zero
   transitions exact identity operations. Only positive-count transitions round a signal; their occupancies
   strictly increase. With coordinate spacings h_j and full-word coordinate ranges D_j, the total absolute
   log error is at most K(K-1) sum_j D_j h_j / (8 sigma^2). A drift-inclusive lattice cover is derived.
   This removes the earlier possible M-fold error accumulation, not O(M K^2 R) work or the dimension curse.
2. **D, positive features for cross factors.** First truncate singular directions of the bilinear interaction,
   bounding that log error by the next singular value times the endpoint radii. Shift retained coordinates
   so exp(-alpha dot beta) becomes positive unary factors times exp(X dot Y), with X,Y nonnegative.
   Total-degree L Taylor truncation gives binomial(d+L,L) nonnegative separable features. Its factor ratio
   lies between Pr(Poisson(Lambda) <= L) and 1, where Lambda bounds X dot Y. The log-ratio range adds to
   the geometric remainder. On C's forest, edge cost D_r D_q becomes R_rq(D_r+D_q), with exact cardinality
   polynomials and full count readout retained. Low bilinear rank alone does not ensure small feature rank.
3. **E, positive connected-component restriction.** For IID slots, divide every pair factor by a common
   positive lower bound b0. The common b0^(K choose 2) cancels, while F=b/b0-1 is nonnegative. Expand into
   an auxiliary graph and condition on all components having size <= q. This gives one positive approximate
   count law, including repeated words, assembled from marked connected blocks through q. TV is at most
   the discarded auxiliary-graph probability. A conditional independent-edge/tree union bound gives
   eta <= min(1, binomial(K,q+1)(q+1)^(q-1) p_max^q). A weighted rooted-tree evidence bound can be tighter.
   Its operator is F x = B x / b0 - 1 sum(x), so it directly retains the polar pair contraction, unlike a
   general entrywise absolute remainder kernel. Full coefficient discovery, marked assembly and certificate
   costs are counted; none is free from the apparent small component cutoff.
4. **F, excluded-word residual closure.** Exact count j of w has prior binomial mass times the residual
   likelihood of K-j IID draws excluding w. Replace that sum by its matched Gaussian only inside the
   smoothed likelihood expectation, keeping its determinant. A one-increment-at-a-time second-order Taylor
   argument bounds the likelihood error by third absolute moments; a stronger version uses the Gaussian
   smoothing already accumulated in the replacement sequence. Carry errors into count mass intervals,
   then normalize using event/complement bounds. These per-word approximate count distributions need not
   be one joint posterior, so the valid guarantee is direct tail/Bayes-action error, not full-posterior TV.
5. **Exact encoder resources reduce F's construction cost.** Uniform polar characters give the first two
   moments from zero/equal generator-row classes, without summing all payloads. Excluding a queried word
   is a rank-one covariance correction. Gaussian queries can therefore reuse K base factorizations, with
   conservative dense work O(K n^3 + M K n^2) after moments. Error certification and readout are accounted
   separately. This still visits M words.
6. **Preserve modes before closing their internal variation.** Partition the excluded-word source into groups;
   retain their exact multinomial occupancies and close only the within-group residual sums. The mixture's
   finite smoothing error is the prior-weighted conditional error. Point-mass groups are exact, but too many
   groups restore count enumeration. The same unknown-center/occupancy cost identified in C reappears here.

#### Limits found during this pass

- The fixed-grid schedule needs a cover of its approximate reachable states, including accumulated drift.
  Skipping zero-count merges is compatible precisely because representatives remain valid at the next cut.
- A small geometric singular rank does not bound the rank needed after exponentiation. The product of
  coordinate ranges divided by the physical noise scale controls the positive series degree. Feature indices
  can multiply across cyclic separators; isolated low-rank edges do not make a general graph cheap.
- Positivity is not free in E. For unit-energy IID columns, repeated words force the floor e^(-1/sigma^2);
  most distinct pairs can then have large auxiliary edge probabilities even when their raw correlations
  are small. The tree bound can be vacuous. Hard distinct-message zeros have no positive floor; retaining
  exclusions outside the expansion couples components and invalidates the simple assembly factorization.
- The residual Gaussian bound is a finite smoothing statement, not a generic CLT claim. A +/-1 residual at
  r=1 gives exact likelihood tending to 1/2 at low noise while the moment-matched Gaussian likelihood tends
  to zero. Small absolute errors also need a useful evidence denominator. Distinct sampling would need a
  different without-replacement derivation.
- Group mixture weights are prior weights for a likelihood expectation. They are not posterior guarantees
  for discarding occupancy sectors. Every discarded sector still needs an evidence/mass argument.
- The new constructions make no novelty claim or empirical performance ranking. Fourier-mode truncation
  and finite sampling closures remain open; singular-direction truncation in D is not Fourier truncation.

The report and current handoff now include these deductions. This pass contains no receiver implementation,
test, simulation or tuning run; verification is algebra review plus document compilation/layout inspection.

Document verification: the expanded 22-page PDF compiles with resolved references and no LaTeX box warnings.
All pages were rendered and inspected; the new sections received additional full-page checks. The report
source, log and handoff were checked for whitespace issues. These checks are not receiver validation.

### 17 October, continuation: modes, samples and decision-directed queries

The user asks both to continue the algebraic investigation and to explain its endpoint. This pass examines
three gaps: marked Fourier truncation, finite independent-sampling error, and message-prefix queries that
can bound a Bayes decision without reading every leaf. The aim remains a few complete mathematical decoder
constructions with honest discovery, inference, count-readout and error costs. It is not to exhaust all named
approximations before ever choosing a construction. No implementation or receiver experiment is authorized
by this continuation; the current algebra-only stage remains in force.

Working questions and cautions:

- Fourier damping of the unmarked evidence is not automatically damping of a high-count numerator.
  Derive the excluded-word powers and include the cost of accessing likelihood Fourier coefficients.
- A finite sum of positive weighted legal configurations is a valid approximate posterior. Can confidence
  bounds cover every count tail, including words absent from the samples, without enumerating the alphabet?
  Proposal sampling/evaluation and an importance-weight bound must be explicit; posterior samples are not free.
- A prefix's total count bounds the matching gain attainable by any allocation inside that prefix. Can this
  turn tail-error algebra into a complete refinement/stopping rule? Then derive prefix query providers from
  the moment, Fourier and sampling calculations rather than assuming an oracle.
- The transition out of broad exploration should require complete candidate constructions and falsifiable
  accuracy/cost conditions, not a universal tractability proof or an arbitrarily large assumption bank.

#### Settled deductions in this continuation

1. **G, marked Fourier truncation.** Normalize the finite-grid transform to the original source prior.
   Evidence uses chi^K; an exact count j uses its binomial prior mass times the queried-word phase and
   the excluded characteristic function to power K-j. Derive separate omitted-mode bounds for every count.
   A spectral gap for chi transfers only to min(1,(rho+p)/(1-p)) for the excluded law. High multiplicities
   can lose the evidence's damping. The likelihood coefficient l1 tail and its access cost are separate
   conditions. Parseval gives a cheap but potentially exponential square-root-grid bound. Unit-modulus
   modes are exactly the characters constant on the affine signature support; finite-grid Gaussian
   coefficients cannot simply be assumed to be a smooth continuous Gaussian spectrum.
2. **Continuous spectral alternative.** The physical likelihood is the expectation of exp(i Omega dot
   (y-s)) with Omega Gaussian of covariance sigma^-2 I. It avoids finite-grid coefficient construction.
   Count and prefix markers give bounded signed integrands, so independent frequency samples have explicit
   simultaneous numerator/evidence error. This still needs characteristic sums and a useful denominator;
   it is different from a positive sampled-configuration posterior.
3. **H, independent weighted legal samples.** Specify a drawable/evaluable full-support proposal q and
   certified weight bound W_max. A finite positive weighted count law gives a complete multiset decision.
   A directly computable simultaneous tail certificate covers all MK events, including never-sampled words.
   Sparse readout needs at most SK positive tail entries. The source-prior proposal requires no full M
   enumeration, but may need a prohibitive S. With gamma=W_max/Z, a bounded-variance exponential argument
   gives tail error <= sqrt(2 gamma t/S)+4 gamma t/(3S), provided S>=16 gamma t, with
   t=log(2(J+1)/delta). Its useful sample scale is linear in gamma; gamma is not assumed known cheaply.
4. **Sampling can correct a positive approximation.** A surrogate with weight-ratio range Omega gives
   an importance-concentration bound gamma<=exp(Omega), provided its joint law can actually be sampled
   and evaluated. B's state graph and C/D's forest have that conditional structure after preprocessing.
   This does not remove their preprocessing cost or recover support excluded by A. Exact conditional
   block summation preserves numerator/evidence expectations and decreases their variances, including
   the centered normalized-query numerator; no universal finite-sample ratio-MSE claim is made.
   Prescribed sample rounds with summable failure budgets permit valid adaptive stopping. Correlated or
   arbitrarily reused adaptive samples still need separate dependence arguments.
5. **I, prefix gain certificates.** For any cell C, any h-word allocation inside it matches at most
   min(A_C,h), where A_C is its true aggregate count. Thus prefix count tails give an optimistic gain
   for every possible unseen word allocation. On a disjoint covering frontier, allocate K slots to the
   monotone upper tails to get U_*. A real incumbent's lower gain L gives excess loss <=(U_*-L)/K and
   absolute posterior risk <=1-L/K. Derive an explicit split/refine/stop calculation. No prefix is output
   as though it identified a full message.
6. **Prefix query providers are specified.** The IID prefix-count law conditions on j draws inside and
   K-j outside the cell. F's Gaussian closure and finite independent-increment remainder apply to those
   two conditional laws. Conditional polar characters on affine payload-prefix subspaces give exact
   first/second moments without payload enumeration; pattern averaging remains charged. G replaces the
   single-word phase with a cell characteristic sum. H builds all sampled prefix tails in O(SKB) sparse
   work; a union bound over at most K(2M-1) conceptual queries covers adaptively selected prefixes without
   enumerating them. Worst-case visited prefixes and matrix/query work remain explicit.

#### Mathematical review notes

- An empirical posterior can have poor full-configuration TV and still estimate every required count
  tail accurately. Do not force a full posterior support-mass requirement onto this decision-statistic claim.
- A bounded sampled numerator must be divided only with a certified positive normalizer. Probability
  clipping is placed inside the error bound, not inserted as an invalid chained inequality.
- Midpoints of tightened monotone tail intervals define valid per-word tails. They need not be marginals
  of one fixed-K joint law; the prefix gain proof requires truthful intervals, not that extra assumption.
- Spectral truncation, Monte Carlo confidence and deterministic moment closure have different error types.
  Adaptive cell selection is covered only by simultaneous event bounds; adaptive stopping additionally
  needs a time-uniform construction such as the stated summable round budgets.
- The prefix moment matrices change by more than rank one in general. The O(K n^3) per-cell baseline is
  charged; the cheap leaf downdate from F is not silently extended to arbitrary cells.
- Near-Bayes excess loss alone does not certify good absolute recovery. The incumbent gain supplies a
  separate posterior-risk upper bound, and later source/noise-averaged PUPE remains an empirical question.

#### Endpoint of the broad investigation

The report now distinguishes a complete finite mathematical decoder, a favourable conditional cost/error
regime, and validated target-scale performance. Complete options already include C/D contraction with
optional H correction, I with explicit moment providers, and H with a source-prior proposal. None has a
validated favourable target-scale regime. Broad exploration should end when a few distinguishable complete
constructions have explicit discovery/inference/readout, composable errors and full costs, and the remaining
questions refine those constructions. It should not wait for every possible approximation to be listed or
for a universal polynomial-time guarantee.

The subsequent stage would consolidate specifications, then perform small exact-reference checks of tails,
decisions, cost and bound tightness, followed by controlled scaling/PUPE work. This is a future-stage plan,
not an implementation/test run in this pass. Joint cyclic separators, distribution-sensitive remainders and
dependent sampling remain substantial areas for targeted algebra.

Document verification: the expanded 31-page PDF compiles with all references resolved and no LaTeX box
warnings. All pages were rendered and reviewed, with full-page inspection of the new derivations and
synthesis. Source checks found no duplicate labels, missing reference targets or trailing whitespace.
Two optional text-extraction helpers were unavailable; the bundled PDF libraries and renderer completed
the checks. No receiver implementation, test, simulation or tuning run was performed.

### 17 October, synthesis pass: unused exact resources and complete decoder paths

The user asks to revisit the exact report, combine complementary approximations, and finish this pass with
clear full-decoder paths. The scope remains algebra only. The aim is to close the broad exploratory phase
on explicit constructions, not claim that every mathematical approximation has been exhausted.

Working questions:

- Can D's positive edge features and Report 7's joint separators retain cycles by conditioning feature
  indices, rather than immediately deleting the closing interactions? Keep every endpoint's joint parity
  requirements and total occupancy. Compare conditioning on actual states against conditioning features.
- Can the exact reweighting/decision-contrast identity replace worst-case log ranges with a distribution-
  sensitive certificate? The combined correction is a product; separately harmless edges need not compose.
- Can F/I's conditional moments supply a bitwise proposal for H, avoiding an explicit full-message scan?
  Use the exact residual recursion to derive the proposal. Ordered-slot weights must push forward to the
  right factorial count prior without requiring a sum over permutations of the proposal density.
- Can the exact K2/K3 solvers update small blocks inside a full-K sampler? Retain all outside users in the
  conditional residual. An invariant chain is not a finite-time accuracy guarantee; derive a quantitative
  bound with a globally supported proposal before claiming a complete controlled decoder.
- The exact unexcluded residual gives factorial moments rather than count probabilities. Can bounded
  alternating reconstruction turn that former pitfall into a useful low-multiplicity approximation?
  Derive the omitted-order error and keep posterior collision control distinct from a small prior rate.

Potential endpoints to specify: a deterministic positive contraction; a residual-guided importance decoder;
a corrected block-chain variant; and a query-driven deterministic decoder. Every endpoint must include
discovery, normalized count readout, error accounting and construction cost, with a definite failure mode.

#### Settled deductions and their exact sources

1. **J, positive features can retain cycles.** Report 7's joint-separator warning and D's positive expansion
   combine into a feature-index network. Condition enough feature indices to leave a forest, contract each
   sector with the total-K constraint, and sum unnormalized answers. The added sector count is the product
   of cut-feature ranks. A single closing edge costs its rank; multiple cycles still create a potentially
   exponential joint schedule. State conditioning is an alternative with a different actual cost. Group
   exact joint overlap parities first. Sampling sectors requires evidence weighting, and importance
   correction must evaluate the marginal configuration proposal, not its sampled auxiliary assignment.
2. **K, observation-weighted correction control.** The exact covariance identity yields
   TV(P,Q) <= sqrt(v_R)/2, with v_R = E_Q[R^2]/E_Q[R]^2 - 1, and observable error bounded by
   sqrt(Var_Q(H) v_R). A decision contrast pays only where its two events disagree. The correction is the
   whole product of modified factors. Its moments can be as expensive as the original inference; certified
   envelopes or a separately bounded contraction/sampling calculation are required.
3. **Rare bad regions and control variates.** If Q puts mass q on a bad region and its correction there is
   at most b, while the good region has R >= r_minus, the target bad mass is at most
   q b / ((1-q) r_minus + q b). Conditional range bounds then give a global TV bound without deleting it.
   If the surrogate expectation h=E_Q H is known, exact correction can estimate
   E_Q[(R-c)(H-h)]/E_Q R. A nearly constant R reduces this correction variance. Approximate h needs its own
   error bound. Known alias conditionals permit another exact conditional-expectation reduction, with
   class discovery and source weighting still charged.
4. **L, a concrete full-support proposal without an M-word normalization.** Exact posterior bit branches
   are proportional to their prior branch mass times residual continuation likelihood. Replace that
   likelihood by F/I's positive conditional Gaussian formula, with exact polar prefix moments. Two child
   queries per bit give a drawable/evaluable ordered-slot proposal. A whole-configuration prior mixture
   gives W_max <= 1/epsilon; all tuple permutations push forward to the correct factorial count prior.
   Exact importance correction removes guide bias from the target. Its cost/concentration and conditional
   moment error remain explicit; no informative guide bound is assumed on unsampled prefixes.
5. **L, exact-target local blocks and finite-chain control.** K2/K3 can draw a block conditional on the
   current outside messages, using their full residual and any source exclusions. This retains all K-user
   dependence in the stationary target. Approximate block proposals need the exact acceptance ratio.
   A fixed global independence proposal with P/q <= gamma_bar minorizes the target by 1/gamma_bar; mixing
   it with probability beta into fixed invariant local updates gives kappa=beta/gamma_bar. S records spaced
   t >= log(S/delta_mix)/kappa transitions apart are within delta_mix of S IID target draws. Hoeffding then
   gives simultaneous count-tail control. Every transition and local contraction is charged. This
   conservative bound does not prove that the chain improves on independent importance sampling.
6. **M, the former factorial-moment pitfall becomes a controlled approximation.** Unexcluded residuals
   give mu_r=E binom(a_w,r). Their finite alternating combination reconstructs T_wj with coefficients
   (-1)^(r-j) binom(r-1,j-1). Truncation at m is an upper/lower bound according to parity and has error
   <= binom(m,j-1) mu_(m+1). A beta-integral/Taylor remainder proves this pointwise, without simulations.
   Global residual moments/covariances can be reused across words. Approximate moment error must also
   be propagated through the alternating coefficients.
7. **A bounded use of means.** For M>=K, selecting K distinct words by exact posterior mean has excess
   multiset risk <= C2/K, where C2=E posterior sum_w binom(a_w,2). Uniform mean error e adds 2e.
   This is a conditional approximation guarantee for the unchanged loss, not a return to the old metric.
   Small prior collision probability alone is insufficient; the simple C2 <= binom(K,2)/(M Z) bound
   exposes the evidence-normalization obstacle.

#### Review decisions

- The finite sampling guarantees apply directly on ordered slots. Dividing a slot target by an unknown
  symmetrized count-proposal density is neither needed nor correct for the stated estimator.
- A guide's approximation error controls importance concentration or mixing. After exact correction,
  adding its full deterministic posterior bias again would double-count an error that has been corrected.
- The chain guide and kernel-selection settings are frozen for each run. Invariance of a fixed kernel is
  not a license for arbitrary adaptive MCMC, or for treating successive states as independent samples.
- K2/K3 marginals alone are not joint samples. The report derives the conditional sequential draw, and
  charges the full pair/triangle construction. Outside messages are chain state, never known truth.
- Alias correction makes the ratio constant within each corrected class/group, not automatically across
  distinct groups that happen to share a signal. Partial disjoint groups still need exact conditional weights.
- The old paragraph saying cycles, weighted remainders and dependent sampling were wholly unresolved has
  been replaced. Their finite constructions now exist; useful ranks, concentration and mixing remain open.

#### Broad-pass endpoint

The report now specifies three complete paths: (1) guided independent importance decoding, with a corrected
block-chain variant; (2) deterministic prefix refinement with truthful gain bounds; (3) positive structured
contraction with optional importance/control-variate correction. Each includes discovery, target/source
handling, count readout, error accounting, stopping and full cost. Existing A-I entries have clear roles as
backends or query providers; no assumption atlas or empirical family ranking was added.

This is a useful point to close broad option generation and move to concrete decoder specifications and
symbolic cost/error comparisons. It is not a claim that all mathematical approximations are exhausted or
that a target-scale receiver has been established. Further algebra should address a named bottleneck in a
complete path. Implementation and small exact-reference validation remain a later stage.

Document verification: the final 40-page PDF compiles with resolved references/citation and no LaTeX box
warnings. All pages were rendered; the new derivations and decoder specifications received full-page
inspection, followed by reinspection of the final layout edits. Source checks passed for unique labels,
reference targets, control characters and whitespace. The delivered PDF matches that reviewed build.
Only the report, this log, the current handover and the documentation index were edited in this pass.
No receiver code, tests, simulation results or empirical rankings were added.

### 18 October 2026: continue the approximation investigation

The user explicitly keeps this stage open. The preceding recommendation to end broad exploration and
start decoder specifications is superseded. The guided independent and deterministic prefix compositions
remain useful contexts, but this pass develops further algebra and combinations, without choosing or
specifying receivers, running experiments, or building an assumption atlas.

#### Questions being worked through

- Can the residual closure be applied after an exact exponential change of measure, so that it approximates
  the distribution near the observed residual? Track its normalizing factor, excluded/prefix source laws,
  finite error, and the cost of the tilted polar partition and moments.
- Does fixed K and unit waveform energy permit a positive Gaussian auxiliary-field representation?
  If so, retain full count marks, derive the cost and variance rather than treating positivity as efficiency,
  and ask what a controlled reduction of the field directions would discard.
- Can bounded integer counts and a consistent prefix tree extract stronger decision bounds from the
  existing factorial moments? Distinguish a valid set of possible posteriors from an assumed posterior;
  use dual polynomial bounds to expose which extra moment would actually tighten a query.

Exact-report connections under review: one-user partition functions; residual source exclusions;
identity-preserving inside/outside marks; compatible future projections and their rank obstructions;
joint pair/triangle interfaces; convex support bounds and prior-weighted decision contrasts; connected
reference changes; Fourier marks; fixed-cardinality gauges; and equal-signal conditional identities.

#### Settled deductions

1. **N: tilt first, then close the residual.** For independent, possibly different, residual sources,
   let kappa(theta) be the sum of their log MGFs. Exactly,
   L(r) = A_theta(r) E_theta g(r - sigma^2 theta - S), with
   log A = kappa - theta.r + sigma^2 ||theta||^2/2. The unique minimizing tilt solves
   sum mu_theta + sigma^2 theta = r. This centers the remaining query under the tilted law.
   Applying F's finite replacement proof there gives relative error
   delta / (J_tilde - delta) when J_tilde > delta: the common small factor A cancels.
   This does not establish a small remainder/determinant ratio. Any trial tilt remains valid, so approximate
   optimization adds no separate error if its actual residual and moments are used.
2. **A connection to the exact geometric bound, with a price.** Jensen gives
   A exp(-(shifted-misfit^2 + tr V)/(2 sigma^2)) <= L <= A.
   The optimized upper bound is no weaker than the convex-hull likelihood envelope because the log MGF
   is bounded above by the support function. It retains prior probabilities lost by the maximum.
   Tilted polar moments generally lose uniform-character orthogonality: their partition/derivative cost
   can restore payload enumeration. A shared tilt preserves M's across-word factorization reuse at fixed
   moment order; separately optimized tilts may improve bounds but lose this reuse.
3. **O: exact positive auxiliary fields.** For the stacked K waveforms x and
   C = (K I - 11^T) tensor I / sigma^2, Gaussian fields Gamma with covariance C satisfy
   E exp(Gamma.x) = exp((K^2 - ||sum phi||^2)/(2 sigma^2)).
   The evidence is therefore a positive Gaussian integral of K one-user polar partition functions.
   Under its evidence-weighted field law, the IID slot messages are conditionally independent.
   Count/prefix polynomials retain repetitions and total K. Distinct sampling keeps a coupled injectivity
   constraint and cannot reuse the IID product partition.
4. **Positive does not mean concentrated.** The Gaussian-base weight h has second-moment ratio
   E h^2/(E h)^2 = E_(two independent exact posterior tuples) exp(x^T C x') <= exp(K^2/sigma^2).
   The two-user scalar +/-1, y=0 example attains asymptotic ratio exp(4/sigma^2)/2.
   Naive field sampling is therefore not an automatic improvement over the signed Fourier or guided routes.
   The weights are unbounded; H's bounded-weight theorem does not apply directly.
5. **Two different controlled field reductions.** In whitened coordinates U, the posterior field is a
   mixture of N(a_W,I) with ||a_W|| <= K/sigma. Thus the target mass outside radius R is bounded by the
   standard Gaussian radial tail beyond R-K/sigma, with no evidence denominator. Domain restriction uses
   A's omitted-mass rule. Dropping field directions instead multiplies each tuple weight by exp(-D(W)/2),
   where D=x^T C_omit x >=0. A certified range [D_minus,D_plus] gives
   TV <= tanh((D_plus-D_minus)/8); K's weighted correction bounds are an alternative.
   Exact integration of the missing directions restores the coupling. All nonzero eigenvalues of C
   are equal, so its spectrum alone does not select useful directions.
6. **A combination that does not help automatically.** A common waveform center is annihilated by C,
   so it leaves the auxiliary slopes and the discarded-direction quadratic unchanged. Connected-reference
   improvement therefore does not transfer mechanically to this representation. Direction truncation or
   field quadrature can also break exact alias conditionals; positivity is not sufficient to preserve them.
7. **P: use all bounded-count information.** Optimize count-tail or capped-gain expectations over
   probabilities p_0,...,p_K satisfying the available factorial-moment intervals. This is a finite LP,
   sharp for those constraints. For K=3, mean 1 and second factorial moment 1/5, it gives the attainable
   first-tail interval [4/5,13/15], sharper than [4/5,1] from the first alternating pair alone.
   Dual binomial-basis polynomials prove the bounds and expose sensitivity to individual moment errors.
   Their feasibility is checked only at integers 0,...,K.
8. **Shared evidence and prefix consistency.** Unnormalized moment constraints remain linear after
   introducing one common reciprocal evidence tau. Binary split tables for child counts, with row,
   column and sum marginals, enforce total K using O(N_tree K^2) entries. A recursive construction proves
   these tables admit a global leaf-count law; this is a feasibility witness, not posterior conditional
   independence. Selected mixed factorial moments come from marked IID slots with an unrestricted
   remainder and can be bounded by F/N. They constrain ambiguous splits without requiring all mixed moments.
   Fixed allocation contrasts are linear objectives; joint selection of every possible allocation is
   not claimed to become one LP.

#### Combinations and limits retained

- N supplies positive guides to L and certified intervals to I/P; these are different roles with different
  requirements. Exact importance correction removes guide bias, not finite sampling error or construction cost.
- F/G/N moment/evidence bounds can meet in P, provided their normalizations and marked source laws agree.
  Intersecting valid constraints does not assume independent approximation errors.
- O combines field-domain restriction with conditional integration error, or direction removal with K's
  correction bounds. Unknown field evidence/marginal proposal densities do not become free.
- Exact joint parity grouping, compatible projections and alias conditional restoration remain useful
  preprocessing/constraints. None of the new derivations bypasses unknown patterns, joint separator growth,
  whole-message identity or the discrete-mode failure of a Gaussian closure.
- The read-through also retained the connected/Fourier caveats: tilting changes their source/reference,
  so their characteristics and remainder must be recomputed; signed sums cannot be relabeled as positive laws.

The three new entries and a combination discussion are in Report 9. Its previous closing recommendation
and the handover/index now explicitly defer decoder specifications. Earlier compositions are preserved
as examples of how the algebra can connect, with no architecture selection or performance ranking.
Open work is the usefulness and access cost of these reductions: tilted partition queries, field
integration/direction choice, and the strength of a small moment/consistency constraint set.

Document verification: the final 48-page PDF compiles with no LaTeX warnings and resolved references.
All pages were rendered for layout review; the new sections, opening/scope and closing pages were also
inspected at full-page scale. Final layout changes were rerendered and checked. Source labels, references,
citations, control characters and whitespace were checked. The delivered PDF is the reviewed build.
Only Report 9 (source/PDF), this log, the current handover and the documentation index were changed in
this pass. No receiver implementation, mathematical test script, simulation or empirical result was added.

### 18–19 October 2026: close the outstanding theoretical dependencies

The user now requests a thorough final investigation before decoder implementation. This supersedes the
previous instruction to leave broad exploration open, without authorizing receiver experiments in this
pass. The completion criterion is concrete: every retained calculation must have defined source/count
handling, an evaluable finite construction, error propagation and a charged cost; remaining claims about
useful accuracy or runtime must be explicitly separated from missing algebra. This is not a claim that
every possible approximation, or every stronger theorem, can be exhausted.

Working dependencies to close:

1. Tilted polar partition/moment access: exploit conditional characters, build positive finite series,
   propagate the inner approximation through the outer residual closure, and preserve pattern/source weights.
2. Field reduction/integration: choose slot-symmetric physical directions, bound the discarded dispersion,
   and derive a bounded field proposal in a stated regime rather than assume naive Gaussian draws work.
3. Reference choice: connect the exact common-center gauge to a variational objective and distinguish
   reference optimization from a posterior-accuracy or connected-remainder certificate.
4. Sampling/query composition: convert certified child likelihood intervals into a specified robust branch
   proposal, charging all-prefix concentration rather than inferring it from visited paths.
5. Signed compression: identify when SVD/tensor approximations can supply certified marked queries even
   without positive approximate factors, including the cost of contracting their error envelope.
6. Moment completion: prove what a short hierarchy cannot determine, and separate a feasible count
   relaxation from an assertion about the physical posterior. Audit A-P and the new deductions together.

#### Settled deductions

1. **Q: finite polar inner queries.** For a conditioned pattern/payload prefix, group equal character
   keys in the linear score. Character multiplication is XOR and averaging selects the zero key.
   A degree-d polynomial therefore supplies the partition and inserted waveform moments, with at most
   binomial(D+d,d) monomials before grouping, or a dense 2^b character table. Pattern work remains explicit.
   A complementary cosh/tanh subset expansion has a certified elementary-symmetric omitted-subset bound.
   Independent character keys make that partition factor exactly, but there can be at most b such keys.
2. **A coherent positive approximation matters.** Shift a score into [0,L] and truncate its exponential
   Taylor series. The ratio to the exact exponential is in [rho_d(L),1], where
   rho_d(L)=exp(-L) sum_{j=0}^d L^j/j!. This provides positive source weights, partition intervals and
   normalized TV bounds. Obtain moments by inserting waveforms into the same polynomial, not by
   differentiating the truncated partition. Different child polynomials instead define a sequential
   proposal with its actual recorded density.
3. **Nested residual errors are now explicit.** For independent approximate tilted sources with TV
   bounds eta_i, the sum-law error is at most 1-product(1-eta_i). Add this to the Gaussian replacement
   error computed from those actual approximate sources, and multiply lower/upper bounds by the matching
   tilted partition products. No exact nonuniform moment oracle is assumed.
4. **A convergent alternative when Gaussian closure fails.** Apply the positive Taylor construction
   to the bounded log likelihood itself. Degree d needs sum moments through degree 2d, with up to
   binomial(n+2d,2d) monomials. One common shift/degree defines a positive full posterior whose log-ratio
   bound tends to zero. Query-specific shifts instead give valid local intervals. This removes the
   Gaussian assumption at a potentially enormous dimension/degree cost, not a claimed fast receiver.
   The analogous complex Taylor calculation gives continuous Fourier characteristic queries; projection
   onto the unit disk and telescoping powers control inner bias before frequency-sampling normalization.
5. **R: field compression is omitted pairwise dispersion.** Retaining the same physical subspace in
   each slot gives D(W)=sum_{i<j}||P_perp(phi_i-phi_j)||^2/sigma^2. Diameter and common-center radius
   bounds give worst-case certificates. Prior covariance eigenvectors minimize the prior average
   K(K-1) tr(P_perp Sigma)/sigma^2, not the posterior error. Conditional one-user field moments compute
   E_Q D without pair enumeration, and convexity bounds E_Q(exp(D/2)-1), hence TV, using D_max.
6. **A bounded field proposal has a precise regime.** A global tilted covariance envelope Sigma_bar
   gives h_0 <= h(BU) <= h_0 exp(U^T A U/2), A=B^T(I tensor Sigma_bar)B. If A<I, a Gaussian with
   covariance (I-A)^-1 has bounded importance ratio and concentration parameter at most
   det(I-A)^-1/2. This avoids the small physical-evidence denominator but can still be exponential.
   Unit norm always permits Sigma_bar=I; a smaller global envelope requires proof. The field Hessian
   exposes when the origin is not a local mode, so clipping a bad Gaussian precision is unjustified.
   Outside that sufficient regime, finite-ball restriction plus bounded integration remains a defined
   route. Uniform inner errors, direction loss, domain loss and integration error stay separate.
7. **S: a principled common center, with a counterexample.** In the identical-slot product family,
   the evidence lower bound is strictly concave in the one-user distribution. Its unique exponential
   reference satisfies sigma^2 theta+(K-1) mu(theta)=y. This is Report 7's common-center reference,
   distinct from N's K-user residual tilt. Strong convexity and the unit-energy covariance bound give
   a finite gradient-descent accuracy construction; Q supplies bounded inner errors. Any trial reference
   is valid, so exact optimization is unnecessary. Freeze the target-K reference and its positive edge
   floor throughout the connected recurrence. An evidence upper bound minus this objective bounds
   reverse KL and TV. The K=2 scalar +/-1 example remains inaccurate even at the exact product optimum.
8. **T: signed compression is a query route.** A cut's operator-norm residual bounds a marked bilinear
   contraction only with its actual incoming norms. For many approximate factors, telescope their product
   and contract the positive absolute-error envelope with the original source/count markers. This gives
   evidence and numerator intervals even when the surrogate is signed. Constructing the rank certificate
   and contracting its envelope can cost as much as the original calculation; no sign clipping or
   sampled-entry rank estimate substitutes for that bound.
9. **Robust bit guides.** A certified conditional branch interval p_0 in [l,u] has minimax proposal
   q_0=u/(1+u-l), with worst conditional target/proposal ratio 1+u-l. Multiplying uniform all-prefix
   width factors bounds complete-tuple concentration. Unvisited prefixes need bounds too; width one
   gives a valid potentially exhaustive-scale factor. These history-conditioned slot queries are not
   I/P's unconditional prefix-count events.
10. **Low-order moment information really can run out.** The even and odd halves of binomial(m+1,1/2),
    normalized separately, have identical first m factorial moments but tail differences
    2^-m binomial(m,j-1). A complementary leaf K-A retains cardinality and all mixed moments of total
    degree <=m. Mixing a common full-support law preserves the obstruction. This is a statement about
    the information in the constraints, not a claim that both are posteriors of the selected polar bank.
    Once numerical moment errors are small, wide LP bounds require new information, not another solve.
11. **Distinct-source control remains distinct.** Without-replacement draws are IID draws conditioned
    on no repeat; their prior TV is 1-(m)_h/m^h. This supplies an optional absolute residual-likelihood
    comparison on the same allowed alphabet, using the correct distinct count masses. At the full
    posterior level, the error instead depends on posterior repeat probability. A small prior collision
    rate cannot justify exchanging the two decoder targets.

#### Completion and combination audit

- The final report table audits all A–T by finite calculation, refinement limit and remaining cost or
  informativeness question. The concrete preparatory dependencies above are closed. This is not a
  universal claim that no other approximation or stronger theorem exists.
- Support loss eta, conditional target error tau and numerical tail error e compose as
  eta+(1-eta) min(1,tau+e), with confidence failure accounted separately. Exact importance correction
  removes proposal bias; approximate target weights do not. Valid numerator/moment intervals can share
  one exact evidence even when they do not arise from a common approximate joint law.
- Estimated tails must support the stated decision optimization. Sorting each word's clipped tails
  cannot increase a uniform error bound relative to true ordered tails, and makes greedy allocation
  valid; this does not assert joint posterior consistency.
- The report and handover now identify implementation specification and small exact-reference comparison
  as the next stage. Actual error, certificate tightness/failure and full cost must be measured separately.
  No architecture is selected, and no decoder or mathematical test was implemented in this pass.

Document verification: the completed 61-page PDF builds without LaTeX warnings, overflow or unresolved
references. All pages were rendered for contact-sheet review; the new derivations, opening, completion
audit and references were also inspected at full-page scale. The final edited page was rerendered and
checked. Source labels/citations/environments, control characters and whitespace pass static checks.
The delivered PDF is the reviewed build. This pass changes Report 9 (source/PDF), this log, the handover
and documentation index only; unrelated working-tree changes are preserved.
