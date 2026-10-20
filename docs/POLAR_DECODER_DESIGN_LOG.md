# Polar decoder design log

This log starts the decoder-composition stage after the exact and approximation investigations.
[Report 7](reports/07_polar_decoding_from_posterior.pdf) owns exact identities and obstructions;
[Report 9](reports/09_polar_bayesian_approximations.pdf) owns approximation algebra and error bounds.
[Report 10](reports/10_polar_decoder_designs.pdf), with [source](reports/10_polar_decoder_designs.tex),
owns complete decoder options and their variations. This log records the design work and, after the
user's authorization below, the first common-harness implementations and their verification.

## 19 October 2026: assemble the decoder options before implementing

The user requests the full list suggested by the existing mathematics, with understandable descriptions
and any additional derivation needed to make each option a full decoder. Selection and implementation
come after reviewing that list. The prior recommendation to choose a first implementation is therefore
superseded by this composition/review stage.

The fixed target remains the known-K, unit-energy polar/CRC encoder with unknown messages and patterns,
real physical AWGN and IID uniform complete messages. The distinct source is a separately conditioned
control. Every returned decision is a full-message multiplicity vector with total K, judged using
posterior count tails. No learning, oracle patterns or replacement MAP objective is introduced.

### Questions and working organization

1. Separate complete decoder families from approximation ingredients. A partition calculation, moment
   LP, covariance, local K2/K3 update or candidate list is not by itself a decoder.
2. Every family must close discovery, legal inference, count-tail readout, final allocation, stopping
   and error/cost reporting. Explicit alphabet traversal is a valid finite construction when charged.
3. Candidate families: direct residual queries, a variational product decoder, certified prefix search,
   independent guided importance sampling, corrected local/global chains, compatible quantized states,
   positive regional/feature contraction, connected expansions, Fourier queries, signed low-rank queries,
   auxiliary-field inference and sequential particles. Keep exact reference evaluations as one separate
   reference family. These labels distinguish execution styles, not disjoint statistical targets.
4. New closure to investigate: field proposals can keep their auxiliary variable in the importance
   target, avoiding an unavailable marginal tuple density. Determine how to correct omitted directions
   and approximate inner categorical guides without changing the physical posterior.
5. New closure to investigate: prefix-guided particle resampling. Define its telescoping weights, prove
   unbiased unnormalized count queries and obtain finite confidence from independent particle batches,
   without treating correlated descendants as independent samples.
6. Close the product decoder's message discovery: identify how many high-score words suffice for its
   Bayes action, and specify bounded prefix search and partition accuracy.
7. Map every Report-9 entry A–T and every major Report-7 exact resource to a retained role. A finite
   catalogue of families and compatible variations is feasible; every possible future hybrid is not
   an enumerable or meaningful completeness claim.

No receiver code, mathematical test scripts, simulations, tuning or PUPE results are part of this pass.

### First deductions and corrections to the organization

- There are two distinct search spaces: message-prefix cells ask how many users occupy a set of words;
  configuration-prefix search explores explanations of all K slots. The latter supplies a separate
  complete bounded-list/beam option, with a disjoint unresolved frontier and explicit omitted mass.
  It should not be hidden inside the deterministic message-prefix option.
- For an identical-slot product law, its Bayes action needs only K highest-probability words (or all
  words if M<K). An outside occurrence can be exchanged for the first occurrence of an unused inside
  word, whose first tail is at least as large. A priority search using conditional character score
  bounds discovers these words without assuming the list is supplied. It can still visit every word.
- Auxiliary fields can be corrected on their joint space. With retained field Gamma=BU and omitted
  dispersion D(W), use joint unnormalized weight c_y phi_d(U) M^-K exp(t.sum(phi)+Gamma.x+D/2).
  Its message marginal is exactly the physical target. A proposal q_U times K evaluable categorical
  guides has a directly evaluable joint importance ratio; no field-integrated tuple density is needed.
  Approximate inner positive polynomials can serve as guides and their error is exactly corrected.
  D/2 correction removes direction bias but may inflate concentration exponentially.
- A field-ball restriction after that correction still has a uniform posterior tail bound: conditional
  on W, U is Gaussian with mean B^T x_W and norm at most K/sigma. Conditional-product count-polynomial
  readout is available without tuple coupling only when the retained target factorizes given U.
  The D correction generally breaks that conditional factorization; sampled complete tuples remain valid.
- A prefix particle scheme can be made a full decoder without an unproved effective sample size.
  Use gamma_t(prefix)=prior(prefix)*ell_t(prefix), with ell_0=1 and terminal ell_T equal to the exact
  physical likelihood. Incremental weights telescope. Multinomial resampling preserves the expectation
  of the unnormalized marked measure by conditioning one step at a time. Independent particle batches
  therefore provide unbiased evidence/numerator replicates. Their deterministic product weight bound
  permits the same finite concentration argument as Report 9 H. Particles inside one batch remain
  correlated, and the resulting bound may be extremely conservative.
- This is classical sequential importance/resampling machinery, used here with a finite prefix/count
  proof rather than an imported scalability claim. Context source: Del Moral, Doucet and Jasra (2006),
  https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf .

The retained organization is now one exact reference family plus thirteen execution families. Shared
query providers, moment LPs, exact projections/aliases and proposal corrections are variations or modules.
The report will map all of them explicitly, rather than count their Cartesian product as new decoders.

### Completed design catalogue

| ID | Complete execution family | Important variations or limits |
|---|---|---|
| REF | Exact count-tail Bayes evaluation | Residual, inside/outside, projected frontier, elimination, Fourier, connected and K2/K3 evaluations share the target. |
| DEC-01 | Direct excluded-residual count queries | Gaussian, mode mixture, tilt, likelihood polynomial, factorial moments/LP; explicit full-word traversal. |
| DEC-02 | Variational product decoder | Identical-slot optimized/trial tilt with binomial tails and top-word search; slot-specific coordinate updates with Poisson-binomial full-word readout. |
| DEC-03 | Message-prefix gain refinement | Any bounded cell-count provider; consistent moments; actual full-word incumbent and optimistic gain certificate. |
| DEC-04 | Bounded configuration search/list | Disjoint tuple-prefix frontier, exact completed weights, retained-list decision and omitted-evidence bound; beam pruning retains its bounds. |
| DEC-05 | Independent guided importance | Positive point guides or robust interval guides, whole-tuple defensive mixture, surrogate proposals and exact conditional integration. |
| DEC-06 | Corrected local/global chains | K1/K2/K3 conditional or guided block moves plus global refresh; finite minorization schedule, not assumed burn-in. |
| DEC-07 | Sequential particles | Prefix-growth and fixed-temperature full-tuple variants; independent batches supply the finite confidence unit. |
| DEC-08 | Quantized continuation states | Compatible projection, occupied-only rounding, marked reverse pass and optional corrected backward samples; remains explicit in M. |
| DEC-09 | Positive regional/feature contraction | Forests, joint cyclic cutsets, constrained bit/pattern factors and exact marked elimination of the approximate model. |
| DEC-10 | Connected-component contraction | Positive auxiliary-component restriction or signed expansion with remainder; target-K reference held fixed. |
| DEC-11 | Fourier count queries | Finite marked modes or continuous Gaussian frequencies with inner polynomial bounds; full-word or DEC-03 readout. |
| DEC-12 | Signed low-rank count queries | Actual cut-norm bounds and positive propagated envelopes; construction/certificate cost may remove the gain. |
| DEC-13 | Auxiliary-field inference | Conditional count integration or exact joint field/tuple correction; reduced/full fields, bounded Gaussian/ball proposals or a deterministic cube grid with remainder. |

These are not thirteen statistically distinct objectives or a promise that every variation is useful.
Decoder IDs are unrelated to the historical D0–D4 names. A covariance, moment LP, local block update,
or candidate list is not counted as an additional full decoder.

### Additional derivation details and audit

1. **Product discovery.** For an identical-slot multinomial law, take min(K,M) words with greatest
   probabilities. Any allocation using an outside word leaves an inside word unused; its first tail
   dominates the removed outside tail. Swapping proves an optimum exists within this candidate set.
   Conditional-character score bounds support a finite priority search, initialized over all patterns.
   The result is specific to this product law and does not justify top-K marginal means for the
   interacting posterior. Partition approximation remains a separate cost/error.
2. **Joint field correction.** The target on (u,W) integrates to the exact tuple weight because the
   retained Gaussian quadratic plus D/2 equals the original full quadratic. For categorical guides
   proportional to positive polynomial weights f_tilde in [rho*f,f], the exact importance ratio is
   c_y phi(u)/q_U(u) exp(D/2) product(psi_tilde*f/f_tilde).
   Under R's curvature regime it is bounded by c_y D_A h_0 exp(D_max/2) rho^-K.
   Approximate partitions are proposal quantities here; correcting their actual density removes bias.
   For a ball, its posterior-mass bound follows from U|W ~ N(B^T x_W,I) even after direction correction.
   The D correction generally recouples conditional words, so the uncorrected count polynomial cannot
   be used after that correction without additional integration.
3. **Particle batches.** Gamma_t(prefix)=pi(prefix)*ell_t(prefix), ell_0=1, ell_T=g.
   Incremental weight is pi(bit|prefix)*ell_t/[q_t*ell_(t-1)]. Conditional averaging over multinomial
   resampling and propagation proves the unnormalized marked measure is unbiased by induction.
   Guide clipping and defensive branch probabilities give a product bound on each batch's normalizer.
   Pool unnormalized evidence/numerators across independent batches, not equal averages of normalized
   batch ratios. Read out a positive mixture of terminal legal tuples and cover unseen words too.
4. **Tempering variation.** On full tuples, gamma_t=pi*g^beta_t with a fixed 0-to-1 schedule has
   increments G_t=g^(beta_t-beta_(t-1)) <=1. Resampling and any exactly invariant stage mutation preserve
   gamma_(t-1)(G_t K_t f)=gamma_t(f). Each batch's normalizer is therefore bounded by one and its
   marked unnormalized queries are unbiased. No claim of stationarity after a finite local mutation
   is needed for this identity; finite-batch concentration can still be poor. Adaptive temperatures
   chosen from the same particles are outside this proof.
5. **Common readout.** A bounded independent pair (Y,X_H), with 0<=X_H<=Y<=C and expectations Z,ZT_H,
   gives simultaneous normalized-tail confidence. The independent unit can be a tuple, a field draw,
   or an entire particle batch. All MK tails are covered without allocating an M-array; sparse updates
   cost K per tuple. Signed Fourier variables use a different bounded-variable argument.
6. **Source and error boundaries.** IID slot weighting induces the exact factorial count prior;
   no second factorial is inserted. Distinct laws retain exclusions and their own normalization.
   Positive graph components and product field integrals are IID constructions. A source-approximation
   bound or exact coupled distinct computation is required before using them for that control.
   Support loss, target bias, numerical uncertainty and failure probability are kept separate.
7. **Combinations.** Prefix discovery wraps bounded query engines; exact correction wraps evaluable
   positive joint proposals; exact K2/K3 integrates or updates blocks inside full-user dependence.
   Mixtures use actual mixture densities. Shared evidence/moment constraints can combine providers
   without claiming a shared approximate joint law. The final coverage tables account for every A–T
   entry and the major exact-report resources.

The current stage is review of this catalogue and its variations before choosing implementations.
The handover/index now say so explicitly; the previous suggestion to start one implementation is superseded.
Reports 7 and 9 remain unchanged. No implementation, numerical mathematics test or receiver experiment
has been added. Practical concentration, tightness and scaling remain questions for the eventual
small exact-reference comparison.

### Final variation gap check

- **Slot-specific product laws:** the objective is
  -[||y-sum(mu_k)||^2+sum(1-||mu_k||^2)]/(2 sigma^2)-sum KL(nu_k||uniform).
  Each exact coordinate update is exponential with theta_k=(y-sum_{i!=k}mu_i)/sigma^2.
  Fixed cyclic sweeps from declared initial tilts define a finite construction; exact sweeps improve the
  objective, not necessarily globally. Count polynomials are Poisson-binomial and direct readout costs
  O(MK^2). The evidence-gap certificate still holds after marginalizing to counts; the identical-slot
  top-K shortcut and equality of slot/count KL are not reused. Symmetrizing slots leaves counts unchanged.
- **Sampled cutsets:** uniform selection from L sectors plus each sector's contracted evidence/numerator
  produces the unbiased pair (L Z_sector,L N_sector). A positive full-model evidence envelope bounds
  each sector and hence supplies a finite independent-replicate guarantee. This closes the sampling
  variant without assuming access to an optimal sector law.
- **Deterministic field quadrature:** on [-R,R]^d the exact unnormalized count integrand has gradient
  bounded by (sqrt(d) R+K/sigma) F_max, with
  F_max=(2pi)^(-d/2) exp(K||t||+(K/sigma)sqrt(d) R).
  Midpoint error is at most cube-volume times this bound times half the cell diagonal. Positive
  Gaussian-mixture centers of norm <=K/sigma bound posterior mass outside the cube by the radial tail
  past R-K/sigma. Node-value interval error is added separately. The m^d node cost is explicit.

These variations remain inside the same thirteen families. All newly retained alternatives now specify
the inference calculation, word/count readout and a bound; merely naming quadrature or sector sampling
is no longer doing the work of an algorithm.

### Delivery checks

- Report 10 compiles to 23 pages with no LaTeX warnings, unresolved references or over/underfull boxes.
  Labels, citations, environment pairing, text/control characters and page bounds pass static checks.
- Rendered every page, inspected both contact sheets and the principal tables, new derivations and
  final review page at readable resolution. Fixed the isolated spill paragraph and product-variation
  page break; no clipped equations or overlapping table rows remain in the reviewed output.
- The final mathematical edge-case pass specifies a legal incumbent when independent replicate
  weights are all zero, and uses min(K,M) in the product discovery stopping rule.
- Updated the handover and documentation map for decoder-option review. Added the new log to the
  existing curated-document exceptions in .gitignore so it is available for versioning.
- Verification here is document/static/layout work. No receiver implementation, numerical mathematics
  test, simulation or empirical performance claim was added; Reports 7 and 9 were not changed.


## 19 October 2026: authorized first implementation and shared benchmark

The user has now requested an executable comparison harness, the exact reference and one or two
additional receivers, historical D2/native ODMA comparisons, reference bounds, plots and B12/B14
CPU feasibility measurements. This supersedes the design-only stopping point above. Reports 7, 9
and 10 retain their mathematical scope; this entry records the implementation and small experiments.

### Initial implementation choices

- REF enumerates sorted count configurations, with IID source mass K!/(M^K product c_w!), or the
  uniform distinct-source mass. It sums physical Gaussian likelihoods, computes every count tail
  and minimizes expected multiset PUPE. It is exact algebra evaluated in floating point, not an
  interval-arithmetic certificate. State/memory guards reject infeasible work before enumeration.
- DEC-02 is the identical-slot product variation. A finite L-BFGS solve minimizes the strictly convex
  tilt objective sigma^2 ||theta||^2/2 + (K-1) log psi(theta) - y.theta. Its categorical probabilities
  give exact binomial count tails for that approximate law. K=1 has the direct exact solution.
  Optimizer convergence and gradient norm are recorded; a finite unconverged iterate is a declared
  trial law. The implementation traverses all M words and has no prefix discovery yet. IID only.
- DEC-05 uses independent ordered tuples. A sequential isotropic-residual guide is mixed with the
  physical source prior at the whole-tuple level. The default prior fraction is .1 and sample count
  1,024. We evaluate the actual mixture density, original source probability and physical likelihood,
  then form normalized importance-weighted count tails. No second factorial is inserted: ordered
  IID tuples already induce the factorial count prior. Distinct guides mask earlier selected words.
  Its fixed-sample Hoeffding guarantee covers all MK tails; ESS is only a diagnostic. The initial
  implementation still uses explicit M-way proposals and an MK tail table.
- D2 uses the existing recurrence and training loop, with frozen encoder and declared eight-layer
  defaults. Uniform-energy ODMA invokes the existing baseline unchanged. For balanced energies the
  ODMA chain uses the known amplitudes in LLRs and exact waveform cancellation; this variant is
  explicitly labelled an adaptation. Native duplicate suppression and missing outputs are scored
  as returned, without oracle repair.

REF is separate from DEC-01 through DEC-13. All thirteen design IDs are reserved, but only DEC-02
and DEC-05 are implemented in this first pass. Other IDs fail before training/evaluation instead
of aliasing a different receiver. The existing run(scenario, **params) -> (counts, meta) contract
keeps the bank open to the remaining families and new combinations.

### Comparison and storage decisions

The common runner fixes the polar/CRC construction, uses the analytically balanced energies by
default, and checks every explicit column has unit energy. Uniform energies form a separate
control. The physical real noise variance is 1/(2 B 10^(Eb/N0/10)). K is known to every receiver;
messages, patterns and truth-dependent candidates are not supplied.

Per-frame seeds depend only on the experiment seed, source, K and frame number. Different receivers,
separate jobs, decoder ordering and SNR ordering preserve the same truth/standardized noise.
Training and validation have separate streams. Every result uses multiplicity-aware PUPE with the
same output budget. IID and distinct panels remain separate. Historical distinct comparisons remain
meaningful only when their full encoder/training contract matches; old IID membership scores must
not be pasted into these panels. The new harness uses float64, including D2; its legacy training
recurrence/protocol is preserved but historical float32 training is not a bitwise replay.

Saved artifacts include the resolved configuration, source hashes, codebook, actual inputs, per-frame
sparse decoded counts, metrics, latency, diagnostics, checkpoints, learning history and aggregate
JSON/CSV. Setup, training and inference costs are separate. With REF present, conditional regret and
count-tail error are evaluated afterwards, never given to another receiver. Single-seed uncertainty
uses frame-level t intervals; multiple-seed aggregation uses seed means. Distribution-free Hoeffding
intervals are also stored. Paired differences against REF, D2 and ODMA are retained.

The two plot references are the existing Polyanskiy p_t+q_1 achievable-error upper bound and the
historical genie/list-Fano lower bound. The former is a coding-achievability comparison, not a floor
or a fixed-polar guarantee. For the latter, reveal the other K-1 users and bound the remaining word's
information by n/2 log(1+2 B gamma/n). Its conditional alphabet is M for IID or M-K+1 for distinct.
The list-Fano root bounds list-miss loss, which is no greater than multiset loss, so it remains a valid
but often weak lower bound. Tiny same-encoder REF is the useful decoder-specific target.

### Completed smoke observations

Three smoke runs use B4/n12/Nc8/CRC4, K=1/2/3/4, Eb/N0=-4/4 dB and 128 frames per point:
balanced IID, uniform IID, and uniform distinct. D2 has a shortened 30-epoch cap. Together they contain
14,336 decoder-frame evaluations. The distinct run excludes the IID-only DEC-02. At K4 and 4 dB:

| Encoder/source | REF | DEC-02 | DEC-05 | D2 | ODMA |
|---|---:|---:|---:|---:|---:|
| Balanced IID | .05664 | .15625 | .05859 | .19531 | .37109 |
| Uniform IID | .07227 | .16406 | .06836 | .22266 | .40234 |
| Uniform distinct | .08203 | — | .08203 | .16602 | .39063 |

These are tiny-instance checks, not a production ranking or paper-scale performance evidence.
Empirical errors can put an approximation slightly below REF on finite frames; checked conditional
regrets are nonnegative. In the balanced K4/4 dB cell DEC-05's mean conditional regret is .00194 and
mean maximum tail error .02960, compared with DEC-02's .10314 and .52797. DEC-05's mean ESS is about
80 of 1,024. The conservative excess-risk bounds remain 1: good observed decisions do not imply a
useful certificate. The balanced D2 validation objective decreases from .20321 to .19219, while
validation PUPE stays .08984; no validation-PUPE improvement is claimed from that trace.

Plots include PUPE versus active users, inference time versus active users, D2 learning and exact
posterior diagnostics. The generated artifacts remain ignored under jobs/035_polar_decoders/results_local_*.

### B12/B14 feasibility and queued-work preparation

Timing-only probes at n256, K7/26 and 4 dB completed on the local Apple M2/8 GB laptop with two CPU
threads. DEC-05 at 1,024 samples takes about .71–2.61 seconds/frame for B12 and 2.85–10.56 for B14.
For 8,192 observations per seed (four loads, eight SNRs, 256 frames), endpoint-rate extrapolation
suggests 1.6–5.9 hours and 6.5–24 hours respectively. These are not walltime bounds: SNR, geometry,
threading and hardware matter. DEC-02, D2 and ODMA inference are much cheaper. At the D2 20,000-batch
cap, measured batch costs suggest roughly 23–33 minutes for B12 and 58 minutes for B14, plus validation;
early stopping may shorten this. These probes do not measure production PUPE.

CPU HPC is recommended for repeated guided-sampling grids; no GPU is used by the current paths.
The checked-in PBS array has 24 rows, B12/14 times three seeds times four practical receivers,
eight CPUs/32 GB/72 hours per row. Optional uniform-energy and distinct-source manifests are supported.
REF is excluded: even K7 needs about 3.9e21/6.3e25 count states at B12/14. No job has been submitted.
The 035 README contains all commands and parameters. Full B12/B14 quality and concentration remain
unmeasured until those experiments run; implementing more decoder families remains separate work.


### Delivery verification

- 46 focused regression tests pass: independent exact count posterior/Bayes decisions, repetitions
  and aliases, normalized proposal densities and importance correction, K1 product exactness,
  D2 recurrence equivalence, unchanged uniform ODMA, metric/source guards, paired job splitting,
  missing/duplicate-record rejection, checkpoint reuse and training-contract rejection.
- Recomputed all 14,336 principal smoke scores from the stored truths/decoded counts and verified
  observation fingerprints, legal output budgets, distinct-source outputs and unit column energies.
- Ran the real job-row entry point for four tiny rows (two receivers, two seeds), then merged them
  with the job merger. Seed-level confidence and per-seed matrix provenance were checked. A final
  small all-five-decoder CLI run also completed with plots and a one-epoch D2 training pass.
- Inspected the main balanced/uniform/distinct panels, training/runtime/reference plots and the
  merged single-SNR layout. Widened single-panel figures to prevent title/footer clipping and
  suppressed empty reference-only diagnostic panels. Source compilation, shell syntax and
  git diff --check pass. Generated experiment artifacts remain ignored.
- Merger outputs retain each original run's setup/training costs and matrix identity. Multi-seed
  aggregates do not pretend to have one encoder or seed. The job merger also checks saved configs
  against its manifest. Full research experiments and scheduler execution remain pending.


## 20 October 2026: all thirteen finite decoder variants

The user requested the whole decoder bank, one complete HPC campaign, mathematical smoke validation
and a queue command. This supersedes the two-family implementation scope above. Historical entries
remain records of their dates; the current registry and job-035 README own executable defaults.

### Implementation choices and the algebra preserved

The common Scenario/readout helpers moved to `framework/polar_decoder_core.py`; the public registry
still exposes the same `run(scenario, **params) -> (counts, meta)` contract. Query, search and contraction
modules implement different families without circular imports. No truth or exact-reference posterior
enters an approximation. All thirteen currently enumerate the message alphabet; no M-free claim is made.

- **DEC-01:** match the mean/covariance of the sum of the other K-j words conditional on excluding the
  queried word. Compute all j masses with the binomial prior, Gaussian determinant and quadratic term,
  then normalize per word. The excluded covariance is a rank-one downdate of the shared covariance.
  For distinct sources, use the finite-population covariance factor h(N-h)/(N-1), N=M-1, and j=0/1.
  These separate Gaussian queries need not define a consistent joint posterior. Modeling error is
  honestly bounded by one, rather than estimated from residual variance.
- **DEC-03:** binary cells partition the full message IDs. The optimized product law supplies exact
  binomial cell-count tails. A coarse cell can give an optimistic gain, while discovered leaves supply
  a legal word-level incumbent. Split selected unresolved cells; charge the optimistic/incumbent gap
  plus the provider's variational error. This is a genuine prefix decision process, but the current
  provider still costs a full M-way tilt. It should match DEC-02 once its decision gap closes.
- **DEC-04:** retain width-32 ordered tuple prefixes using a residual guide, then weight completed
  tuples by the physical likelihood and ordered source prior. Every pruned prefix contributes a
  disjoint omitted-evidence upper bound via the unit-norm completion inequality. Pruning is not
  equivalent to proving the discarded states irrelevant. Exhausting the beam recovers REF.
- **DEC-06:** single-slot Gibbs updates use the exact all-word conditional. Fixed-probability global
  independence-MH refreshes use a defensive product/prior mixture with its full joint density.
  Optional K2/K3 block proposals are uniform legal joint proposals with the physical MH correction;
  they are not separate marginal updates. Finite chains return empirical count tails. The conservative
  prior-based minorization is recorded, and correlated retained states do not receive an independent
  sample certificate. The default excess-risk bound remains one.
- **DEC-07:** grow word prefixes with multinomial resampling and a defensive proposal. Intermediate
  residual guides are clipped positive potentials; the terminal potential is the physical Gaussian
  likelihood. Incremental weights include source/proposal and new/old potential ratios. Pool batch
  evidence times terminal marked estimates before normalizing. The independent confidence unit is
  the whole particle batch, not each resampled particle. Fixed budgets are four batches of 64.
- **DEC-08:** a positive word-count recurrence keeps occupancy, a compatible projected continuation
  state, log weight and conditional count marks. Orthogonal Gaussian terms are absorbed exactly.
  Quantized/nearest representative merges keep all legal path weight with midpoint log corrections;
  their support-function ranges accumulate into a log-error/TV bound. The four-representative cap
  merges states rather than silently deleting paths. Candidate restriction has a separate bound.
  Zero rounding with enough representatives matches REF for both sources.
- **DEC-09:** partition words into small regions, retain every within-region count state, delete
  cross-region energy terms and contract the positive regional polynomials with an exact total-K
  constraint. Marked outside products recover count tails. The default uses singleton regions.
  A fugacity tilt stabilizes all cardinality convolutions: its factor exp(-lambda*K) cancels at fixed
  total K, so it changes numerical scale, not the approximate model. One full region is exact; on an
  orthogonal codebook singleton regions are exact. This is the simplest declared regional variant,
  not an implementation of every feature-factor/cutset alternative in the report.
- **DEC-10:** retain positive auxiliary components of orders up to q=1/2/3, assemble all disconnected
  combinations with Bell recurrences, and mark each queried word's multiplicity. Pair terms use f12;
  triple terms use f12*f13 + f12*f23 + f13*f23 + f12*f13*f23. The reference center, unary field and global
  pair floor are fixed for the target K. Logs prevent high-SNR coefficient overflow. Full alphabet
  and q=K are exact at the implemented two/three-user endpoints. The default q=2 restricts connected
  component size, not total user count; it has no nontrivial omitted-component certificate yet.
- **DEC-11:** shared continuous Gaussian frequencies evaluate the physical Gaussian kernel through
  characteristic functions. Marked j-count integrands use the excluded-word characteristic function
  and binomial coefficient. Event/complement intervals retain signed uncertainty; no negative mass
  is clipped into a pretend positive posterior. A legal lower-gain decision uses one-user scores
  only to break uninformative ties. At 512 frequencies these intervals can be entirely vacuous,
  so a functioning receiver need not have useful PUPE. K1 uses the exact analytic endpoint.
- **DEC-12:** split ordered users in two, retain a beam on each half, construct the physical cut table
  with the original full-alphabet prior, and take its signed SVD approximation. Exact distinct-source
  cross-half exclusions remain enforced. Actual entrywise residual envelopes give event/complement
  intervals; the exact retained normalizer is available because the table was constructed. Omitted
  list prior mass is bounded separately. Full lists/full rank recover REF. An empty legal cross-half
  list returns an explicitly labelled legal observation-only incumbent with bound one. This verifies
  the algebra and interface, not cheap construction of a large cut table.
- **DEC-13:** sample a bounded auxiliary Gaussian field and a defensive conditional tuple proposal,
  using the joint identity Omega(u,W)=pi(W)*g(y-s_W)*phi(u-B^T*x_W). The weight includes the Gaussian
  density ratio and full tuple proposal density. Retaining two physical eigen-directions does not
  drop the physical interactions: exact joint correction keeps the target, while proposal quality
  changes. A ball remainder and bounded-weight sampling error are charged separately. Rank zero
  gives an independently checked special case. Strong field/likelihood concentration remains a
  practical failure mode at a finite sample budget.

DEC-02/05 and D2/ODMA retain their preceding definitions. The new Gaussian/prefix/beam/contraction
paths also use exact K1 endpoints where appropriate. Sampling paths DEC-05/07 retain their stochastic
K1 calculation. IID-only families are DEC-02/03/10/11/13; the other eight support distinct sources too.

The implementation is complete at the **one declared variant per family** level. Mixture residual
providers, alternate factor graphs, tempered SMC, adaptive Fourier proposals, richer quantization and
other catalogue combinations remain possible future variants, not silently promised capabilities.

### Mathematical and executable validation

Independent small-alphabet tests cover Gaussian excluded moments, exact product-prefix decisions,
beam exhaustion, joint proposal normalization and detailed balance, chain frequencies, independent
SMC evidence/marks, exact quantization and error bounds, regional exact limits, connected pair/triangle
identities, Fourier mark/normalizer identities and signed intervals, SVD residual certificates and
rank-zero/rank-two joint field corrections. The first pass caught and corrected an actual source-mode
bug in regional state enumeration (an IID source string had been treated as a Boolean distinct flag).
Both-source exact-limit tests now protect that case.

The previous harness tests were extended to all stochastic families and load sharding. They compare
saved count decisions, metrics and observation fingerprints against a combined run, including different
K order. The checked-in manifest is compared with its builder and excludes infeasible research REF.

### HPC campaign and measured cost

The complete default campaign contains 648 rows: two B values (12/14), two energy mappings, three seeds,
13 new-family by four-load rows, plus one D2 and one ODMA row per encoder/seed. Each new-family row has
2,048 frames; each baseline row has 8,192. This is 1,474,560 decoder-frame evaluations. D2 trains once
per encoder/seed using the unchanged full training-load contract. Separate output roots preserve earlier
five-receiver results. Research rows request eight CPU cores, 32 GB and 72 hours; no GPU is used.

All 15 practical receivers completed timing-only calls at B12/B14 K7/26 on the M2 laptop (two threads).
At B14 K26, seconds/frame were 10.44 for guided importance, 7.11 for joint fields, 4.05 for particles,
3.46 for Fourier, 1.72 for chains, 1.38 for quantized states and 1.10 for regions. Other paths were below
one second, with strong candidate/list restrictions explaining the cheapest contractions. The slowest
measured one-load-row extrapolation is about 5.9 hours. These single-observation probes establish local
execution and rough scheduling cost, not PUPE, worst-case time or cluster performance. REF remains
combinatorially infeasible at research loads. Full performance measurements are pending HPC execution;
no scheduler job has been submitted.

A final metadata audit removed auxiliary product ELBOs from chain, field and connected-contraction
outputs. They were proposal/centering diagnostics, not bounds for those receivers; a restricted
alphabet or distinct source would also change their interpretation. The variational-reference diagnostic
now arises only for the actual IID product and product-prefix paths. Count decisions are unchanged.
Invalid component/block orders are rejected during config validation, before training starts.


### Completed all-family smoke evidence

Three final-source panels completed: balanced IID and uniform IID with all sixteen receivers, and
uniform distinct with the eleven supported receivers. Each uses B4, K1–4, two SNRs and 128 frames/cell:
**44,032 decoder-frame evaluations**. The independent saved-artifact audit reconstructed every observation
and every PUPE score, checked column energies, fingerprints, legal budgets/distinct outputs, candidate
recall and summary means. All passed; no measured conditional regret exceeded a reported numeric bound.
Most bounds are one, so that last check is not evidence of useful certification.

At K4 and Eb/N0=4 dB:

| Decoder | Balanced IID | Uniform IID | Uniform distinct |
|---|---:|---:|---:|
| REF | 0.05664 | 0.07227 | 0.08203 |
| DEC-01 | 0.17188 | 0.19141 | 0.20508 |
| DEC-02 | 0.15625 | 0.16406 | unsupported |
| DEC-03 | 0.15625 | 0.16406 | unsupported |
| DEC-04 | 0.05469 | 0.07422 | 0.09180 |
| DEC-05 | 0.05859 | 0.06836 | 0.08203 |
| DEC-06 | 0.05859 | 0.08398 | 0.10938 |
| DEC-07 | 0.06250 | 0.07031 | 0.09766 |
| DEC-08 | 0.34375 | 0.33008 | 0.36133 |
| DEC-09 | 0.30859 | 0.33984 | 0.34961 |
| DEC-10 | 0.36719 | 0.35938 | unsupported |
| DEC-11 | 0.66992 | 0.67969 | unsupported |
| DEC-12 | 0.06445 | 0.07031 | 0.09375 |
| DEC-13 | 0.21875 | 0.22266 | unsupported |
| D2 | 0.19531 | 0.22266 | 0.16602 |
| ODMA | 0.37109 | 0.40234 | 0.39062 |

These are short-training, tiny-alphabet smoke results, not a production ranking. In the balanced cell,
mean conditional regrets were .00176 (beam), .00194 (independent guide), .01220 (chain), .00332 (particles)
and .00205 (signed cut), versus .10314 (product) and .14150 (D2). An empirical PUPE slightly below REF
is sampling fluctuation: no action beats its exact conditional Bayes risk. The chain's equal empirical
PUPE to guided sampling does not imply equal posterior approximation quality.

The results expose real approximation limitations. Quantized four-state and deleted-edge regional
models lose substantial high-SNR accuracy. The positive connected-pair model matches REF at K2 but
becomes poor at K4; on the balanced fixture its high-SNR PUPE .36719 slightly exceeds low-SNR .35547.
We do not impose monotonicity on an incorrect approximate posterior. The exact pair/triangle endpoint
tests still pass. Product-prefix decisions agree with the product provider on these tiny banks.
Fourier positive-evidence certification is zero at the K4/4dB cells; its weak incumbent explains its
large error. Joint fields have mean ESS only about 1.86/1.70 out of 512 in balanced/uniform K4/4dB,
so their exact correction does not make this proposal effective at the default budget.

Higher SNR improves empirical PUPE at every tested K for the other paths and all uniform/distinct
curves. PUPE need not increase monotonically with K: per-user energy, collisions and multiset scoring
all matter. Neither smooth trends nor a small-B win over the ODMA adapter proves large-bank recovery.

### Final job and numerical checks

- All **61 regression tests pass**, including the expanded stochastic/load-sharding comparison.
- The actual row CLI and job merger completed **16 rows / 64 frame records**, four stochastic families,
  two loads and two seeds. Seed-level confidence and per-seed matrices survive merging.
- An initial extra balanced B4 fixture at seed3218 was rejected because it leaves a channel resource
  uncovered. This is the intended strict construction guard. The two-seed merge plumbing check was
  explicitly switched to uniform energies; the production manifest's seeds/construction were not changed.
- All **12 actual research encoders** (B12/14 × two mappings × three seeds) build and satisfy unit energy.
- PUPE, runtime, learning and posterior heatmaps were visually inspected, including multi-seed single-SNR
  layout. Frame/seed interval labels, baseline adaptations and the two coding-reference bounds remain visible.
- Python compilation, PBS shell syntax and whitespace checks pass. All generated results stay ignored.

The B14/K26 endpoint check at -4/10 dB exposed a scale issue in the **particle guide setting**, not in
its exact correction: the smoke floor -80 can flatten Gaussian continuation potentials at n256, whose
normalization alone costs hundreds of log units. The research manifest therefore explicitly sets
DEC-07 guide_log_floor=-1000, retaining the same particle count and batch schedule. Tiny smoke defaults
remain -80. This choice follows potential scaling; a one-frame endpoint comparison is numerical
verification, not a performance selection or a new recovery claim. Both settings retain the physical
terminal likelihood and valid, usually vacuous finite-batch bounds.


All **52** numerical endpoint calls (13 new families × two mappings × -4/10 dB, B14/K26) completed
with finite legal outputs. Four additional particle calls verified the research guide floor. On the
single high-SNR observation, the lower floor changed particle PUPE from .96154 (balanced) / 1.0
(uniform) to .03846 in both mappings, consistent with restoring a useful continuation target. This is not a measured error curve;
low-SNR observations remain difficult and all these certificates remain vacuous. The 32-candidate
paths covered only about .269–.538 of the true copies in these probes, so their reported restriction
already precludes strong recovery on those observations. This is visible candidate recall measured
after decoding, not an oracle supplied to the algorithms. These limitations justify retaining coverage,
concentration and runtime alongside PUPE in the full campaign.

Final regression rerun: 61/61 pass after manifest/floor validation. The scripts and complete manifest
are ready to transfer and queue. Scheduler submission and full B12/B14 quality runs are still pending.


## 20 October 2026: reduce scheduler footprint to 90 jobs

The user objected to the 648-job array and requested fewer than 100 jobs to avoid excessive use of
shared university HPC. The preceding 648-row layout is superseded. It came from B12/14 × two mappings
× three seeds × (13 decoder families × four K shards + two baseline rows).

The new manifest has **90 bundles = 2 B × 3 seeds × 15 receivers**. A bundle contains the balanced and
uniform configurations, run sequentially; each configuration now includes all four K values. Both
mappings, three seeds, eight SNRs, 256 frames/cell, decoder budgets and D2 training settings are retained.
The 180 child runs still total **1,474,560 decoder-frame evaluations**. Grouping is not a reduction in
statistical work. A balanced-only campaign is documented as an explicit half-work alternative.

PBS now requests two CPU cores and 16 GB per job, with at most four simultaneous subjobs (`1-90%4`):
at most eight CPU cores and 64 GB reserved by this array. The 72-hour limit is retained because bundles
are longer: two-thread B14 guided-importance endpoint timings suggest about 13–48 hours for both
mappings/all loads, before setup and hardware variation. Grouping two mappings instead of three seeds
keeps the slowest bundles shorter. No scheduler action has been taken.

The runner executes each bundle sequentially and skips already completed child runs only after matching
config/source hashes. Partial or incompatible runs remain explicit errors, never overwritten. The merger
flattens bundles into the same source-checked per-mapping/per-seed analysis. Manifest tests check the
90-job/180-run counts, unchanged evaluation work, source-specific families, budgets and PBS cap. A new
executable test covers both mappings, complete-run retry, incompatible config/source rejection, partial
output protection and merged panels. Decoder algorithms and the previous smoke evidence are unchanged.

Verification: **62 regression tests pass**. Two actual bundled CLI jobs completed four child runs,
merged both mapping panels, and safely skipped both completed children on retry. Python compilation,
PBS shell syntax and whitespace checks pass. The submission command is now
`qsub -J '1-90%4' jobs/035_polar_decoders/035_polar_decoders.sh`; the script embeds the same cap.
