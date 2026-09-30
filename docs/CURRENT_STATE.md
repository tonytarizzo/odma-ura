# ODMA-URA: Neutral Project Handoff

## Current next experiment: published end-to-end benchmarks

Job [`032`](../jobs/032_published_baselines/README.md) tests the foundation before adding generator restrictions:
ODMA–polar, dynamic CS and complete two-pass CCS-AMP/BP versus fixed/joint explicit dense and sparse references.
Each published encoder also feeds the same D0/D1 receivers. All D0–D4 have fixed-encoder and joint-learning
dense/sparse references. D3/D4 use 1,024 non-oracle full-alphabet matched-filter proposals and record recall.
Their hard shortlist is not differentiable or guaranteed complete; these are not scalable message searches.

Returned on 30 September: all 80 pilot rows and 24/36 native rows (ODMA and CCS), with matching provenance and complete
metrics. Dynamic CS's 12 rows have no saved SNR cell yet. Pilot selection is ready; the 192-row comparison is deferred.
ODMA's ten-round cap truncates active recovery at K100; local paired replays confirm lost performance. CCS K100 also
needs a receiver-tuning check. Neither is a fully verified paper reproduction. Next: the separate **20-row `032_checks`**
array, covering ODMA cap30/60, focused CCS budgets, and four short native dynamic-CS runtime checks. Dynamic CS can now
cache in float64 to avoid repeated casts, preserving its exact operator; new checks have resumable per-frame output.
Original manifests/results are untouched. Main training remains 200 epochs / patience 10 with best-pair restoration.

Bound analysis now uses Polyanskiy's original numerical recipe: all Gallager p_t terms plus q_1, integrating the
shared-noise user minimum. It remains an achievable-error upper bound, not a converse. Prepared curves cover
B=6/12/14/100/128; distinct and iid cases remain separate. At B12,K22/26 the iid collision correction exceeds 5%,
so this bound cannot certify that target. Nominal-energy native codes do not acquire a peak-power guarantee from
the overlay. See the [job README](../jobs/032_published_baselines/README.md#bounds) for the audit and analysis command.
Do not pull analysis changes into the HPC checkout while arrays run: source fingerprints include benchmark files.

## Research aim

The project asks whether URA codewords can retain the favourable recovery behaviour of a dense random global codebook
while using a compact, structured representation that remains executable when the payload has `B≈100--128` bits. The
target is not merely to compress storage. The encoder must be generated procedurally, satisfy a per-codeword unit-energy
constraint, and admit a decoder that recovers complete unsourced messages from a multiuser superposition.

The empirical lead is that **sparsity itself is not the observed problem**. At `B=12/14`, sparse-global codebooks whose
columns use 25% of resource rows were near dense performance under the same learned decoder, whereas ODMA codebooks with
the same density but only four reused placement masks were substantially worse. The open question is whether the useful
independent-support behaviour can be generated and decoded without storing one support per one of `2^B` messages.

The working research discipline is to preserve the broad explicit model class until a computational obstruction forces
a restriction, then test that restriction against its parent before adding another. Compact storage, cheap forward
generation, isolated-message injectivity, and tractable noisy multiuser inversion are separate requirements. This is why
the current direction returns to global `L=1` sparse geometry rather than treating the unsuccessful sectioned route as
the only scalable option.

## System model and notation

- One payload is `w in {0,1}^B`; hence there are `M=2^B` possible unsourced messages.
- `K_a` active devices independently select payloads. Repeated payloads create a nonnegative integer count vector
  `a in Z_+^M` with `1^T a=K_a`.
- A unit-energy codebook `Phi=[phi_0,...,phi_(M-1)] in C^(n x M)` produces
  `y=Phi a=sum_m a_m phi_m`, followed by the known channel/fading and AWGN used by the scenario.
- PUPE is the primary recovery metric. In the low-collision large-`B` regime it is close to set recovery; at small `B`,
  multiplicities must be handled explicitly.

Binary payload bits index the codeword; they are not binary channel symbols. Current physical codewords are real or
complex vectors, so a channel use can carry amplitude and phase.

## Stage 1: initial ODMA-aware decoders

The original model represented each message as a block/pattern choice and a local codeword. A factor-graph BP/EP decoder
used Gaussian resource-to-variable messages and discrete variable-to-resource posteriors with activity/count priors.
Several damping, activity, count, and fading variants were explored. They established useful algebra and exposed
instability/identifiability issues, but generic global recovery (especially NNOMP with known `K_a`) was a stronger and
cleaner comparison. This motivated studying the codebook/support geometry separately from decoder embellishment.

## Stage 2: explicit global support recovery

Jobs `001--008`, `013--014` show a substantial practical dense-versus-ODMA gap under NNOMP in stressed regimes. Jobs
`012`, `016`, and `017` then supplied the true support and fitted counts by NNLS. Under that oracle the ODMA-minus-dense
gap was only about `-0.01--0.28 dB` across the tested geometries, whereas the non-oracle support-recovery losses were
several dB. The defensible interpretation is that the tested ODMA penalty is mainly a practical support-search problem,
not a demonstrated oracle-geometry or information-theoretic penalty.

## Stage 3: factorised explicit encoder and D0/D1

The global matrix was written as

```text
Phi = sum_l B_l U_l T_l,       B_l = [R_(l,1) C_l | ... | R_(l,Q_l) C_l].
```

`R` places/transforms local codewords, `C` is a local alphabet, `U` selects legal local `(q,v)` atoms, and `T` maps each
global message to one local atom. The implementation applies this factorisation through forward/adjoint operators, so it
does not normally materialise `Phi`. It still keeps `T` and decoder states of length `M`; it is therefore an implicit
matrix implementation, not a `B=100` solution.

D0 unrolls exact data-consistency steps `r=y-Phi a`, `g=Phi^H r` and learns scalar evidence/prior calibration, damping,
and step sizes. D1 adds learned nonlocal context grouped by product factors. Jobs `021--022` found:

- D0 was close to a calibrated matched filter (mean PUPE gain about `0.004`).
- D1 improved high-SNR PUPE by about `0.042` on average at 8/12 dB, but also improved dense controls; a specifically
  factor-aware gain was not isolated. Median runtime was about `5.2x` D0.
- Product sharing was competitive but did not beat dense controls on average; learning `C` was mixed.
- Independent 25%-sparse global supports were within `0.016` PUPE of dense in all four geometries and essentially tied at
  `n=256`. Four-mask ODMA was much worse and left roughly 30% of rows unused.

## Stage 4: scalable section-domain encoder

The scalable representation removes the global message axis. A procedural outer encoder maps bits directly to a legal
path

```text
f_out : {0,1}^B -> X_1 x ... x X_L,       path(w)=(i_1,...,i_L),
```

and local banks `F_l in C^(n_l x N_l)` produce

```text
phi(w) = Q_mix [sqrt(E_1) F_1[:,i_1]; ...; sqrt(E_L) F_L[:,i_L]],
y      = sum_l sqrt(E_l) F_l s_l.
```

Here `s_l in Z_+^(N_l)` is the section occupancy and the executable state is `sum_l N_l`, not `2^B`. Unit-norm local
columns, `sum_l E_l=1`, disjoint latent subspaces, and an orthogonal mixer `Q_mix` guarantee `||phi(w)||_2^2=1` for every
payload, including unseen payloads after training.

The current sparse-linear outer code splits `B` payload bits into information symbols in `Z_(2^J)`, adds fixed parity
symbols, and represents validity by a small modular parity-check matrix `H`: `H x=0 mod 2^J`. Its graph/configuration is
a hyperparameter, not learned; local atom banks and decoder calibration can be learned. At `B=128,J=16`, the default has
eight information and eight parity sections, each of size 65,536: 1,048,576 local states rather than `2^128` states.

Section D0 applies the same residual/adjoint principle locally and uses a Binomial count prior. Modular sum-product BP
passes soft evidence through parity factors. An evaluation-only beam enumerates promising information-symbol choices,
constructs their parity symbols procedurally, and optionally fits multiplicities for `B<=20`.

## What is verified

- Small constructions reproduce the explicit forward and adjoint maps; at `L=1`, global and section-compatible D0 are
  exactly equal in layer logits, soft outputs, hard outputs, and parameter gradients.
- Job `025` independently confirms that equality on the HPC path; the Binomial prior changes PUPE by less than `0.0037`.
- Exact-energy tests cover real, complex, explicit, and implicit-Hadamard local banks.
- Job `024` executes `B=128` with no global message object and energy error below `4.8e-7`.
- Job `026` confirms explicit/section signal equivalence within `6.0e-7` for controlled `L>1` cases.

## What currently fails or remains unproved

- Job `024` is not a successful decoder result: PUPE is 1.0, logits saturate near the lower clamp, support loss is about
  15, and initial gradients are about `1e-14--1e-13`.
- Local section counts do not by themselves associate atoms belonging to the same user. Permuting cross-section pairings
  leaves every `s_l` and hence `y` unchanged. Job `026` verifies this with the identity/no-outer control.
- Triadic parity constraints improve association at low load (`K=9`, high-SNR PUPE 0.414 for the learned row) but degrade
  sharply with occupancy (0.947 by `K=30`). Current marginal BP worsens every reported high-SNR association cell.
- These failures do not prove that all sectioned/procedural encoders exclude an optimum. The current outer graph,
  locally trained encoder, evidence calibration, marginal BP semantics, and beam association are all part of the tested
  system.
- D1 has not yet been rebuilt for the scalable section-domain model.

## Latest completed evidence: sparse-global frontier

Job `027_sparse_density_frontier` is now complete: all 72 rows contain summaries and checkpoints and the strict merger
passes with no completeness notes. The four repaired full-density rows resample exact-zero Gaussian entries and satisfy
the intended support invariant. The seed-2702 full-density reruns are not a clean nested paired endpoint because fixing
the draw changed random-number consumption; the intermediate-support comparisons remain the reliable frontier.

At `B=12,n=256`, D0 remains essentially flat through `s≈48`, while D1's transition occurs at smaller support but is
less precisely located under its shorter training budget. In the final strict aggregate D1 at `s=16` is about `0.0293`
PUPE above dense. At equal density `s=64`, arbitrary sparse global still beats four-mask ODMA by about `0.243` PUPE for
D0 and `0.263` for D1. Mask availability is not the observed bottleneck: support/sign patterns remain distinct well into
the degradation region, while correlation tails rise and active users occupy fewer rows.

This remains an explicit `M=4096` result. Sparse columns are stored in dense tensors and normal D0/D1 score all messages.
It identifies a forgiving model class, not a scalable implementation.

## Generated support: current working design

Jobs `028` and `029` are complete (36/36 and 20/20 runs). At `B=14,n=256,T=16/32`, selected affine-hash
supports are close to arbitrary sparse supports under the tested D0/D1 conditions. We retain one resource per table,
`row_t=tR+integer(A_t w+b_t mod 2)`, with full local rank for bin balance and stacked rank B for support-tuple
injectivity. This supports using the skeleton, not a claim of universally optimal geometry. The small-B best-of-128
selection enumerates XOR differences; the large-B forward check only samples rank-valid maps.

## Amplitudes: coordinate-wise sharing under test

Job `030` is complete, including J=10/12 extensions (64 manifest runs). Whole-vector sharing lost performance as J fell;
that implementation remains for historical reproduction, not the main next direction.

Job `031_coordinate_amplitude_frontier_B14` instead uses
`g_t(w)=V[t,integer(P_t w)]`, with a separate fixed `P_t in GF(2)^(J x B)` per table and
`V in R^(T x 2^J)`. Gather T entries and normalize each message: `alpha=g/||g||`.
J=B recovers arbitrary amplitudes on fixed supports. Binary rank checks prevent avoidable label aliases; centering and
unit-RMS scaling balance each bank row at initialization only. Learning can subsequently change those statistics.

The 72-row batch tests coordinate J=4/8/10, raw-versus-balanced J=4, fixed-versus-joint learning, and matched J=14 parents,
with shared-J4 diagnostic controls, D0/D1 and three seeds. All use at most 120 epochs, patience five, deterministic
calibration/validation and best-checkpoint restoration including epoch zero. Local operator/gradient/energy tests,
12 smoke paths, B=14 row execution and four small learning runs pass; HPC performance is pending.

The procedural encoder has no global message axis and generates sampled B=100 columns with exact energy. The
certification adapter and current D0/D1 still score all `2^B` messages. Scalable noisy multiuser inversion remains open.
See Report 5 for the matrix algebra and `jobs/031_coordinate_amplitude_frontier_B14/README.md` for the exact contract.

## Decoder ladder implemented; comparative evidence pending

D0 remains the historical scalar-calibrated global baseline and D1 its expressive full-message correction. D2 replaces
D0's residual-energy heuristic by an analytic effective variance from physical noise and codebook Gram-row energy,
then uses an interference-cancelled matched-filter statistic, a Bernoulli likelihood ratio, and an exact
expected-cardinality projection. The nominal PGD step cancels from this likelihood ratio, so D2 has no step-size or
spectral-norm pass. It still has an `M=2^B` state, but computes geometry in bounded column chunks without materialising
the full codebook or its `M x M` Gram matrix. Its mean-field variance is exact only under equal, uncorrelated
off-coordinate errors; fixed-`K` projection and iterative observation reuse violate that idealisation.

D3 applies exactly the D2 recurrence to a supplied bounded list and generates only those codewords from message bits.
An exhaustive small-`B` list reproduces D2 numerically, and a `B=100,n=256` test executes without a global message
axis. This is scalable **conditional refinement**, not a solved candidate search. D4 adds a zero-initialised,
permutation-equivariant graph correction over candidate Gram, hash-overlap, amplitude-label, state and uncertainty
features. Its correction is centred over candidates because a shared logit shift is unidentifiable after known-`K`
projection. It equals D3 before training and costs quadratic work in candidate-list size.

All D0--D4 implementations currently know realised `K`. D2--D4 use the negligible-collision Bernoulli model; the exact
section-local Binomial denoiser remains available for collision-rich local states. Smoke and laptop checks show finite
gradients and decreasing validation loss for every decoder, but are implementation checks rather than comparative
research evidence. These earlier D3/D4 checks use oracle-complete lists and are not proposer results; job 032 adds
non-oracle full-alphabet matched-filter proposals and counts their misses in PUPE.

In the deterministic `B=6` laptop check, D1, D3 and D4 stopped after 39, 46 and 33 epochs; D0 and D2 were still
improving at the 120-epoch ceiling. D4 reduced candidate-list validation loss from `0.1377` to `0.1044`, but its final
PUPE (`0.0885`) was slightly worse than D3 (`0.0807`). This tiny oracle-list run is not a ranking result, but it already
shows why loss calibration and PUPE must be reported separately.

The complex channel simulators now use `noise_var = E[|z|^2]` consistently. A pre-existing extra factor of one half in
complex AWGN generation was removed; real-valued experiments are unaffected, while any earlier complex-channel result
should be rerun before being cited.

## Files to inspect next

- Narrative: `docs/reports/01_*.pdf` through `06_*.pdf`.
- Exact global evidence: `results/03_results.md`.
- Framework/sectioned evidence: `results/04_results.md`.
- Current jobs and commands: `docs/EXPERIMENT_BANK.md`, `jobs/README.md`.
- Core implementation: `framework/hash_skeleton.py`, `framework/prototype_amplitudes.py`, `framework/encoder.py`,
  `framework/learned_decoders.py`, `framework/candidate_decoders.py`, `framework/candidate_geometry.py`,
  `framework/sectioned.py`, `framework/outer_code.py`, and `framework/outer_decoder.py`.

The six PDFs now form the detailed chronological record; this file is deliberately only a neutral handoff. Report 1
retains the original BP/EP development, Report 2 the wider decoder algebra and oracle decomposition, Report 3 the full
explicit framework, Report 4 the scalable sectioned construction and failure, and Report 5 the current generated
sparse-global direction. Report 6 defines the D0--D4 ladder and its candidate-search boundary.
