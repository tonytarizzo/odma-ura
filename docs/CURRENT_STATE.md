# ODMA-URA: Neutral Project Handoff

## Research aim

The project asks whether URA codewords can retain the favourable recovery behaviour of a dense random global codebook
while using a compact, structured representation that remains executable when the payload has `B≈100--128` bits. The
target is not merely to compress storage. The encoder must be generated procedurally, satisfy a per-codeword unit-energy
constraint, and admit a decoder that recovers complete unsourced messages from a multiuser superposition.

The empirical lead is that **sparsity itself is not the observed problem**. At `B=12/14`, sparse-global codebooks whose
columns use 25% of resource rows were near dense performance under the same learned decoder, whereas ODMA codebooks with
the same density but only four reused placement masks were substantially worse. The open question is whether the useful
independent-support behaviour can be generated and decoded without storing one support per one of `2^B` messages.

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

## Active next stage: generated hash skeleton

Job `028_hash_skeleton_B14` is implemented and locally certified but has no HPC performance results yet. It tests whether
the arbitrary size-`T` sparse support can be restricted to exactly one resource in each of `T` disjoint tables without
losing the useful explicit-global performance. With `R=n/T=2^r`, the compact candidate is

```text
h_t(w) = A_t w + b_t mod 2,
row_t(w) = tR + integer(h_t(w)),
A_t in GF(2)^(r x B), b_t in GF(2)^r.
```

Every `A_t` has rank `r` for exact bin balance, and the stacked matrix has rank `B` for an injective complete support
tuple. An exact small-`B` collision score enumerates every nonzero XOR difference `d` and measures how many tables obey
`A_t d=0`; the selected family chooses the best of 128 random full-rank banks.

The injective capacity is `T log2(n/T)` bits. Thus job `028`'s `T=16/32` choices are valid for `B=14`; at `B=100,n=256`,
one valid choice is `T=64,R=4`. A local test builds only that compact hash state, checks rank, and generates a few
selected-message supports. It is a regression check against a hidden `2^B` encoder axis, not a decoding result.

The 36-row `B=14,n=256` manifest compares, at `T=16` and `T=32`, four fixed-amplitude families: iid arbitrary sparse,
balanced random tables, random binary linear hash, and geometry-selected binary linear hash. Dense is a contextual
reference. D0/D1, loads, SNRs, training budgets, Gaussian amplitudes, and held-out data are paired as far as the family
comparison permits. The sequential causal questions are table constraint, then linearity, then offline selection.

Only the support rule is compact at this stage. The `B=14` adapter materialises `Phi` so the existing global D0/D1
comparison remains valid; Gaussian amplitude decorations still have one value per message, and scalable candidate
proposal/inversion is deliberately deferred. A favourable result would justify solving those two problems for this
skeleton. An unfavourable balanced-table result would reject the restriction before decoder complexity is added.

Local verification covers exact support and unit energy, GF(2) ranks, support injectivity, procedural/materialised
equality, paired amplitudes, offline selection, all family/decoder paths at `B=8`, and reduced-training D0/D1 rows at the
actual `B=14` manifest size. These are execution checks, not recovery results.

Job `029_joint_encoder_decoder_B14` adds the missing co-adaptation control. It learns dense amplitudes, or amplitudes on
a fixed iid/hash support, jointly with D0 or D1. A gradient mask keeps sparse zeros exact and post-step projection keeps
every column at unit energy. The focused 20-row bank uses only dense, iid sparse, and selected hash at `T=16,32`, two
seeds, and 120 epochs. Pre/post geometry and full loss curves distinguish a support limitation from under-training.
The learned sparse amplitudes are still stored per message, so this tests the model class rather than large-`B` execution.

## Files to inspect next

- Narrative: `docs/reports/01_*.pdf` through `05_*.pdf`.
- Exact global evidence: `results/03_results.md`.
- Framework/sectioned evidence: `results/04_results.md`.
- Current jobs and commands: `docs/EXPERIMENT_BANK.md`, `jobs/README.md`.
- Core implementation: `framework/hash_skeleton.py`, `framework/encoder.py`, `framework/learned_decoders.py`,
  `framework/sectioned.py`, `framework/outer_code.py`, and `framework/outer_decoder.py`.
