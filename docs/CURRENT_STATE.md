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

## Latest experiment and open directions

Job `027_sparse_density_frontier` returned all 72 array logs but only 68 summaries/checkpoints. Four full-density
seed-2702 rows completed training and evaluation, then failed in posthoc support diagnostics because a finite-precision
Gaussian draw contained one exact zero. The generator now resamples exact zeros, but those four rows must be rerun before
the strict merger passes. The other 68 artifacts have the expected histories/evaluation cells, finite values, paired
D0/D1 codebooks, nested supports, and unit-energy error below `9.54e-7`.

The partial evidence gives a clear qualitative frontier at `B=12,n=256`:

- after including the completed log-only full-density evaluations for context, dense and full sparse-global codebooks
  agree within about `0.0044` PUPE for D0 and `0.0014` for D1;
- D0 is essentially flat through `s=48` (81.25% zeros) and only about `0.012` above dense at `s=32`;
- D1 is within about `0.008` of dense at `s=16` (93.75% zeros), although two-seed variation and a still-decreasing
  20-epoch training loss make this a transition band rather than a precise optimum;
- at equal density `s=64`, sparse global beats four-mask ODMA by `0.243` PUPE with D0 and `0.263` with D1;
- even `s=4` is still better than ODMA, despite using only 1.56% of the resource rows per codeword;
- mask availability is not the observed bottleneck: support/sign patterns are essentially unique through `s=3--4`,
  while degradation begins earlier as correlation tails rise and a `K=30` active set occupies fewer rows.

This is still an explicit `M=4096` experiment. Sparse columns are stored in dense tensors, so it identifies a promising
model class but does not itself remove the `2^B` state or storage.

Reasonable next branches, not predetermined decisions, are:

1. Rerun the four failed full-density controls, then confirm only `s=8,16,32,64` with more seeds/evaluation samples.
2. Test those few supports at one larger explicit payload, preferably `B=14,n=256`, to distinguish an absolute-support
   rule from a density rule before extrapolating.
3. Research compact support generators with balanced row use, controlled intersections/coherence, and an inverse/search
   procedure; compare their geometry and recovery against job `027`, not merely their number of possible masks.
4. In parallel, repair large-alphabet local evidence using scale/clamp/top-`K` diagnostics before another `B=128` run.
5. Treat complete-path association as the sectioned decoder target and retain modular BP only if it beats controlled
   no-BP and exact-reference cases.

## Files to inspect next

- Narrative: `docs/reports/01_*.pdf` through `04_*.pdf`.
- Exact global evidence: `results/03_results.md`.
- Framework/sectioned evidence: `results/04_results.md`.
- Current jobs and commands: `docs/EXPERIMENT_BANK.md`, `jobs/README.md`.
- Core implementation: `framework/encoder.py`, `framework/sectioned.py`, `framework/outer_code.py`,
  `framework/learned_decoders.py`, and `framework/outer_decoder.py`.
